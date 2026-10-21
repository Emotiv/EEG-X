"""Sequential isolated screen jobs on the selected GPU, with a shared ledger."""

import json
import os
import signal
import subprocess
import traceback
import time
from pathlib import Path

from pretrain.pipeline.budget import Budget, BudgetStop
from pretrain.pipeline.study import (
    latest,
    read_plan,
    resumable_checkpoint,
    runnable,
    verify_dataset,
    dataset_for,
)
from pretrain.metadata import atomic_json


class GpuBusy(RuntimeError):
    pass


def gpu_snapshot(physical_gpu):
    """Read device load without allocating CUDA memory or changing workloads."""
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    uuid, memory, free, utilization = [item.strip() for item in result.stdout.strip().split(",")]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    # The host keeps a small Triton service on both cards. Its idle reservation
    # is compatible with training; active workloads and other owners still wait.
    busy = any(
        device.strip() == uuid
        and pid.strip() != str(os.getpid())
        and Path(name.strip()).name != "tritonserver"
        for device, pid, name in (row.split(",", 2) for row in processes.stdout.splitlines())
    )
    return {
        "uuid": uuid,
        "used_mib": int(memory),
        "free_mib": int(free),
        "utilization_percent": int(utilization),
        "other_compute_workload": busy,
    }


def available_gpu(physical_gpu=0, *, allow_busy=False):
    state = gpu_snapshot(physical_gpu)
    if allow_busy:
        if state["free_mib"] < 2048:
            raise GpuBusy(f"Physical GPU {physical_gpu} lacks free memory for a shared check")
        return state["uuid"]
    if (
        state["other_compute_workload"]
        or state["used_mib"] > 2048
        or state["free_mib"] < 32768
        or state["utilization_percent"] > 5
    ):
        raise GpuBusy(f"Physical GPU {physical_gpu} is busy; leave unrelated work running")
    return state["uuid"]


def next_attempt(root, job):
    folder = Path(root) / "jobs" / job
    folder.mkdir(parents=True, exist_ok=True)
    number = len(list(folder.glob("attempt-*"))) + 1
    attempt = folder / f"attempt-{number:04d}"
    attempt.mkdir()
    return attempt


def worker(root, job_id, attempt):
    plan = read_plan(root)
    job = next(item for item in plan["jobs"] if item["id"] == job_id)
    attempt = Path(attempt).resolve()
    resume = resumable_checkpoint(root, job_id)
    run = attempt / "run"
    status = {"status": "running", "id": job_id, "output": str(run)}
    atomic_json(attempt / "status.json", status)
    budget = Budget(
        plan["budget_root"],
        gpu_hours=plan["gpu_hours"],
        setup_seconds=plan.get("setup_seconds", 300),
    )
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: setattr(budget, "stop_requested", True))
    try:
        verify_dataset(plan, job["task"], job["config"]["seed"])
        # Recheck immediately before allocating the device. The controller never allocates CUDA.
        uuid = available_gpu(plan["physical_gpu"])
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for key, folder in {
            "TMPDIR": "tmp",
            "TRITON_CACHE_DIR": "triton",
            "TORCHINDUCTOR_CACHE_DIR": "inductor",
        }.items():
            path = Path(root).resolve() / "cache" / folder
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        with budget.lease():
            import torch

            from pretrain.pipeline import finetuning, frozen, training

            torch.set_num_threads(4)
            manifest = dataset_for(plan, job["task"], job["config"]["seed"])["manifest"]
            config = job["config"]
            kwargs = {"device": "cuda:0", "budget": budget}
            if resume is not None and job["mode"] != "frozen":
                kwargs["resume"] = resume
            if job["mode"] == "pretrain":
                checkpoint = training.run(config, manifest, run, **kwargs)
                from pretrain.pipeline.exports import export

                export(checkpoint, manifest, run / "release")
            else:
                dependency = latest(root, job["depends_on"])
                if not dependency or dependency["status"] != "complete":
                    raise ValueError("Pretraining dependency is not complete")
                checkpoint = Path(dependency["output"]) / "best.pth"
                module = frozen if job["mode"] == "frozen" else finetuning
                module.run(config, manifest, checkpoint, run, **kwargs)
            budget.check()
            status["status"] = "complete"
    except GpuBusy as error:
        status.update(status="deferred", reason=str(error))
    except BudgetStop as error:
        status.update(status="budget_stopped", reason=str(error))
    except Exception as error:
        status.update(status="failed", reason=str(error), traceback=traceback.format_exc())
    finally:
        atomic_json(attempt / "status.json", status)
    return status


def wait_available(wait, physical_gpu=0):
    while True:
        try:
            available_gpu(physical_gpu)
            break
        except GpuBusy as error:
            if not wait:
                raise
            print(str(error), flush=True)
            time.sleep(60)


def execute(root, *, wait_for_gpu=False):
    root = Path(root).resolve()
    from pretrain.pipeline.reporting import report

    while True:
        plan = read_plan(root)
        report(root)
        for job in runnable(plan, root):
            while True:
                wait_available(wait_for_gpu, plan["physical_gpu"])
                status = launch(root, job)
                report(root)
                if status["status"] != "deferred" or not wait_for_gpu:
                    break
            if status["status"] != "complete":
                return
        current = read_plan(root)
        if any(runnable(current, root)):
            continue
        if not wait_for_gpu or not any(
            item["status"] == "preparing" for item in current.get("datasets", {}).values()
        ):
            break
        print("Waiting for source-backed preparation; GPU is not reserved", flush=True)
        time.sleep(60)


def launch(root, job):
    attempt = next_attempt(root, job["id"])
    session = f"eegx-{root.name}-{job['id']}-{attempt.name}"
    command = [
        "screen",
        "-L",
        "-Logfile",
        str(attempt / "console.log"),
        "-D",
        "-m",
        "-S",
        session,
        "uv",
        "run",
        "--locked",
        "--no-active",
        "python",
        "-m",
        "pretrain",
        "worker",
        "--study",
        str(root),
        "--job",
        job["id"],
        "--attempt",
        str(attempt),
    ]
    atomic_json(attempt / "launch.json", {"screen": session, "command": command})
    # -D -m does not fork. Completion of this process is the job completion event.
    completed = subprocess.run(command, check=False)
    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"id": job["id"]}
    terminal = {"complete", "failed", "deferred", "budget_stopped"}
    if status.get("status") not in terminal or (
        status.get("status") == "complete" and completed.returncode != 0
    ):
        status.update(
            status="failed",
            reason=f"Screen exited without a valid terminal worker receipt: {completed.returncode}",
        )
    status["screen_returncode"] = completed.returncode
    atomic_json(status_path, status)
    print(json.dumps(status), flush=True)
    return status
