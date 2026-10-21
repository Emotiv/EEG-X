from pretrain.pipeline import execution
from pretrain.pipeline.study import resumable_checkpoint
from pretrain.metadata import atomic_json
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest


def test_killed_worker_is_terminal_and_keeps_its_output(tmp_path, monkeypatch):
    real_run = subprocess.run

    def killed_worker(command, **kwargs):
        attempt = tmp_path / "jobs" / "killed" / "attempt-0001"
        script = (
            "import json, os, signal; from pathlib import Path; "
            f"p = Path({str(attempt / 'status.json')!r}); "
            "p.write_text(json.dumps({'status': 'running', 'id': 'killed', "
            "'output': str(p.parent / 'run')})); os.kill(os.getpid(), signal.SIGKILL)"
        )
        return real_run([sys.executable, "-c", script], check=False)

    monkeypatch.setattr(execution.subprocess, "run", killed_worker)
    result = execution.launch(tmp_path, {"id": "killed"})
    assert result["status"] == "failed"
    assert "worker" in result["reason"].lower()
    assert result["output"].endswith("attempt-0001/run")
    receipt = tmp_path / "jobs" / "killed" / "attempt-0001" / "status.json"
    assert json.loads(receipt.read_text()) == result


@pytest.mark.parametrize(
    "process,memory,free,utilization,allowed",
    [
        ("tritonserver", 1629, 95622, 0, True),
        ("python", 1629, 95622, 0, False),
        ("tritonserver", 1629, 95622, 30, False),
        ("tritonserver", 3000, 94000, 0, False),
        ("tritonserver", 1629, 16000, 0, False),
    ],
)
def test_selected_gpu_allows_small_idle_service_but_rejects_other_work(
    monkeypatch, process, memory, free, utilization, allowed
):
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        if "--id=1" in command:
            return SimpleNamespace(stdout=f"GPU-selected, {memory}, {free}, {utilization}\n")
        return SimpleNamespace(stdout=f"GPU-other, 100, python\nGPU-selected, 200, {process}\n")

    monkeypatch.setattr(execution.subprocess, "run", query)
    if allowed:
        assert execution.available_gpu(1) == "GPU-selected"
    else:
        with pytest.raises(execution.GpuBusy, match="Physical GPU 1 is busy"):
            execution.available_gpu(1)
    assert "--id=1" in calls[0]


def test_shared_gpu_requires_opt_in_and_free_memory(monkeypatch):
    state = {
        "uuid": "GPU-zero",
        "used_mib": 72000,
        "free_mib": 24000,
        "utilization_percent": 99,
        "other_compute_workload": True,
    }
    monkeypatch.setattr(execution, "gpu_snapshot", lambda gpu: state)
    with pytest.raises(execution.GpuBusy, match="busy"):
        execution.available_gpu(0)
    assert execution.available_gpu(0, allow_busy=True) == "GPU-zero"
    state["free_mib"] = 1024
    with pytest.raises(execution.GpuBusy, match="free memory"):
        execution.available_gpu(0, allow_busy=True)


@pytest.mark.parametrize("other_process", [False, True])
def test_gpu_snapshot_excludes_its_own_training_process(monkeypatch, other_process):
    def query(command, **kwargs):
        if "--id=0" in command:
            return SimpleNamespace(stdout="GPU-zero, 4000, 93000, 50\n")
        rows = "GPU-zero, 123, python\nGPU-zero, 456, tritonserver\n"
        if other_process:
            rows += "GPU-zero, 789, python\n"
        return SimpleNamespace(stdout=rows)

    monkeypatch.setattr(execution.os, "getpid", lambda: 123)
    monkeypatch.setattr(execution.subprocess, "run", query)
    assert execution.gpu_snapshot(0)["other_compute_workload"] is other_process


def test_wait_applies_to_every_job_and_busy_device_races_are_retryable(tmp_path, monkeypatch):
    jobs = [{"id": "one"}, {"id": "two"}]
    waits, launched = [], []
    complete = set()
    replies = iter([{"status": "deferred"}, {"status": "complete"}, {"status": "complete"}])
    monkeypatch.setattr(execution, "read_plan", lambda root: {"physical_gpu": 1})
    monkeypatch.setattr(
        execution, "runnable", lambda plan, root: (job for job in jobs if job["id"] not in complete)
    )
    monkeypatch.setattr(
        execution, "wait_available", lambda wait, physical_gpu: waits.append((wait, physical_gpu))
    )
    monkeypatch.setattr("pretrain.pipeline.reporting.report", lambda root: None)

    def launch(root, job):
        launched.append(job["id"])
        status = next(replies)
        if status["status"] == "complete":
            complete.add(job["id"])
        return status

    monkeypatch.setattr(execution, "launch", launch)
    execution.execute(tmp_path, wait_for_gpu=True)
    assert launched == ["one", "one", "two"]
    assert waits == [(True, 1), (True, 1), (True, 1)]


def test_busy_device_deferral_preserves_previous_safe_resume(tmp_path):
    original = tmp_path / "jobs" / "one" / "attempt-0001"
    (original / "run").mkdir(parents=True)
    checkpoint = original / "run" / "last.pth"
    checkpoint.write_bytes(b"saved_optimizer_boundary")
    atomic_json(
        original / "status.json", {"status": "budget_stopped", "output": str(original / "run")}
    )
    for number in (2, 3):
        deferred = original.parent / f"attempt-{number:04d}"
        deferred.mkdir()
        atomic_json(deferred / "status.json", {"status": "deferred"})
    assert resumable_checkpoint(tmp_path, "one") == checkpoint


def test_preparation_finishing_during_queue_scan_does_not_leave_ready_jobs(tmp_path, monkeypatch):
    reads, done = [], []

    def read(root):
        reads.append(True)
        state = "preparing" if len(reads) == 1 else "ready"
        return {"physical_gpu": 1, "datasets": {"tuab": {"status": state}}}

    monkeypatch.setattr(execution, "read_plan", read)
    monkeypatch.setattr(
        execution,
        "runnable",
        lambda plan, root: iter(
            [{"id": "tuab"}] if plan["datasets"]["tuab"]["status"] == "ready" and not done else []
        ),
    )
    monkeypatch.setattr(execution, "wait_available", lambda wait, physical_gpu: None)
    monkeypatch.setattr("pretrain.pipeline.reporting.report", lambda root: None)

    def launch(root, job):
        done.append(job["id"])
        return {"status": "complete"}

    monkeypatch.setattr(execution, "launch", launch)
    execution.execute(tmp_path, wait_for_gpu=True)
    assert done == ["tuab"]
