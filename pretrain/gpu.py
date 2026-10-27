"""Read GPU availability without changing other workloads."""

import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import time


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


@contextmanager
def record_runtime(output, physical_gpu, before, tracker=None):
    """Measure this process's CUDA allocation separately from shared device use."""
    import torch

    from pretrain.metadata import atomic_json

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        yield
    finally:
        values = {
            "runtime/wall_seconds": time.perf_counter() - started,
            "runtime/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "runtime/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        atomic_json(
            Path(output) / "gpu.json",
            {
                "physical_gpu": physical_gpu,
                "before": before,
                "after": gpu_snapshot(physical_gpu),
                **values,
            },
        )
        if tracker:
            tracker.summary(values)
