from types import SimpleNamespace

import pytest

from pretrain import gpu


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

    monkeypatch.setattr(gpu.subprocess, "run", query)
    if allowed:
        assert gpu.available_gpu(1) == "GPU-selected"
    else:
        with pytest.raises(gpu.GpuBusy, match="Physical GPU 1 is busy"):
            gpu.available_gpu(1)
    assert "--id=1" in calls[0]


def test_shared_gpu_requires_opt_in_and_free_memory(monkeypatch):
    state = {
        "uuid": "GPU-zero",
        "used_mib": 72000,
        "free_mib": 24000,
        "utilization_percent": 99,
        "other_compute_workload": True,
    }
    monkeypatch.setattr(gpu, "gpu_snapshot", lambda gpu: state)
    with pytest.raises(gpu.GpuBusy, match="busy"):
        gpu.available_gpu(0)
    assert gpu.available_gpu(0, allow_busy=True) == "GPU-zero"
    state["free_mib"] = 1024
    with pytest.raises(gpu.GpuBusy, match="free memory"):
        gpu.available_gpu(0, allow_busy=True)


@pytest.mark.parametrize("other_process", [False, True])
def test_gpu_snapshot_excludes_its_own_training_process(monkeypatch, other_process):
    def query(command, **kwargs):
        if "--id=0" in command:
            return SimpleNamespace(stdout="GPU-zero, 4000, 93000, 50\n")
        rows = "GPU-zero, 123, python\nGPU-zero, 456, tritonserver\n"
        if other_process:
            rows += "GPU-zero, 789, python\n"
        return SimpleNamespace(stdout=rows)

    monkeypatch.setattr(gpu.os, "getpid", lambda: 123)
    monkeypatch.setattr(gpu.subprocess, "run", query)
    assert gpu.gpu_snapshot(0)["other_compute_workload"] is other_process
