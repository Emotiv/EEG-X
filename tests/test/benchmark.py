import json
import signal
import copy

import pytest

from pretrain.pipeline.benchmark import (
    BenchmarkComplete,
    UpdateTimer,
    benchmark,
    compare_benchmarks,
)
from pretrain.pipeline.budget import Budget, BudgetStop


def benchmark_result():
    return {
        "config": {"logical_batch": 64, "microbatch": 64, "precision": "bf16"},
        "dataset": "same-training-windows",
        "physical_gpu": 1,
        "gpu_uuid": "GPU-one",
        "gpu_name": "same-device",
        "warmup_updates": 5,
        "software": {"torch": "same", "python": "same", "uv_lock": "same"},
        "model_contract": {"name": "same", "geometry": [192, 11, 3, 504]},
        "cuda_version": "same",
        "cudnn_version": "same",
        "benchmark_sha256": "same",
        "profile_enabled": False,
        "seconds_per_update": [1.0, 1.1, 0.9],
    }


def test_comparison_accepts_only_fixed_workload_without_accumulation():
    before = benchmark_result()
    after = copy.deepcopy(before)
    after["seconds_per_update"] = [0.5, 0.55, 0.45]
    assert compare_benchmarks(before, after)["speedup"] == 2
    after["config"]["microbatch"] = 16
    with pytest.raises(ValueError, match="configurations differ"):
        compare_benchmarks(before, after)
    before["config"]["microbatch"] = 16
    with pytest.raises(ValueError, match="one physical batch"):
        compare_benchmarks(before, after)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "different"),
        ("gpu_uuid", "different"),
        ("warmup_updates", 2),
        ("profile_enabled", True),
        ("seconds_per_update", [1.0]),
        ("model_contract", {"name": "different"}),
        ("benchmark_sha256", "different"),
    ],
)
def test_comparison_rejects_changed_measurement_conditions(field, value):
    before = benchmark_result()
    after = copy.deepcopy(before)
    after[field] = value
    with pytest.raises(ValueError):
        compare_benchmarks(before, after)


def test_benchmark_counts_complete_updates_and_preserves_shared_usage(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("pretrain.pipeline.benchmark.time.perf_counter", lambda: clock[0])
    shared = Budget(tmp_path, gpu_hours=1)
    with shared.lease():
        timer = UpdateTimer(shared, updates=2, warmup=1, max_seconds=10, synchronize=lambda: None)
        timer.check()
        for _ in range(2):
            clock[0] += 0.5
            timer.check()
        clock[0] += 0.5
        with pytest.raises(BenchmarkComplete):
            timer.check()
        assert timer.times == [0.5, 0.5, 0.5]
    state = json.loads((tmp_path / "budget.json").read_text())
    assert state["used_seconds"] == 1.5
    assert not state["active"]


def test_benchmark_time_limit_stops_at_an_update_boundary(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("pretrain.pipeline.budget.time.perf_counter", lambda: clock[0])
    with Budget(tmp_path, gpu_hours=1).lease() as shared:
        timer = UpdateTimer(shared, updates=20, warmup=1, max_seconds=1, synchronize=lambda: None)
        timer.check()
        clock[0] += 2
        with pytest.raises(BudgetStop, match="time limit"):
            timer.check()


def test_benchmark_cannot_start_a_new_uncharged_study(tmp_path):
    with pytest.raises(ValueError, match="existing shared study budget"):
        benchmark({}, "unused", tmp_path / "output", tmp_path / "missing", gpu=1)


def test_benchmark_rejects_accumulation_before_selecting_gpu(tmp_path, monkeypatch):
    (tmp_path / "budget.json").write_text("{}")

    def unexpected_selection(*_):
        raise AssertionError("Batch contract must be checked before selecting a GPU")

    monkeypatch.setattr("pretrain.pipeline.benchmark.available_gpu", unexpected_selection)
    with pytest.raises(ValueError, match="one physical batch"):
        benchmark(
            {"logical_batch": 64, "microbatch": 4}, "unused", tmp_path / "out", tmp_path, gpu=1
        )


@pytest.mark.parametrize("gpu", [-1, True, "0"])
def test_benchmark_rejects_an_invalid_device_before_any_gpu_query(tmp_path, gpu):
    with pytest.raises(ValueError, match="nonnegative integer"):
        benchmark({}, "unused", tmp_path / "out", tmp_path, gpu=gpu)


def test_gpu_zero_reaches_the_availability_check(tmp_path, monkeypatch):
    from pretrain.pipeline.execution import GpuBusy

    (tmp_path / "budget.json").write_text("{}")
    selected = []

    def unavailable(gpu):
        selected.append(gpu)
        raise GpuBusy("busy device")

    monkeypatch.setattr("pretrain.pipeline.benchmark.available_gpu", unavailable)
    with pytest.raises(GpuBusy, match="busy device"):
        benchmark(
            {"logical_batch": 4, "microbatch": 4}, "unused", tmp_path / "out", tmp_path, gpu=0
        )
    assert selected == [0]


def test_shared_gpu_timings_cannot_establish_a_code_speedup():
    before, after = benchmark_result(), benchmark_result()
    after["allow_busy"] = True
    with pytest.raises(ValueError, match="Shared-GPU"):
        compare_benchmarks(before, after)


def test_shared_gpu_requires_a_memory_limit_before_querying_hardware(tmp_path):
    with pytest.raises(ValueError, match="explicit memory limit"):
        benchmark({}, "unused", tmp_path / "out", tmp_path, gpu=0, allow_busy=True)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_benchmark_stop_signals_request_a_safe_boundary_and_restore_handlers(tmp_path, signum):
    previous = signal.getsignal(signum)
    shared = Budget(tmp_path, gpu_hours=1)
    with shared.lease(), shared.stop_on_signals():
        signal.raise_signal(signum)
        with pytest.raises(BudgetStop):
            shared.check()
    assert signal.getsignal(signum) == previous
    assert not json.loads((tmp_path / "budget.json").read_text())["active"]
