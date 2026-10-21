"""Measure bounded real-data optimizer updates through the training pipeline."""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.budget import Budget, BudgetStop
from pretrain.pipeline.execution import available_gpu, gpu_snapshot
from pretrain.metadata import atomic_json


class BenchmarkComplete(BudgetStop):
    """The requested number of optimizer updates has completed."""


class UpdateTimer:
    """Time complete logical updates, including input loading and optimizer work."""

    def __init__(self, shared, *, updates, warmup, max_seconds, synchronize):
        self.shared = shared
        self.target = updates + warmup
        self.max_seconds = max_seconds
        self.synchronize = synchronize
        self.case_start = shared.elapsed
        self.started = None
        self.times = []

    def check(self):
        self.shared.check()
        self.synchronize()
        now = time.perf_counter()
        if self.started is not None:
            self.times.append(now - self.started)
        if len(self.times) >= self.target:
            raise BenchmarkComplete("Benchmark optimizer update count reached")
        if self.shared.elapsed - self.case_start >= self.max_seconds:
            raise BudgetStop("Benchmark time limit reached")
        self.shared.flush()
        self.started = time.perf_counter()

    def flush(self):
        self.shared.flush()


def _fixed_batch_size(config):
    for key in ("logical_batch", "microbatch"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if config["logical_batch"] != config["microbatch"]:
        raise ValueError(
            "Benchmark requires one physical batch: logical_batch must equal microbatch"
        )
    return config["logical_batch"]


def compare_benchmarks(before, after):
    """Compare code changes with identical workloads and one physical batch."""
    if before.get("allow_busy") or after.get("allow_busy"):
        raise ValueError("Shared-GPU timings cannot establish a code speedup")
    if before["config"] != after["config"]:
        raise ValueError(
            "Benchmark configurations differ; batch sizes and workload must stay fixed"
        )
    config = before["config"]
    _fixed_batch_size(config)
    for field in (
        "dataset",
        "physical_gpu",
        "gpu_uuid",
        "gpu_name",
        "warmup_updates",
        "model_contract",
        "cuda_version",
        "cudnn_version",
        "benchmark_sha256",
    ):
        if before[field] != after[field]:
            raise ValueError(f"Benchmark comparison differs: {field}")
    for field in ("torch", "python", "uv_lock"):
        if before["software"][field] != after["software"][field]:
            raise ValueError(f"Benchmark environment differs: {field}")
    if before["profile_enabled"] or after["profile_enabled"]:
        raise ValueError("Disable profiling for throughput comparisons")
    left, right = before["seconds_per_update"], after["seconds_per_update"]
    if (
        not left
        or len(left) != len(right)
        or not all(math.isfinite(value) and value > 0 for value in [*left, *right])
    ):
        raise ValueError("Benchmark timings must be positive with equal measured update counts")
    baseline, candidate = statistics.median(left), statistics.median(right)
    return {
        "config": config,
        "dataset": before["dataset"],
        "physical_batch": config["microbatch"],
        "measured_updates": len(left),
        "before_seconds_per_update": baseline,
        "after_seconds_per_update": candidate,
        "before_windows_per_second": config["logical_batch"] / baseline,
        "after_windows_per_second": config["logical_batch"] / candidate,
        "speedup": baseline / candidate,
    }


def benchmark(
    config,
    manifest,
    output,
    budget_root,
    *,
    gpu,
    updates=5,
    warmup=2,
    max_seconds=300,
    profile=False,
    allow_busy=False,
    max_memory_gib=None,
):
    """Keep the original training path and stop at a safe optimizer boundary."""
    if type(gpu) is not int or gpu < 0:
        raise ValueError("Physical GPU index must be a nonnegative integer")
    if max_memory_gib is not None and (
        isinstance(max_memory_gib, bool) or not math.isfinite(max_memory_gib) or max_memory_gib <= 0
    ):
        raise ValueError("Memory limit must be positive and finite")
    if allow_busy and max_memory_gib is None:
        raise ValueError("Shared-GPU checks require an explicit memory limit")
    if updates < 1 or warmup < 0 or not 0 < max_seconds <= 600:
        raise ValueError("Invalid benchmark update count, warmup, or time limit")
    budget_root = Path(budget_root).resolve()
    if not (budget_root / "budget.json").is_file():
        raise ValueError("Use the existing shared study budget; do not reset charged work")
    _fixed_batch_size(config)
    uuid = available_gpu(gpu, allow_busy=True) if allow_busy else available_gpu(gpu)
    before = gpu_snapshot(gpu)
    if allow_busy and before["free_mib"] < (max_memory_gib + 1) * 1024:
        raise ValueError("Shared-GPU check requires free memory beyond its allocator limit")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    from contextlib import nullcontext
    from pretrain.pipeline import training
    from pretrain.models.factory import resolve_model_config
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.pipeline.data import Windows

    if torch.cuda.is_initialized():
        raise ValueError("Start a fresh process so physical GPU selection is unambiguous")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key, folder in {
        "TMPDIR": "tmp",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
    }.items():
        path = output / "cache" / folder
        path.mkdir(parents=True)
        os.environ[key] = str(path)
    torch.set_num_threads(4)
    config = resolve_model_config({
        **config,
        "purpose": "performance_benchmark",
        "gpu_sharing": "shared" if allow_busy else "exclusive",
        "gpu_memory_limit_gib": max_memory_gib,
    })
    view = Windows(manifest, "train", verify_signals=False)
    logical_batch = config["logical_batch"]
    if (len(view) + logical_batch - 1) // logical_batch <= updates + warmup:
        raise ValueError("Dataset needs another training batch after the measured updates")
    source = software_contract()
    shared = Budget(budget_root, setup_seconds=300)
    with shared.lease(), shared.stop_on_signals():
        started = shared.elapsed
        if max_memory_gib is not None:
            total = torch.cuda.get_device_properties(0).total_memory
            fraction = max_memory_gib * 2**30 / total
            if fraction > 1:
                raise ValueError("Allocator limit exceeds physical GPU capacity")
            torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)
        timer = UpdateTimer(
            shared,
            updates=updates,
            warmup=warmup,
            max_seconds=max_seconds,
            synchronize=torch.cuda.synchronize,
        )
        torch.cuda.reset_peak_memory_stats()
        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if profile
            else nullcontext()
        )
        with profiler:
            try:
                training.run(config, manifest, output / "run", device="cuda:0", budget=timer)
            except BenchmarkComplete:
                pass
        times = timer.times[warmup:]
        if len(times) != updates:
            raise RuntimeError("Benchmark did not complete the requested optimizer updates")
        checkpoint = torch.load(output / "run" / "last.pth", map_location="cpu", weights_only=True)
        result = {
            "scope": "bounded training benchmark, not paper evaluation",
            "config": config,
            "software": source,
            "model_contract": checkpoint["model"]["_extra_state"],
            "benchmark_sha256": sha256_file(Path(__file__)),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "dataset": view.metadata["fingerprint"],
            "physical_gpu": gpu,
            "allow_busy": allow_busy,
            "allocator_limit_gib": max_memory_gib,
            "gpu_before": before,
            "gpu_after": gpu_snapshot(gpu),
            "attention_backends": json.loads((output / "run" / "attention.json").read_text()),
            "gpu_uuid": uuid,
            "gpu_name": torch.cuda.get_device_name(),
            "warmup_updates": warmup,
            "seconds_per_update": times,
            "median_seconds_per_update": statistics.median(times),
            "windows_per_second": logical_batch / statistics.median(times),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "charged_seconds": shared.elapsed - started,
            "profile_enabled": profile,
        }
        if profile:
            events = sorted(
                profiler.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
            )
            result["profile"] = [
                {
                    "name": e.key,
                    "cpu_us": e.self_cpu_time_total,
                    "gpu_us": e.self_device_time_total,
                    "calls": e.count,
                }
                for e in events[:25]
            ]
            result["attention_operators"] = [e.key for e in events if "attention" in e.key]
        atomic_json(output / "benchmark.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", help="Existing shared study ledger directory")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="Explicitly allow a shared-GPU check; timings cannot establish speedup",
    )
    parser.add_argument(
        "--max-memory-gib", type=float, help="Cap this process's PyTorch CUDA allocator"
    )
    args = parser.parse_args()
    if args.compare:
        result = compare_benchmarks(*(json.loads(Path(path).read_text()) for path in args.compare))
        atomic_json(Path(args.output), result)
        print(json.dumps(result), flush=True)
        return
    if not args.config or not args.manifest or not args.budget or args.gpu is None:
        parser.error("Training benchmarks require --config, --manifest, --budget, and --gpu")
    if not os.environ.get("STY"):
        parser.error("Run the benchmark in its own screen session")
    result = benchmark(
        json.loads(Path(args.config).read_text()),
        args.manifest,
        args.output,
        args.budget,
        gpu=args.gpu,
        updates=args.updates,
        warmup=args.warmup,
        max_seconds=args.max_seconds,
        profile=args.profile,
        allow_busy=args.allow_busy,
        max_memory_gib=args.max_memory_gib,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "median_seconds_per_update",
                    "windows_per_second",
                    "peak_allocated_gib",
                    "charged_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
