"""Measure a bounded EEG-X configuration without creating a paper or W&B run."""

import argparse
from contextlib import nullcontext, contextmanager
import fcntl
import json
import math
from pathlib import Path
import os
import statistics
import sys
import threading
import time


class BenchmarkUnavailable(RuntimeError):
    """No isolated admission window is available for a performance measurement."""


@contextmanager
def profiling_budget(budget_type, ledger, *, reserve_live, max_seconds, concurrency):
    """Keep a live probe's admission slot until it is removed from the ledger."""
    ledger = authoritative_ledger(ledger)
    if not reserve_live:
        budget = budget_type(ledger, setup_seconds=300, max_concurrent=concurrency)
        with budget.stop_on_signals(), budget.lease():
            yield budget
        return

    class ReservedBudget(budget_type):
        reservation_owner = None
        deadline = None

        @contextmanager
        def _locked(self):
            owner = (os.getpid(), threading.get_ident())
            if self.reservation_owner == owner:
                yield
            else:
                with super()._locked():
                    yield

        def check(self):
            super().check()
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                from pretrain.pipeline.budget import BudgetStop

                raise BudgetStop("Reserved profiling window reached its deadline")

        @contextmanager
        def reservation(self):
            # Frozen paper writers also take this blocking lock. They can keep
            # computing, but transitions/flushes wait rather than failing when
            # a probe uses the temporary fourth slot.
            with super()._locked():
                state = self._read()
                self._settle(state)
                pending = 0.0
                meters = state.get("external_meters", {})
                for path, charged in meters.items():
                    meter = json.loads((Path(path) / "budget.json").read_text())
                    elapsed = 0.0
                    if meter.get("active"):
                        elapsed = max(0, time.perf_counter() - meter["updated_monotonic"])
                    pending += max(0, meter["used_seconds"] + elapsed - charged)
                # Include setup/cleanup headroom and the second device's debit,
                # which the guardian settles once the reservation releases.
                required = (max_seconds + 120) * (1 + len(meters))
                if self.limit - state["used_seconds"] - pending <= required:
                    raise BenchmarkUnavailable(
                        "Insufficient shared budget for a live profiling window"
                    )
                self.reservation_owner = (os.getpid(), threading.get_ident())
                self.deadline = time.perf_counter() + max_seconds
                try:
                    with self.lease():
                        yield self
                finally:
                    self.reservation_owner = None

    budget = ReservedBudget(ledger, setup_seconds=300, max_concurrent=concurrency)
    with budget.stop_on_signals(), budget.reservation():
        yield budget


def authoritative_ledger(ledger):
    """Reject a secondary meter as a separate allowance."""
    ledger = Path(ledger).resolve()
    for parent in ledger.parents:
        state = parent / "budget.json"
        if state.is_file() and str(ledger) in json.loads(state.read_text()).get(
            "external_meters", {}
        ):
            raise ValueError("Benchmark configurations must use the authoritative GPU ledger")
    return ledger


@contextmanager
def benchmark_window(ledger):
    """Exclude queue dispatch for the whole benchmark, before any GPU allocation."""
    ledger = authoritative_ledger(ledger)
    with (ledger / "queues.lock").open("a") as gate:
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BenchmarkUnavailable(
                "A live paper queue is dispatching; benchmark requires an idle window"
            ) from error
        state = json.loads((ledger / "budget.json").read_text())
        meters = [
            ledger / "budget.json",
            *[Path(p) / "budget.json" for p in state.get("external_meters", {})],
        ]
        if any(json.loads(path.read_text()).get("active", False) for path in meters):
            raise BenchmarkUnavailable("GPU work is active; benchmark requires an idle window")
        yield


@contextmanager
def profile_stages(training, torch, budget, *, enabled):
    """Attach timing scopes without changing the training or model source."""
    handles, restore = [], []

    def wrap(owner, attribute, name):
        original = getattr(owner, attribute)

        def measured(*args, **kwargs):
            with torch.profiler.record_function(name):
                return original(*args, **kwargs)

        setattr(owner, attribute, measured)
        restore.append((owner, attribute, original))

    def module_scope(module, label):
        stack = []

        def start(current, inputs):
            name = label() if callable(label) else label
            context = torch.profiler.record_function(name)
            context.__enter__()
            stack.append(context)

        def finish(current, inputs, output):
            if stack:
                stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(start))
        handles.append(module.register_forward_hook(finish, always_call=True))

    class Tracker:
        def parameters(self, model, **kwargs):
            handles.append(model.register_forward_pre_hook(lambda *_: budget.check()))
            if not enabled:
                return
            wrap(model.encoder, "tokenize", "stage/tokenizer")
            module_scope(model.encoder.contex_encoder, "stage/encoder")
            module_scope(model.Predictor, "stage/predictor")
            module_scope(model.Decoder, "stage/decoder")
            wrap(model, "patch_errors", "stage/raw_loss")
            if getattr(model, "hydra_target", None) is not None:
                module_scope(model.hydra_target.transform, "stage/hydra_target")
                module_scope(model.hydra_target.head, "stage/hydra_target_head")
            if model.hydra is not None:
                module_scope(
                    model.hydra,
                    lambda: (
                        "stage/dictionary_prediction"
                        if torch.is_grad_enabled()
                        else "stage/dictionary_target"
                    ),
                )
                backward = []

                def before_backward(*_):
                    context = torch.profiler.record_function("stage/dictionary_backward")
                    context.__enter__()
                    backward.append(context)

                def after_backward(*_):
                    if backward:
                        backward.pop().__exit__(None, None, None)

                handles.append(model.hydra.register_full_backward_pre_hook(before_backward))
                handles.append(model.hydra.register_full_backward_hook(after_backward))

        def log(self, *_):
            pass

        def summary(self, *_):
            pass

        def checkpoint(self, *_args, **_kwargs):
            pass

    if enabled:
        wrap(training, "masks", "stage/masks")
        wrap(training, "to_device", "stage/host_to_device")
        wrap(torch.Tensor, "backward", "stage/backward_total")
        wrap(torch.nn.utils, "clip_grad_norm_", "stage/gradient_clip")
        original_factory = training.optimizer_for

        def optimizer_for(*args, **kwargs):
            optimizer = original_factory(*args, **kwargs)
            wrap(optimizer, "step", "stage/optimizer")
            return optimizer

        training.optimizer_for = optimizer_for
        restore.append((training, "optimizer_for", original_factory))
    try:
        yield Tracker()
    finally:
        for handle in reversed(handles):
            handle.remove()
        for owner, attribute, original in reversed(restore):
            setattr(owner, attribute, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--memory-gib", type=float, default=64)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--reserve-live-slot",
        action="store_true",
        help="Reserve a bounded free slot; paper admission waits until probe cleanup",
    )
    args = parser.parse_args()
    if args.updates < 1 or args.warmup < 0 or not 0 < args.max_seconds <= 600:
        parser.error("Use positive updates, nonnegative warmup and a time limit up to 600 seconds")
    if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error("memory-gib must be positive and finite")
    sys.path.insert(0, str(Path(args.source).resolve()))
    from pretrain.train import read_config

    settings = read_config(args.config)
    if settings["stage"] != "pretrain" or settings["device"] == "cpu" or "seeds" in settings:
        parser.error("Use one pretraining seed on an explicitly selected CUDA device")
    ledger = Path(settings["budget"])
    if not (ledger / "budget.json").is_file():
        parser.error("The existing shared GPU ledger is required")
    if args.reserve_live_slot:
        if settings.get("compile", False):
            parser.error("Live reservations support eager execution only")
        return measure(args, settings, parser)
    try:
        with benchmark_window(ledger):
            return measure(args, settings, parser)
    except BenchmarkUnavailable as error:
        parser.error(str(error))


def measure(args, settings, parser):
    from pretrain.gpu import available_gpu, gpu_snapshot
    from pretrain.train import RUN_KEYS

    ledger = Path(settings["budget"])
    physical = int(settings["device"].split(":")[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu(physical, allow_busy=settings["shared_gpu"])
    before = gpu_snapshot(physical)
    if before["free_mib"] < (args.memory_gib + 2) * 1024:
        parser.error("Not enough free memory for this bounded probe and its headroom")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cache = Path(args.output).resolve().parent / "profile-compile-cache"
    for name, folder in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")]:
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    import torch
    from pretrain.pipeline.budget import Budget, BudgetStop
    from pretrain.pipeline.benchmark import UpdateTimer, BenchmarkComplete
    from pretrain.pipeline.data import Windows
    from pretrain.pipeline.dataset import DatasetReference
    from pretrain.pipeline import training
    from pretrain.metadata import atomic_json
    from pretrain.channel_identity import sha256_file

    torch.set_num_threads(2)
    reference = DatasetReference(
        Path(settings["dataset"]),
        settings.get("seed", 241),
        settings.get("pretraining_scope", "subject_holdout"),
    )
    view = Windows(reference, "train", verify_signals=False)
    config = training.resolve_training_config(
        {
            **{key: value for key, value in settings.items() if key not in RUN_KEYS},
            "purpose": "engineering_configuration_benchmark",
            "probe_interval": 0,
            "diagnostic_test_probes": False,
            "input_samples": view.metadata["window_samples"],
        }
    )
    if (len(view) + config["logical_batch"] - 1) // config[
        "logical_batch"
    ] <= args.updates + args.warmup:
        parser.error("The measured updates must stay within one training epoch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", config)

    class ProfiledTimer(UpdateTimer):
        profiler = None

        def check(self):
            completed = self.started is not None
            try:
                super().check()
            finally:
                if completed and self.profiler is not None:
                    self.profiler.step()

    result = {
        "scope": "configuration capacity/timing; not a code-speedup claim or paper result",
        "config": config,
        "dataset": view.metadata["fingerprint"],
        "physical_gpu": physical,
        "gpu_before": before,
        "shared_gpu": settings["shared_gpu"],
        "memory_limit_gib": args.memory_gib,
        "profile_enabled": args.profile,
        "warmup_updates": args.warmup,
        "requested_measured_updates": args.updates,
        "driver_sha256": sha256_file(Path(__file__)),
    }
    timer = None
    started = time.perf_counter()
    try:
        with profiling_budget(
            Budget,
            ledger,
            reserve_live=args.reserve_live_slot,
            max_seconds=args.max_seconds,
            concurrency=settings["concurrent_gpu_runs"],
        ) as shared:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(args.memory_gib * 2**30 / total)
            torch.cuda.reset_peak_memory_stats()
            timer = ProfiledTimer(
                shared,
                updates=args.updates,
                warmup=args.warmup,
                max_seconds=args.max_seconds,
                synchronize=torch.cuda.synchronize,
            )
            profiler = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    schedule=torch.profiler.schedule(
                        wait=0, warmup=args.warmup, active=args.updates, repeat=1
                    ),
                )
                if args.profile
                else nullcontext()
            )
            with (
                profiler as trace,
                profile_stages(training, torch, shared, enabled=args.profile) as tracker,
            ):
                timer.profiler = trace if args.profile else None
                try:
                    training.run(
                        config,
                        reference,
                        output / "training",
                        device="cuda:0",
                        budget=timer,
                        tracker=tracker,
                    )
                except BenchmarkComplete:
                    result["status"] = "measured"
        if args.profile:
            trace.export_chrome_trace(str(output / "trace.json"))
            averages = trace.key_averages()
            result["operators"] = [
                {
                    "name": event.key,
                    "device_type": str(event.device_type),
                    "calls": event.count,
                    "self_cpu_ms": event.self_cpu_time_total / 1000,
                    "self_cuda_ms": event.self_device_time_total / 1000,
                }
                for event in sorted(
                    averages,
                    key=lambda event: event.self_cpu_time_total,
                    reverse=True,
                )[:40]
            ]
            result["stages"] = [
                {
                    "name": event.key,
                    "device_type": str(event.device_type),
                    "calls": event.count,
                    "cpu_total_ms": event.cpu_time_total / 1000,
                    "cuda_total_ms": event.device_time_total / 1000,
                }
                for event in averages
                if event.key.startswith("stage/") or "DataLoaderIter.__next__" in event.key
            ]
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except BudgetStop as error:
        result.update(status="stopped", error=str(error))
    except Exception as error:
        result.update(status="failed", error=str(error))
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        result["seconds_per_update"] = timer.times[args.warmup :] if timer else []
        if result["seconds_per_update"]:
            median = statistics.median(result["seconds_per_update"])
            result.update(
                median_update_seconds=median, windows_per_second=config["logical_batch"] / median
            )
        atomic_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
    return 0 if result.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
