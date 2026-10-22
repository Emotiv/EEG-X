"""Compare original and current EEG-X model training costs on prepared EEG.

This is a shared-GPU raw-MSE diagnostic, not a matched code-speed benchmark.
The original release lacks its imported DiCT implementation. Both models use
the same full-waveform raw MSE and AdamW here, rather than either full recipe.
Run with the analysis extra, which supplies pandas for the original model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_REVISION = "be72d06"
ORIGINAL_FILES = (
    "Models/EEG_X.py",
    "Models/Attention.py",
    "Models/position_embedding.py",
    "notebooks/electrode_positions.csv",
)
TASKS = {"dreamer": "dreamer-arousal-v1", "tuab": "tuab-v1"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_source(output):
    source = output / "original"
    for name in ORIGINAL_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            subprocess.check_output(["git", "show", f"{ORIGINAL_REVISION}:{name}"], cwd=ROOT)
        )
    (source / "Models/__init__.py").touch()
    return source


def measure(args):
    sys.path.insert(0, str(ROOT))
    from pretrain.gpu import available_gpu, gpu_snapshot
    from pretrain.pipeline.budget import Budget

    if not (args.budget / "budget.json").is_file():
        raise ValueError("An existing shared GPU budget is required")
    uuid = available_gpu(args.gpu, allow_busy=True)
    before = gpu_snapshot(args.gpu)
    if before["free_mib"] < (args.memory_gib + 1) * 1024:
        raise ValueError("Insufficient free memory beyond the allocator cap")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pretrain.compute import autocast_context, configure_compute
    from pretrain.metadata import atomic_json
    from pretrain.models.factory import attention_backends, create_model, masks
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.pipeline.data import Windows, batches, to_device

    torch.set_num_threads(4)
    random.seed(241)
    np.random.seed(241)
    torch.manual_seed(241)
    manifest = args.data / TASKS[args.task] / "manifest.json"
    view = Windows(manifest, "train", verify_signals=True)
    samples, channels = view.metadata["window_samples"], len(view.ids)
    config = {
        "depth": 8,
        "width": 64,
        "heads": 8,
        "input_samples": samples,
        "seed": 241,
        "objective": None,
        "precision": "bf16",
        "deterministic": False,
    }
    source = original_source(args.output)
    software = software_contract()
    case = args.output / f"{args.task}-{args.trial}-{args.variant}"
    case.mkdir(exist_ok=False)
    budget = Budget(args.budget)
    with budget.lease(), budget.stop_on_signals():
        charged_start = budget.elapsed
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_gib * 2**30 / total_memory)
        device = torch.device("cuda:0")
        if args.variant == "before":
            sys.path.insert(0, str(source))
            from Models.EEG_X import EEG_X

            legacy = {
                "Data_shape": [len(view), channels, samples],
                "emb_size": 64,
                "layers": 8,
                "num_heads": 8,
                "dim_ff": 256,
                "dropout": 0.1,
                "patch_size": 128,
                "patch_stride": 32,
                "pre_layers": 2,
                "mask_ratio": 0.5,
                "momentum": 0.99,
                "device": "cuda:0",
                "num_labels": len(view.metadata["classes"]),
            }
            previous = Path.cwd()
            try:
                os.chdir(source)
                model = EEG_X(legacy).to(device).train()
            finally:
                os.chdir(previous)
            model.copy_weight()
            optimized_modules = (
                model.InputEmbedding,
                model.contex_encoder,
                model.Predictor,
                model.Decoder,
            )
            parameters = [p for module in optimized_modules for p in module.parameters()]
            contract = legacy
        else:
            model = create_model(config).to(device).train()
            parameters = list(model.parameters())
            contract = model.contract
        optimizer = torch.optim.AdamW(parameters, lr=0.00024, weight_decay=0.01)
        loader = iter(
            batches(view, batch_size=4, seed=241, workers=2, pin_memory=True, shuffle=True)
        )
        timings, identities, losses = [], [], []
        with configure_compute(config, device):
            for update in range(args.warmup + args.updates):
                budget.check()
                if budget.elapsed - charged_start > 300:
                    raise RuntimeError("Case exceeded the bounded GPU allowance")
                if update == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                batch = to_device(next(loader), device)
                model.zero_grad(set_to_none=True)
                with autocast_context(device, "bf16"):
                    if args.variant == "before":
                        _, _, _, reconstruction = model.pretrain_forward(batch["eeg"], batch["eeg"])
                    else:
                        mask = masks(
                            batch["window_id"], batch["channel_ids"], samples=samples, seed=241
                        ).to(device)
                        prediction, _ = model.predict_masked(
                            batch["eeg"], batch["channel_ids"], mask=mask
                        )
                        reconstruction = prediction.reshape(4, samples // 32, channels, 32)
                        reconstruction = reconstruction.permute(0, 2, 1, 3).reshape(
                            4, channels, samples
                        )
                    loss = F.mse_loss(reconstruction.float(), batch["eeg"].float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                if args.variant == "before":
                    model.momentum_update()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if update >= args.warmup:
                    timings.append(elapsed)
                identities.extend(batch["window_id"])
                losses.append(float(loss.detach()))
                budget.flush()
        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        if not all(np.isfinite(losses)) or not any(p.grad is not None for p in parameters):
            raise RuntimeError("Training diagnostic lacks finite loss or parameter gradients")
        if not all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
            raise RuntimeError("Training diagnostic has nonfinite parameter gradients")
        result = {
            "scope": "original versus current EEG-X; shared-GPU raw-MSE model diagnostic",
            "variant": args.variant,
            "task": args.task,
            "trial": args.trial,
            "original_revision": subprocess.check_output(
                ["git", "rev-parse", ORIGINAL_REVISION], cwd=ROOT, text=True
            ).strip(),
            "original_sources": {name: digest(source / name) for name in ORIGINAL_FILES},
            "harness_sha256": digest(__file__),
            "software": software,
            "model_contract": contract,
            "dataset": view.metadata["fingerprint"],
            "window_order_sha256": hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
            "physical_gpu": args.gpu,
            "gpu_uuid": uuid,
            "gpu_name": torch.cuda.get_device_name(),
            "allow_busy": True,
            "allocator_limit_gib": args.memory_gib,
            "gpu_before": before,
            "gpu_after": gpu_snapshot(args.gpu),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "precision": "bf16",
            "cpu_threads": 4,
            "workers": 2,
            "physical_batch": 4,
            "warmup_updates": args.warmup,
            "optimizer": "AdamW",
            "learning_rate": 0.00024,
            "weight_decay": 0.01,
            "gradient_clip_norm": 5.0,
            "objective": "full-waveform raw MSE against identical input",
            "dictionary_enabled": False,
            "compilation": False,
            "parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_parameters": sum(p.numel() for p in parameters),
            "seconds_per_update": timings,
            "losses": losses,
            "median_seconds_per_update": statistics.median(timings),
            "windows_per_second": 4 / statistics.median(timings),
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "attention_backends": attention_backends(model)
            if args.variant == "after"
            else "manual",
            "charged_seconds": budget.elapsed - charged_start,
        }
        atomic_json(case / "benchmark.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("task", "trial", "variant", "windows_per_second", "peak_allocated_gib")
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / ".runs/data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--variant", choices=("before", "after"))
    args = parser.parse_args()
    args.output, args.budget, args.data = (
        path.resolve() for path in (args.output, args.budget, args.data)
    )
    if args.updates < 1 or args.warmup < 0 or args.pairs < 1 or not 0 < args.memory_gib <= 8:
        parser.error("Positive bounded updates, pairs and memory are required")
    if args.variant:
        if not args.task:
            parser.error("A child case requires --task")
        measure(args)
        return
    if not os.environ.get("STY"):
        parser.error("Run the comparison in its own screen session")
    for task in TASKS:
        for trial in range(args.pairs):
            order = ("before", "after") if trial % 2 == 0 else ("after", "before")
            for variant in order:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output",
                        str(args.output),
                        "--budget",
                        str(args.budget),
                        "--data",
                        str(args.data),
                        "--gpu",
                        str(args.gpu),
                        "--memory-gib",
                        str(args.memory_gib),
                        "--warmup",
                        str(args.warmup),
                        "--updates",
                        str(args.updates),
                        "--task",
                        task,
                        "--trial",
                        str(trial),
                        "--variant",
                        variant,
                    ],
                    check=True,
                    cwd=ROOT,
                )


if __name__ == "__main__":
    main()
