"""Weighted logical updates, validation-only selection and exact progress snapshots."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch

from pretrain.compute import configure_compute, autocast_context as context
from pretrain.optimizers.stable_adamw import StableAdamW
from pretrain.pipeline.budget import BudgetStop, NoGpuBudget
from pretrain.pipeline.checkpoint import (
    contract,
    load,
    model_weights,
    optimizer_settings,
    restore_rng,
    save,
    set_seed,
    snapshot,
    source_revision,
    stream_seed,
    write_history,
)
from pretrain.pipeline.data import Windows, batches, chunks, evaluation_batches, to_device
from pretrain.pipeline.metrics import LossTotals
from pretrain.models.factory import attention_backends, create_model, masks, resolve_model_config
from pretrain.metadata import atomic_json


PEAK_LR = 2.4e-4
MINIMUM_LR = 1e-5
GRADIENT_CLIP_NORM = 5.0
DEFAULT_OPTIMIZER = {
    "lr": PEAK_LR,
    "betas": [0.9, 0.95],
    "eps": 1e-9,
    "weight_decay": 0.01,
}
DEFAULT_SCHEDULER = {"minimum_lr": MINIMUM_LR}


def resolve_training_config(config):
    """Resolve the training recipe selected by one YAML configuration."""
    resolved = resolve_model_config(
        {
            "seed": 241,
            "epochs": 100,
            "warmup_epochs": 10,
            "precision": "bf16",
            "logical_batch": 256,
            "microbatch": 16,
            "workers": 2,
            "probe_interval": 0,
            "diagnostic_test_probes": True,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            **copy.deepcopy(config),
        }
    )
    for name, defaults in (("optimizer", DEFAULT_OPTIMIZER), ("scheduler", DEFAULT_SCHEDULER)):
        supplied = resolved.get(name, {})
        if not isinstance(supplied, dict) or supplied.keys() - defaults.keys():
            raise ValueError(f"Unknown {name} settings; expected {', '.join(defaults)}")
        resolved[name] = {**copy.deepcopy(defaults), **supplied}
    optimizer = resolved["optimizer"]
    for name, value in (
        ("optimizer.lr", optimizer["lr"]),
        ("optimizer.eps", optimizer["eps"]),
        ("gradient_clip_norm", resolved["gradient_clip_norm"]),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")
    for name, value in (
        ("optimizer.weight_decay", optimizer["weight_decay"]),
        ("scheduler.minimum_lr", resolved["scheduler"]["minimum_lr"]),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative finite number")
    betas = optimizer["betas"]
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1
            for value in betas
        )
    ):
        raise ValueError("optimizer.betas must contain two numbers in [0, 1)")
    if resolved["scheduler"]["minimum_lr"] > optimizer["lr"]:
        raise ValueError("scheduler.minimum_lr cannot exceed optimizer.lr")
    for name in (
        "seed",
        "epochs",
        "warmup_epochs",
        "logical_batch",
        "microbatch",
        "workers",
        "probe_interval",
    ):
        minimum = 1 if name in {"epochs", "logical_batch", "microbatch"} else 0
        if type(resolved[name]) is not int or resolved[name] < minimum:
            raise ValueError(f"Invalid training setting: {name}")
    return resolved


def optimizer_for(model, settings=None):
    settings = settings or DEFAULT_OPTIMIZER
    return StableAdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["lr"],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
        return_norms=True,
    )


def schedule(
    epoch,
    batch,
    batches,
    epochs,
    *,
    warmup_epochs=10,
    peak_lr=PEAK_LR,
    minimum_lr=MINIMUM_LR,
):
    """Single warmup and cosine schedule; optimizer state never restarts."""
    peak, minimum = peak_lr, minimum_lr
    progress = epoch + (batch + 1) / batches
    if epoch < warmup_epochs:
        return peak * progress / max(1, warmup_epochs)
    ratio = min(1.0, (progress - warmup_epochs) / max(1.0, epochs - warmup_epochs))
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * ratio))


def validation_loss(model, view, config, device, budget):
    numerator, count = 0.0, 0
    model.eval()
    loader = evaluation_batches(view, config, device)
    with torch.inference_mode():
        for batch in loader:
            budget.check()
            mask = masks(
                batch["window_id"],
                batch["channel_ids"],
                samples=config["input_samples"],
                seed=stream_seed(config["seed"], "validation_mask"),
            ).to(device)
            eeg, ids = batch["eeg"].to(device), batch["channel_ids"].to(device)
            with context(device, config["precision"]):
                prediction, mask = model.predict_masked(eeg, ids, mask=mask)
                errors = model.patch_errors(eeg, prediction)
            numerator += float((errors * mask).sum())
            count += int(mask.sum())
    if not count:
        raise ValueError("Validation contains no masked patches")
    return numerator / count


def run(config, manifest, output, *, device="cpu", budget=None, resume=None, tracker=None):
    config = resolve_training_config(config)
    budget = budget or NoGpuBudget()
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, validation = Windows(manifest, "train"), Windows(manifest, "validation")
    from pretrain.pipeline.splits import validate_run_dataset

    validate_run_dataset(config, train.metadata)
    if train.metadata["window_samples"] != config["input_samples"]:
        raise ValueError("Dataset window length differs from model configuration")
    if config["epochs"] < 1 or config["microbatch"] < 1 or config["logical_batch"] < 1:
        raise ValueError("Positive epoch and batch sizes are required")
    if config.get("warmup_epochs", 10) < 0 or config["precision"] not in {"fp32", "bf16"}:
        raise ValueError("Invalid warmup horizon or precision")
    if device.type == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU checks require FP32")
    state = {
        "epoch": 0,
        "cursor": 0,
        "best_loss": None,
        "stopping_best": None,
        "bad_epochs": 0,
        "pending_probe": None,
        "history": [],
        "epoch_numerator": 0.0,
        "epoch_samples": 0,
        "epoch_mae": 0.0,
        "epoch_hydra": 0.0,
    }
    provenance = contract(config, train.metadata, device=device)
    set_seed(config["seed"])
    model = create_model(config).to(device)
    if config.get("compile", False):
        model.compile()
    optimizer = optimizer_for(model, config["optimizer"])
    warmup = min(config.get("warmup_epochs", 10), max(0, config["epochs"] - 1))
    atomic_json(
        output / "run.json",
        {
            "contract": provenance,
            "source_revision": source_revision(),
            "device": str(device),
            "purpose": config.get("purpose", "pretraining"),
            "resumed_from": str(Path(resume).resolve()) if resume else None,
            "resolved": {
                "model": model.contract,
                "optimizer": optimizer_settings(optimizer),
                "scheduler": {
                    "name": "warmup_cosine",
                    "peak_lr": config["optimizer"]["lr"],
                    "minimum_lr": config["scheduler"]["minimum_lr"],
                    "warmup_epochs": warmup,
                    "epochs": config["epochs"],
                },
                "gradient_clip_norm": config["gradient_clip_norm"],
            },
        },
    )
    stopping = config.get("early_stopping", {"patience": 10, "min_delta": 1e-4})
    if stopping["patience"] < 1 or stopping["min_delta"] < 0:
        raise ValueError("Early stopping requires positive patience and nonnegative min_delta")
    if resume:
        payload = load(resume, provenance)
        model.load_state_dict(payload["model"])
        state = payload["state"]
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload)
        save(output / "last.pth", payload)
        from pretrain.pipeline.probes import carry_completed

        carry_completed(Path(resume), output)
        previous_best = Path(resume).parent / "best.pth"
        if previous_best.exists():
            save(
                output / "best.pth",
                torch.load(previous_best, map_location="cpu", weights_only=True),
            )
    with configure_compute(config, device) as effective_compute:
        atomic_json(output / "compute.json", effective_compute)
        try:
            if state["pending_probe"] is not None:
                from pretrain.pipeline.probes import probe

                probe(
                    model.encoder,
                    manifest,
                    config,
                    output,
                    state["pending_probe"],
                    device,
                    budget,
                    tracker=tracker,
                )
                state["pending_probe"] = None
                save(output / "last.pth", snapshot(model, optimizer, state, provenance))
            for epoch in range(state["epoch"], config["epochs"]):
                if state["bad_epochs"] >= stopping["patience"]:
                    break
                loader = batches(
                    train,
                    batch_size=config["logical_batch"],
                    seed=stream_seed(config["seed"], f"data:{epoch}"),
                    workers=config.get("workers", 0),
                    pin_memory=device.type == "cuda",
                )
                model.train()
                for index, batch in enumerate(loader):
                    if index < state["cursor"]:
                        continue
                    budget.check()
                    optimizer.zero_grad(set_to_none=True)
                    rate = schedule(
                        epoch,
                        index,
                        len(loader),
                        config["epochs"],
                        warmup_epochs=warmup,
                        peak_lr=config["optimizer"]["lr"],
                        minimum_lr=config["scheduler"]["minimum_lr"],
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    total = len(batch["eeg"])
                    batch["mask"] = masks(
                        batch["window_id"],
                        batch["channel_ids"],
                        samples=config["input_samples"],
                        seed=stream_seed(config["seed"], f"mask:{epoch}"),
                    )
                    batch = to_device(batch, device)
                    totals = LossTotals()
                    for micro in chunks(batch, config["microbatch"]):
                        eeg, ids, mask = micro["eeg"], micro["channel_ids"], micro["mask"]
                        with context(device, config["precision"]):
                            result = model(eeg, ids, mask=mask)
                            loss = result["loss"] if isinstance(result, dict) else result[0]
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite pretraining loss")
                        (loss * (len(eeg) / total)).backward()
                        totals.add(
                            len(eeg),
                            loss,
                            result["mae_loss"] if isinstance(result, dict) else loss,
                            result["hydra_loss"]
                            if isinstance(result, dict)
                            else loss.new_zeros(()),
                        )
                    loss_sum, mae_sum, hydra_sum = totals.sums()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
                    )
                    optimizer.step()
                    state["cursor"] = index + 1
                    state["epoch_numerator"] += loss_sum
                    state["epoch_samples"] += total
                    state["epoch_mae"] += mae_sum
                    state["epoch_hydra"] += hydra_sum
                    if tracker:
                        tracker.log(
                            {
                                "epoch": epoch + (index + 1) / len(loader),
                                "train/loss": loss_sum / total,
                                "train/raw_mse": mae_sum / total,
                                "train/soft_dict": hydra_sum / total,
                                "train/lr": rate,
                                "train/grad_norm": float(norm),
                            }
                        )
                    print(
                        f"epoch={epoch + 1} update={index + 1}/{len(loader)} loss={loss_sum / total:.6f} "
                        f"lr={rate:.8g} grad_norm={float(norm):.5g}",
                        flush=True,
                    )
                # Save completed optimizer progress before validation can exhaust the budget.
                save(output / "last.pth", snapshot(model, optimizer, state, provenance))
                val = validation_loss(model, validation, config, device, budget)
                if (epoch + 1) % 10 == 0:
                    from pretrain.pipeline.examples import save as save_examples

                    save_examples(
                        model,
                        validation,
                        config,
                        device,
                        budget,
                        output / f"validation-examples-{epoch + 1:04d}.npz",
                    )
                row = {
                    "epoch": epoch + 1,
                    "train_loss": state["epoch_numerator"] / state["epoch_samples"],
                    "mae_loss": state["epoch_mae"] / state["epoch_samples"],
                    "hydra_loss": state["epoch_hydra"] / state["epoch_samples"],
                    "validation_mse": val,
                }
                state["history"].append(row)
                if tracker:
                    tracker.log({"epoch": epoch + 1, "val/raw_mse": val})
                if state["best_loss"] is None or val < state["best_loss"]:
                    state["best_loss"] = val
                    save(output / "best.pth", model_weights(model, config, provenance))
                if (
                    state["stopping_best"] is None
                    or val < state["stopping_best"] - stopping["min_delta"]
                ):
                    state["stopping_best"] = val
                    state["bad_epochs"] = 0
                elif epoch + 1 > warmup:
                    state["bad_epochs"] += 1
                if (epoch + 1) % 10 == 0:
                    save(
                        output / f"encoder-{epoch + 1:04d}.pth",
                        {
                            "encoder": model.encoder.state_dict(),
                            "config": config,
                            "contract": provenance,
                        },
                    )
                interval = config.get("probe_interval", 0)
                state["pending_probe"] = (
                    epoch + 1 if interval and (epoch + 1) % interval == 0 else None
                )
                state.update(
                    epoch=epoch + 1,
                    cursor=0,
                    epoch_numerator=0.0,
                    epoch_samples=0,
                    epoch_mae=0.0,
                    epoch_hydra=0.0,
                )
                save(output / "last.pth", snapshot(model, optimizer, state, provenance))
                write_history(output / "history.jsonl", state["history"])
                atomic_json(
                    output / "result.json",
                    {
                        "status": "complete" if state["epoch"] == config["epochs"] else "running",
                        "state": copy.deepcopy(state),
                    },
                )
                budget.flush()
                atomic_json(output / "attention.json", attention_backends(model))
                if state["pending_probe"] is not None:
                    from pretrain.pipeline.probes import probe

                    probe(
                        model.encoder,
                        manifest,
                        config,
                        output,
                        epoch + 1,
                        device,
                        budget,
                        tracker=tracker,
                    )
                    state["pending_probe"] = None
                    save(output / "last.pth", snapshot(model, optimizer, state, provenance))
                if state["bad_epochs"] >= stopping["patience"]:
                    break
        except BudgetStop:
            atomic_json(output / "attention.json", attention_backends(model))
            save(output / "last.pth", snapshot(model, optimizer, state, provenance))
            atomic_json(output / "result.json", {"status": "budget_stopped", "state": state})
            raise
    if config.get("probe_interval", 0):
        from pretrain.pipeline.probes import probe

        selected = torch.load(output / "best.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model"])
        try:
            probe(
                model.encoder, manifest, config, output, "selected", device, budget, tracker=tracker
            )
        except BudgetStop:
            # last.pth owns the final training state and optimizer. Keep that
            # pair intact after loading the validation-selected weights above.
            atomic_json(output / "result.json", {"status": "budget_stopped", "state": state})
            raise
    atomic_json(
        output / "result.json",
        {
            "status": "complete",
            "state": copy.deepcopy(state),
            "stop_reason": "early_stopping"
            if state["epoch"] < config["epochs"]
            else "maximum_epochs",
            "model_contract": model.contract,
            "parameters": model.parameter_counts(),
        },
    )
    write_history(output / "history.jsonl", state["history"])
    if tracker:
        tracker.summary(
            {"selected/validation_mse": state["best_loss"], "completed_epochs": state["epoch"]}
        )
    return output / "best.pth"
