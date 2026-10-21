"""Weighted logical updates, validation-only selection and exact progress snapshots."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from pretrain.compute import configure_compute, autocast_context as context
from pretrain.optimizer_factory import build_optimizer
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


def optimizer_for(model):
    return build_optimizer(
        model,
        SimpleNamespace(
            optimizer="stable_adamw",
            lr=PEAK_LR,
            adamw_beta1=0.9,
            adamw_beta2=0.95,
            adamw_eps=1e-9,
            weight_decay=0.01,
        ),
    )


def schedule(epoch, batch, batches, epochs, *, warmup_epochs=10):
    """Single warmup and cosine schedule; optimizer state never restarts."""
    peak, minimum = PEAK_LR, MINIMUM_LR
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


def run(config, manifest, output, *, device="cpu", budget=None, resume=None):
    config = resolve_model_config(config)
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
    optimizer = optimizer_for(model)
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
                    "peak_lr": PEAK_LR,
                    "minimum_lr": MINIMUM_LR,
                    "warmup_epochs": warmup,
                    "epochs": config["epochs"],
                },
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
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
                    model.encoder, manifest, config, output, state["pending_probe"], device, budget
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
                        epoch, index, len(loader), config["epochs"], warmup_epochs=warmup
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
                        model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
                    )
                    optimizer.step()
                    state["cursor"] = index + 1
                    state["epoch_numerator"] += loss_sum
                    state["epoch_samples"] += total
                    state["epoch_mae"] += mae_sum
                    state["epoch_hydra"] += hydra_sum
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

                    probe(model.encoder, manifest, config, output, epoch + 1, device, budget)
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
            probe(model.encoder, manifest, config, output, "selected", device, budget)
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
    return output / "best.pth"
