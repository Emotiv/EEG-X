"""Full supervised fine-tuning with validation-only model selection."""

import copy
import math
from pathlib import Path

import numpy as np
import torch

from pretrain.channel_identity import sha256_file
from pretrain.compute import configure_compute, autocast_context as context
from pretrain.pipeline.budget import BudgetStop, NoGpuBudget
from pretrain.pipeline.checkpoint import (
    contract,
    load,
    optimizer_settings,
    restore_rng,
    save,
    snapshot,
    source_revision,
    stream_seed,
    write_history,
)
from pretrain.pipeline.data import Windows, batches, chunks, evaluation_batches, to_device
from pretrain.pipeline.downstream import classifier, optimizer_groups, validate_pretraining_source
from pretrain.pipeline.metrics import LossTotals, classification
from pretrain.pipeline.selection import selection_metadata
from pretrain.models.factory import attention_backends
from pretrain.metadata import atomic_json


PEAK_LR = 1e-3
MINIMUM_LR = 1e-6
GRADIENT_CLIP_NORM = 1.0


def predictions(model, view, config, device, budget):
    values, labels, identities = [], [], []
    loader = evaluation_batches(view, config, device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            budget.check()
            with context(device, config["precision"]):
                logits = model(batch["eeg"].to(device), batch["channel_ids"].to(device))
            values.append(logits.float().softmax(-1).cpu().numpy())
            labels.append(batch["label"].numpy())
            identities.extend(batch["window_id"])
    return np.concatenate(labels), np.concatenate(values), identities


def learning_rate(progress, epochs, warmup=5, *, peak_lr=PEAK_LR, minimum_lr=MINIMUM_LR):
    warmup = min(warmup, max(0, epochs - 1))
    if warmup and progress <= warmup:
        return peak_lr * progress / warmup
    ratio = min(1.0, (progress - warmup) / max(1, epochs - warmup))
    return minimum_lr + (peak_lr - minimum_lr) * (1 + math.cos(math.pi * ratio)) / 2


def run(
    config, dataset, checkpoint, output, *, device="cpu", budget=None, resume=None, tracker=None
):
    for key in ("epochs", "logical_batch", "microbatch"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    warmup = config.get("warmup_epochs", 5)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("Warmup epochs must be a nonnegative integer")
    warmup = min(warmup, max(0, config["epochs"] - 1))
    budget = budget or NoGpuBudget()
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, validation = Windows(dataset, "train"), Windows(dataset, "validation")
    from pretrain.pipeline.splits import validate_run_dataset

    validate_run_dataset(config, train.metadata)
    config = copy.deepcopy(config)
    settings = {
        "lr": PEAK_LR,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        **config.get("optimizer", {}),
    }
    minimum_lr = config.get("scheduler", {}).get("minimum_lr", MINIMUM_LR)
    clip_norm = config.get("gradient_clip_norm", GRADIENT_CLIP_NORM)
    stopping = config.get("early_stopping", {"patience": 5, "min_delta": 0.0})
    if stopping["patience"] < 1 or stopping["min_delta"] < 0:
        raise ValueError("Early stopping requires positive patience and nonnegative min_delta")
    config["pretraining_checkpoint_sha256"] = sha256_file(checkpoint)
    provenance = contract(config, train.metadata, device=device)
    model, pretrained = classifier(
        checkpoint, len(train.metadata["classes"]), config["seed"],
        normalization_backend=config.get("normalization_backend"),
    )
    validate_pretraining_source(pretrained, config, train.metadata, mode="finetune")
    model.to(device)
    if tracker:
        tracker.parameters(model)
    groups, audit = optimizer_groups(model)
    for group, record in zip(groups, audit):
        group["lr"] = settings["lr"] * group["lr_scale"]
        group["weight_decay"] = settings["weight_decay"] if group["weight_decay"] else 0.0
        record["weight_decay"] = group["weight_decay"]
    optimizer = torch.optim.AdamW(
        groups, lr=settings["lr"], betas=tuple(settings["betas"]), eps=settings["eps"]
    )
    state = {
        "epoch": 0,
        "cursor": 0,
        "global_step": 0,
        "best_accuracy": None,
        "history": [],
        "epoch_loss": 0.0,
        "epoch_samples": 0,
        "bad_epochs": 0,
        "stopping_best": None,
    }
    atomic_json(
        output / "run.json",
        {
            "contract": provenance,
            "source_revision": source_revision(),
            "optimizer_groups": audit,
            "resolved": {
                "optimizer": optimizer_settings(optimizer),
                "scheduler": {
                    "name": "warmup_cosine",
                    "peak_lr": settings["lr"],
                    "minimum_lr": minimum_lr,
                    "warmup_epochs": warmup,
                    "epochs": config["epochs"],
                },
                "gradient_clip_norm": clip_norm,
                "selection": selection_metadata("finetune"),
                "classifier": {
                    "normalization": type(model.head[0]).__name__,
                    "dropout": model.head[1].p,
                    "classes": len(train.metadata["classes"]),
                },
            },
        },
    )
    if resume:
        payload = load(resume, provenance)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        state = payload["state"]
        restore_rng(payload)
        save(output / "last.pth", payload)
        previous = Path(resume).parent / "best.pth"
        if previous.exists():
            save(output / "best.pth", torch.load(previous, map_location="cpu", weights_only=True))
    with configure_compute(config, device) as effective_compute:
        atomic_json(output / "compute.json", effective_compute)
        try:
            for epoch in range(state["epoch"], config["epochs"]):
                if state["bad_epochs"] >= stopping["patience"]:
                    break
                loader = batches(
                    train,
                    batch_size=config["logical_batch"],
                    seed=stream_seed(config["seed"], f"finetune_data:{epoch}"),
                    workers=config.get("workers", 0),
                    pin_memory=device.type == "cuda",
                )
                model.train()
                for index, batch in enumerate(loader):
                    if index < state["cursor"]:
                        continue
                    budget.check()
                    optimizer.zero_grad(set_to_none=True)
                    rate = learning_rate(
                        epoch + (index + 1) / len(loader),
                        config["epochs"],
                        warmup=warmup,
                        peak_lr=settings["lr"],
                        minimum_lr=minimum_lr,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = rate * group["lr_scale"]
                    total = len(batch["eeg"])
                    batch = to_device(batch, device)
                    totals = LossTotals()
                    for micro in chunks(batch, config["microbatch"]):
                        eeg, ids = micro["eeg"].to(device), micro["channel_ids"].to(device)
                        with context(device, config["precision"]):
                            logits = model(eeg, ids)
                            loss = torch.nn.functional.cross_entropy(
                                logits.float(), micro["label"].to(device)
                            )
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite fine-tuning loss")
                        (loss * len(eeg) / total).backward()
                        totals.add(len(eeg), loss)
                    total_loss = totals.sums()[0]
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                    state["global_step"] += 1
                    if tracker:
                        rates = [group["lr"] for group in optimizer.param_groups]
                        tracker.log(
                            {
                                "epoch": epoch + (index + 1) / len(loader),
                                "trainer/global_step": state["global_step"],
                                "train/loss_step": total_loss / total,
                                "train/lr": rate,
                                "train/lr_min": min(rates),
                                "train/lr_max": max(rates),
                            }
                        )
                    state["cursor"] = index + 1
                    state["epoch_loss"] += total_loss
                    state["epoch_samples"] += total
                save(output / "last.pth", snapshot(model, optimizer, state, provenance))
                labels, probabilities, _ = predictions(model, validation, config, device, budget)
                scores = classification(labels, probabilities)
                print(
                    f"epoch={epoch + 1} train_loss={state['epoch_loss'] / state['epoch_samples']:.6f} "
                    f"val_loss={scores['loss']:.6f} val_bal_acc={scores['balanced_accuracy']:.6f}",
                    flush=True,
                )
                if tracker:
                    tracker.log(
                        {
                            "epoch": epoch + 1,
                            "train/loss_epoch": state["epoch_loss"] / state["epoch_samples"],
                            **{f"val/{key}": value for key, value in scores.items()},
                        }
                    )
                    tracker.confusion(labels, probabilities, prefix="val", epoch=epoch + 1)
                state["history"].append(
                    {
                        "epoch": epoch + 1,
                        "validation": scores,
                        "train_loss": state["epoch_loss"] / state["epoch_samples"],
                    }
                )
                if (
                    state["best_accuracy"] is None
                    or scores["balanced_accuracy"] > state["best_accuracy"]
                ):
                    state["best_accuracy"] = scores["balanced_accuracy"]
                    save(
                        output / "best.pth",
                        {
                            "model": model.state_dict(),
                            "contract": provenance,
                            "epoch": epoch + 1,
                            "validation": scores,
                        },
                    )
                if (
                    state["stopping_best"] is None
                    or scores["balanced_accuracy"] > state["stopping_best"] + stopping["min_delta"]
                ):
                    state["stopping_best"] = scores["balanced_accuracy"]
                    state["bad_epochs"] = 0
                else:
                    state["bad_epochs"] += 1
                state.update(epoch=epoch + 1, cursor=0, epoch_loss=0.0, epoch_samples=0)
                save(output / "last.pth", snapshot(model, optimizer, state, provenance))
                write_history(output / "history.jsonl", state["history"])
                budget.flush()
                if state["bad_epochs"] >= stopping["patience"]:
                    break
        except BudgetStop:
            save(output / "last.pth", snapshot(model, optimizer, state, provenance))
            atomic_json(output / "result.json", {"status": "budget_stopped", "state": state})
            raise
        # Selection starts a separate phase. The last training checkpoint owns
        # matching model, optimizer, progress, and random states throughout it.
        try:
            best = torch.load(output / "best.pth", map_location="cpu", weights_only=True)
            model.load_state_dict(best["model"])
            selection = {
                "status": "locked_before_test",
                "criterion": selection_metadata("finetune"),
                "checkpoint_sha256": sha256_file(output / "best.pth"),
                "epoch": best["epoch"],
                "validation": best["validation"],
                "contract": provenance,
                "pretraining_contract": pretrained["contract"],
            }
            atomic_json(output / "selection.json", selection)
            if tracker:
                tracker.summary(
                    {
                        "completed_epochs": state["epoch"],
                        "stop_reason": "early_stopping"
                        if state["bad_epochs"] >= stopping["patience"]
                        else "maximum_epochs",
                    }
                )
                tracker.summary(
                    {
                        "selected/epoch": best["epoch"],
                        "best_epoch": best["epoch"],
                        **{
                            f"selected/val/{key}": value
                            for key, value in best["validation"].items()
                        },
                    }
                )
                tracker.checkpoint(
                    output / "best.pth",
                    metadata={"selected/epoch": best["epoch"], "validation": best["validation"]},
                )
            test = Windows(dataset, "test")
            labels, probabilities, identities = predictions(model, test, config, device, budget)
            scores = classification(labels, probabilities)
            if tracker:
                tracker.final_test(scores)
                tracker.confusion(labels, probabilities)
            atomic_json(output / "attention.json", attention_backends(model))
            np.savez_compressed(
                output / "test_predictions.npz",
                labels=labels,
                probabilities=probabilities,
                window_ids=np.asarray(identities),
            )
            atomic_json(
                output / "result.json",
                {"status": "complete", "selection": selection, "test": scores},
            )
            return scores
        except BudgetStop:
            atomic_json(output / "result.json", {"status": "budget_stopped", "state": state})
            raise
