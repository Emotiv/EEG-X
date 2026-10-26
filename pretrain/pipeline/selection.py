"""Validation-only checkpoint policies for reconstruction and classification."""

from pathlib import Path

from pretrain.pipeline.checkpoint import model_weights, save
from pretrain.pipeline.data import Windows


def selection_metadata(stage):
    metric, mode = {
        "pretrain": ("val/raw_mse", "min"),
        "linear_probe": ("val/bal_acc", "max"),
        "finetune": ("val/bal_acc", "max"),
    }[stage]
    return {"metric": metric, "mode": mode}


class ValidationSelection:
    name = "minimum_validation_loss"
    filename = "best.pth"

    def __init__(self, config, dataset, output, device, budget, tracker, warmup):
        self.config, self.output = config, Path(output)
        self.device, self.budget, self.tracker = device, budget, tracker
        self.warmup = warmup
        self.validation = Windows(dataset, "validation")
        self.stopping = config.get("early_stopping", {"patience": 10, "min_delta": 1e-4})
        if self.stopping["patience"] < 1 or self.stopping["min_delta"] < 0:
            raise ValueError("Early stopping requires positive patience and nonnegative min_delta")

    def observe(self, model, state, provenance, epoch):
        from pretrain.pipeline.training import validation_loss

        value = validation_loss(model, self.validation, self.config, self.device, self.budget)
        if epoch % 10 == 0:
            from pretrain.pipeline.examples import save as save_examples

            save_examples(
                model,
                self.validation,
                self.config,
                self.device,
                self.budget,
                self.output / f"validation-examples-{epoch:04d}.npz",
            )
        if self.tracker:
            self.tracker.log({"epoch": epoch, "val/raw_mse": value})
        if state["best_loss"] is None or value < state["best_loss"]:
            state["best_loss"] = value
            state["best_epoch"] = epoch
            save(
                self.output / self.filename,
                {**model_weights(model, self.config, provenance), "epoch": epoch},
            )
        if (
            state["stopping_best"] is None
            or value < state["stopping_best"] - self.stopping["min_delta"]
        ):
            state["stopping_best"] = value
            state["bad_epochs"] = 0
        elif epoch > self.warmup:
            state["bad_epochs"] += 1
        return {"validation_mse": value}

    def should_stop(self, state):
        return state["bad_epochs"] >= self.stopping["patience"]

    def summary(self, state):
        return {
            "selected/validation_mse": state["best_loss"],
            "selected/epoch": state["best_epoch"],
            "best_epoch": state["best_epoch"],
        }


def checkpoint_selection(config, dataset, output, device, budget, tracker, warmup):
    return ValidationSelection(config, dataset, output, device, budget, tracker, warmup)
