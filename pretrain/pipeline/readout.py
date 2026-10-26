"""Epoch-trained linear classifiers over fixed, train-standardized encoder features."""

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pretrain.pipeline.checkpoint import save, stream_seed, write_history
from pretrain.pipeline.metrics import classification
from pretrain.pipeline.selection import selection_metadata


class LinearReadout:
    def __init__(self, model, scaler, history, selected_epoch, validation):
        self.model, self.scaler = model, scaler
        self.history, self.selected_epoch, self.validation = history, selected_epoch, validation

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict_proba(self, values):
        features = torch.from_numpy(self.scaler.transform(values).astype(np.float32))
        with torch.inference_mode():
            return torch.cat(
                [self.model(batch).softmax(-1) for batch in features.split(2048)]
            ).numpy()

    def save(self, path, *, metadata=None):
        save(
            path,
            {
                "model": self.model.state_dict(),
                "mean": torch.from_numpy(self.scaler.mean_),
                "scale": torch.from_numpy(self.scaler.scale_),
                "epoch": self.selected_epoch,
                "validation": self.validation,
                "history": self.history,
                "selection": selection_metadata("linear_probe"),
                **(metadata or {}),
            },
        )


def fit_readout(
    train_x, train_y, val_x, val_y, config, output, budget, *, tracker=None, settings=None
):
    """Select a linear head on validation while preserving pretraining's random state."""
    settings = {
        "epochs": 20,
        "patience": 5,
        "min_delta": 0.0,
        "lr": 1e-3,
        "batch_size": 64,
        **(settings or {}),
    }
    for key in ("epochs", "patience", "batch_size"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Linear probing requires a positive {key}")
    if settings["lr"] <= 0 or settings["min_delta"] < 0:
        raise ValueError("Linear probing requires positive lr and nonnegative min_delta")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(train_x)
    x = torch.from_numpy(scaler.transform(train_x).astype(np.float32))
    y = torch.from_numpy(np.asarray(train_y, dtype=np.int64))
    validation_x = torch.from_numpy(scaler.transform(val_x).astype(np.float32))
    classes = len(np.unique(train_y))
    if not np.array_equal(np.unique(train_y), np.arange(classes)):
        raise ValueError("Linear-probe training must contain every class")
    seed = stream_seed(config["seed"], "linear_probe")
    history, best, best_scores, best_accuracy, selected_epoch, bad_epochs = (
        [],
        None,
        None,
        None,
        None,
        0,
    )
    stopping_best = None
    global_step = 0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = torch.nn.Linear(x.shape[1], classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.01)
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, settings["epochs"] + 1):
            total = 0.0
            batches = torch.randperm(len(x), generator=generator).split(settings["batch_size"])
            for index, rows in enumerate(batches):
                budget.check()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x[rows]), y[rows])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite linear-probe loss")
                loss.backward()
                optimizer.step()
                global_step += 1
                batch_loss = float(loss.detach())
                total += batch_loss * len(rows)
                if tracker:
                    tracker.log(
                        {
                            "epoch": epoch - 1 + (index + 1) / len(batches),
                            "trainer/global_step": global_step,
                            "train/loss_step": batch_loss,
                            "train/lr": settings["lr"],
                        }
                    )
            budget.check()
            with torch.inference_mode():
                probability = model(validation_x).softmax(-1).numpy()
            scores = classification(val_y, probability)
            history.append({"epoch": epoch, "train_loss": total / len(x), "validation": scores})
            if tracker:
                tracker.log(
                    {
                        "epoch": epoch,
                        "train/loss_epoch": total / len(x),
                        "train/lr": settings["lr"],
                        **{f"val/{key}": value for key, value in scores.items()},
                    }
                )
                tracker.confusion(val_y, probability, prefix="val", epoch=epoch)
            accuracy = scores["balanced_accuracy"]
            if best_accuracy is None or accuracy > best_accuracy:
                best_accuracy, selected_epoch = accuracy, epoch
                best, best_scores = copy.deepcopy(model.state_dict()), scores
            if stopping_best is None or accuracy > stopping_best + settings["min_delta"]:
                stopping_best = accuracy
                bad_epochs = 0
            else:
                bad_epochs += 1
            write_history(root / "readout_history.jsonl", history)
            if bad_epochs >= settings["patience"]:
                break
        model.load_state_dict(best)
    result = LinearReadout(model, scaler, history, selected_epoch, best_scores)
    result.save(root / "best.pth")
    return result
