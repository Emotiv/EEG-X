"""Diagnostic curves with a fixed classifier recipe and separate test outputs."""

import shutil

import numpy as np

from pretrain.pipeline.data import Windows
from pretrain.pipeline.linear_probe import features
from pretrain.pipeline.readout import fit_readout
from pretrain.pipeline.metrics import classification
from pretrain.metadata import atomic_json


def carry_completed(resume, output):
    """Keep completed diagnostic curves when a new attempt resumes progress."""
    previous = resume.parent
    for receipt in (previous / "probes").glob("*/diagnostics.json"):
        if receipt.parent.name != "selected":
            shutil.copytree(receipt.parent, output / "probes" / receipt.parent.name)
    for pattern in ("validation-examples-*.npz", "encoder-*.pth"):
        for path in previous.glob(pattern):
            shutil.copy2(path, output / path.name)
    receipt = previous / "attention.json"
    if receipt.exists():
        shutil.copy2(receipt, output / receipt.name)


def probe(encoder, dataset, config, output, epoch, device, budget, *, tracker=None):
    root = output / "probes" / str(epoch)
    root.mkdir(parents=True, exist_ok=False)
    train = Windows(dataset, "train")
    x, y, _ = features(encoder, train, config, device, budget)
    validation = Windows(dataset, "validation")
    val_x, val_y, _ = features(encoder, validation, config, device, budget)
    classifier = fit_readout(
        x, y, val_x, val_y, config, root, budget, settings=config.get("linear_probe")
    )
    del x
    scores = {
        "purpose": "diagnostic_only",
        "epoch": epoch,
        "dataset": train.metadata["fingerprint"],
    }
    roles = ["validation"] + (["test"] if config.get("diagnostic_test_probes") else [])
    for role in roles:
        view = Windows(dataset, role)
        x, y, identities = features(encoder, view, config, device, budget)
        probability = classifier.predict_proba(x)
        scores[role] = classification(y, probability)
        if tracker:
            prefix = "linear_probe/val" if role == "validation" else "linear_probe/test"
            metrics = {f"{prefix}/{key}": value for key, value in scores[role].items()}
            if isinstance(epoch, int):
                tracker.log({"epoch": epoch, **metrics})
            else:
                tracker.summary({f"selected/{key}": value for key, value in metrics.items()})
            tracker.confusion(
                y,
                probability,
                prefix=prefix if isinstance(epoch, int) else f"selected/{prefix}",
                epoch=epoch if isinstance(epoch, int) else None,
            )
        np.savez_compressed(
            root / f"{role}.npz",
            labels=y,
            probabilities=probability,
            window_ids=np.asarray(identities),
        )
    atomic_json(root / "diagnostics.json", scores)
    return scores
