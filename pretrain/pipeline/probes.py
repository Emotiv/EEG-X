"""Diagnostic curves with a fixed classifier recipe and separate test outputs."""

import shutil

import numpy as np

from pretrain.pipeline.data import Windows
from pretrain.pipeline.frozen import classifier_for, features, fit_probe
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


def probe(encoder, manifest, config, output, epoch, device, budget):
    root = output / "probes" / str(epoch)
    root.mkdir(parents=True, exist_ok=False)
    train = Windows(manifest, "train")
    x, y, _ = features(encoder, train, config, device, budget)
    classifier = classifier_for(config["seed"])
    classifier = fit_probe(classifier, x, y, root, budget)
    del x
    scores = {
        "purpose": "diagnostic_only",
        "epoch": epoch,
        "dataset": train.metadata["fingerprint"],
    }
    roles = ["validation"] + (["test"] if config.get("diagnostic_test_probes") else [])
    for role in roles:
        view = Windows(manifest, role)
        x, y, identities = features(encoder, view, config, device, budget)
        probability = classifier.predict_proba(x)
        scores[role] = classification(y, probability)
        np.savez_compressed(
            root / f"{role}.npz",
            labels=y,
            probabilities=probability,
            window_ids=np.asarray(identities),
        )
    atomic_json(root / "diagnostics.json", scores)
    return scores
