"""Linear probing with a fixed encoder and selection before test access."""

from pathlib import Path

import numpy as np
import torch

from pretrain.channel_identity import sha256_file
from pretrain.compute import configure_compute, autocast_context as context
from pretrain.pipeline.budget import NoGpuBudget
from pretrain.pipeline.checkpoint import contract, source_revision
from pretrain.pipeline.data import Windows, evaluation_batches
from pretrain.pipeline.downstream import load_encoder, validate_pretraining_source
from pretrain.pipeline.metrics import classification
from pretrain.pipeline.readout import fit_readout
from pretrain.pipeline.selection import selection_metadata
from pretrain.models.factory import attention_backends
from pretrain.metadata import atomic_json


def features(encoder, view, config, device, budget):
    values, labels, identities = [], [], []
    loader = evaluation_batches(view, config, device)
    encoder.eval()
    with configure_compute(config, device), torch.inference_mode():
        for batch in loader:
            budget.check()
            with context(device, config["precision"]):
                output = encoder.forward_emotiv_features(
                    batch["eeg"].to(device),
                    batch["channel_ids"].to(device),
                    input_window_normalization="none",
                )
            values.append(output.float().cpu().numpy())
            labels.append(batch["label"].numpy())
            identities.extend(batch["window_id"])
    return np.concatenate(values), np.concatenate(labels), identities


def run(config, dataset, checkpoint, output, *, device="cpu", budget=None, tracker=None):
    budget = budget or NoGpuBudget()
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoder, payload = load_encoder(checkpoint, normalization_backend=config.get("normalization_backend"))
    train, validation = Windows(dataset, "train"), Windows(dataset, "validation")
    from pretrain.pipeline.splits import validate_run_dataset

    validate_run_dataset(config, train.metadata)
    validate_pretraining_source(payload, config, train.metadata, mode="linear_probe")
    encoder.to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    provenance = contract(config, train.metadata, device=device)
    atomic_json(
        output / "run.json",
        {
            "contract": provenance,
            "source_revision": source_revision(),
            "resolved": {
                "classifier": "linear",
                "optimizer": "AdamW",
                "epochs": config.get("epochs", 20),
                "early_stopping": config.get("early_stopping", {"patience": 5, "min_delta": 0.0}),
                "scaler": "train_only_standard_scaler",
                "selection": selection_metadata("linear_probe"),
            },
        },
    )
    train_x, train_y, _ = features(encoder, train, config, device, budget)
    val_x, val_y, _ = features(encoder, validation, config, device, budget)
    stopping = config.get("early_stopping", {"patience": 5, "min_delta": 0.0})
    probe = fit_readout(
        train_x,
        train_y,
        val_x,
        val_y,
        config,
        output,
        budget,
        tracker=tracker,
        settings={
            "epochs": config.get("epochs", 20),
            **stopping,
            "lr": config.get("optimizer", {}).get("lr", 1e-3),
        },
    )
    if tracker:
        tracker.parameters(encoder, classifier_parameters=probe.parameter_count)
        tracker.summary(
            {
                "selected/epoch": probe.selected_epoch,
                "best_epoch": probe.selected_epoch,
                "completed_epochs": len(probe.history),
                **{f"selected/val/{key}": value for key, value in probe.validation.items()},
            }
        )
    validation_scores = probe.validation
    artifact = output / "best.pth"
    probe.save(
        artifact,
        metadata={
            "config": config,
            "contract": provenance,
            "source_checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    selection = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "probe_sha256": sha256_file(artifact),
        "dataset": train.metadata["fingerprint"],
        "config": config,
        "validation": validation_scores,
        "pretraining_contract": payload["contract"],
        "contract": provenance,
        "status": "locked_before_test",
        "criterion": selection_metadata("linear_probe"),
    }
    atomic_json(output / "selection.json", selection)
    if tracker:
        tracker.checkpoint(
            artifact,
            metadata={"selected/epoch": probe.selected_epoch, "validation": validation_scores},
        )
    # No test view or features exist before the selected encoder and probe are saved.
    test = Windows(dataset, "test")
    x, y, identities = features(encoder, test, config, device, budget)
    probabilities = probe.predict_proba(x)
    scores = classification(y, probabilities)
    if tracker:
        tracker.summary({f"val/{key}": value for key, value in validation_scores.items()})
        tracker.final_test(scores)
        tracker.confusion(y, probabilities)
    atomic_json(output / "attention.json", attention_backends(encoder))
    np.savez_compressed(
        output / "test_predictions.npz",
        labels=y,
        probabilities=probabilities,
        window_ids=np.asarray(identities),
    )
    atomic_json(
        output / "result.json",
        {"status": "complete", "selection": selection, "test": scores, "history": probe.history},
    )
    return scores
