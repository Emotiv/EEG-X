"""Train-only frozen probe, with explicit selection before test access."""

from pathlib import Path
import multiprocessing

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pretrain.channel_identity import sha256_file
from pretrain.compute import configure_compute, autocast_context as context
from pretrain.pipeline.budget import NoGpuBudget
from pretrain.pipeline.checkpoint import contract, source_revision
from pretrain.pipeline.data import Windows, evaluation_batches
from pretrain.pipeline.downstream import load_encoder
from pretrain.pipeline.metrics import classification
from pretrain.models.factory import attention_backends
from pretrain.metadata import atomic_json


def _fit(probe, values, labels, output):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        probe.fit(values, labels)
    joblib.dump(probe, output)


def classifier_for(seed):
    """One fixed recipe for diagnostic and final frozen evaluation."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1_000_000, random_state=seed),
    )


def fit_probe(probe, values, labels, output, budget):
    """Keep the CPU solver interruptible while its job reserves the GPU."""
    path = Path(output) / "probe-fitting.joblib"
    process = multiprocessing.get_context("spawn").Process(
        target=_fit,
        args=(probe, values, labels, path),
    )
    budget.check()
    process.start()
    try:
        while process.is_alive():
            budget.check()
            process.join(timeout=0.25)
        budget.check()
        if process.exitcode != 0:
            raise RuntimeError(f"Frozen probe fit failed: exit {process.exitcode}")
        return joblib.load(path)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        path.unlink(missing_ok=True)


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


def run(config, manifest, checkpoint, output, *, device="cpu", budget=None, tracker=None):
    budget = budget or NoGpuBudget()
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoder, payload = load_encoder(checkpoint)
    encoder.to(device)
    train, validation = Windows(manifest, "train"), Windows(manifest, "validation")
    from pretrain.pipeline.splits import validate_run_dataset

    validate_run_dataset(config, train.metadata)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    provenance = contract(config, train.metadata, device=device)
    probe = classifier_for(config["seed"])
    atomic_json(
        output / "run.json",
        {
            "contract": provenance,
            "source_revision": source_revision(),
            "resolved": {
                "classifier": probe.named_steps["logisticregression"].get_params(),
                "scaler": probe.named_steps["standardscaler"].get_params(),
            },
        },
    )
    x, y, _ = features(encoder, train, config, device, budget)
    budget.check()
    probe = fit_probe(probe, x, y, output, budget)
    del x
    x, y, _ = features(encoder, validation, config, device, budget)
    validation_scores = classification(y, probe.predict_proba(x))
    del x
    artifact = output / "probe.joblib"
    joblib.dump(probe, artifact)
    selection = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "probe_sha256": sha256_file(artifact),
        "dataset": train.metadata["fingerprint"],
        "config": config,
        "validation": validation_scores,
        "pretraining_contract": payload["contract"],
        "contract": provenance,
        "status": "locked_before_test",
    }
    atomic_json(output / "selection.json", selection)
    # No test view or features exist before the selected encoder and probe are saved.
    test = Windows(manifest, "test")
    x, y, identities = features(encoder, test, config, device, budget)
    probabilities = probe.predict_proba(x)
    scores = classification(y, probabilities)
    if tracker:
        tracker.log({"epoch": 0, **{f"test/{key}": value for key, value in scores.items()}})
        tracker.summary(
            {
                **{f"val/{key}": value for key, value in validation_scores.items()},
                **{f"test/{key}": value for key, value in scores.items()},
            }
        )
        tracker.confusion(y, probabilities)
    atomic_json(output / "attention.json", attention_backends(encoder))
    np.savez_compressed(
        output / "test_predictions.npz",
        labels=y,
        probabilities=probabilities,
        window_ids=np.asarray(identities),
    )
    atomic_json(
        output / "result.json", {"status": "complete", "selection": selection, "test": scores}
    )
    return scores
