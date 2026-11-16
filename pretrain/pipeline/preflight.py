"""CPU real-window checks before the reviewed production GPU pipeline."""

import argparse
import json

import torch

from pretrain.objectives import DEFAULT_HYDRA
from pretrain.pipeline.checkpoint import set_seed
from pretrain.pipeline.data import Windows, collate
from pretrain.models.factory import Classifier, create_model, masks


def check(dataset):
    torch.set_num_threads(2)
    train = Windows(dataset, "train")
    batch = collate([train[0], train[min(1, len(train) - 1)]])
    samples = train.metadata["window_samples"]
    output = []
    size = "d8w64"
    for objective in (None, DEFAULT_HYDRA):
        set_seed(241)
        config = {"size": size, "input_samples": samples, "objective": objective}
        model = create_model(config)
        mask = masks(batch["window_id"], batch["channel_ids"], samples=samples, seed=241)
        result = model(batch["eeg"], batch["channel_ids"], mask=mask)
        loss = result["loss"] if isinstance(result, dict) else result[0]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite real-data objective")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite real-data gradient")
        encoder_gradient = sum(
            float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None
        )
        if encoder_gradient <= 0:
            raise ValueError("Encoder received no gradient")
        classifier = Classifier(model.encoder, len(train.metadata["classes"])).eval()
        with torch.no_grad():
            logits = classifier(batch["eeg"], batch["channel_ids"])
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite supervised real-data output")
        output.append(
            {
                "size": size,
                "objective": "soft_dict" if objective else "raw",
                "loss": float(loss.detach()),
                "encoder_gradient_l1": encoder_gradient,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        del model, classifier, result, loss
    return {
        "status": "passed",
        "scope": "real-window engineering check; not a paper result or source-history audit",
        "task": train.metadata["task"],
        "dataset": train.metadata["fingerprint"],
        "checks": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.dataset)))


if __name__ == "__main__":
    main()
