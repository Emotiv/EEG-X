"""Fixed validation reconstructions for the accepted learning-curve checkpoints."""

import numpy as np
import torch

from pretrain.pipeline.data import collate
from pretrain.models.factory import masks
from pretrain.pipeline.checkpoint import stream_seed
from pretrain.compute import autocast_context as context


def save(model, validation, config, device, budget, path):
    budget.check()
    batch = collate([validation[index] for index in range(min(2, len(validation)))])
    mask = masks(
        batch["window_id"],
        batch["channel_ids"],
        samples=config["input_samples"],
        seed=stream_seed(config["seed"], "validation_mask"),
    ).to(device)
    model.eval()
    with torch.inference_mode(), context(device, config["precision"]):
        prediction, mask = model.predict_masked(
            batch["eeg"].to(device), batch["channel_ids"].to(device), mask=mask
        )
    np.savez_compressed(
        path,
        eeg=batch["eeg"].numpy(),
        channel_ids=batch["channel_ids"].numpy(),
        predictions=prediction.float().cpu().numpy(),
        mask=mask.float().cpu().numpy(),
        window_ids=np.asarray(batch["window_id"]),
        dataset=validation.metadata["fingerprint"],
    )
