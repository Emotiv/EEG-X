"""HDF5 window datasets and deterministic PyTorch batching."""

import torch
from torch.utils.data import DataLoader

from pretrain.pipeline.dataset import (
    INDEX_DTYPE as INDEX_DTYPE,
    SPLITS as SPLITS,
    TASK_WINDOWS as TASK_WINDOWS,
    PreparedWriter as PreparedWriter,
    Windows as Windows,
    subject_split as subject_split,
)


def batches(view, *, batch_size, seed, workers=0, pin_memory=False, shuffle=True):
    """Keep iterator and worker seeds independent of the model's random state."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=pin_memory,
        multiprocessing_context="spawn" if workers else None,
    )


def evaluation_batches(view, config, device):
    return batches(
        view,
        batch_size=config["microbatch"],
        seed=config["seed"],
        workers=config.get("workers", 0),
        pin_memory=device.type == "cuda",
        shuffle=False,
    )


def chunks(batch, size):
    if type(size) is not int or size < 1:
        raise ValueError("Microbatch size must be a positive integer")
    for start in range(0, len(batch["eeg"]), size):
        yield {key: value[start : start + size] for key, value in batch.items()}


def to_device(batch, device):
    """Transfer a logical batch once; pinned loader tensors can copy asynchronously."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def collate(samples):
    return {
        "eeg": torch.stack([sample["eeg"] for sample in samples]),
        "channel_ids": torch.stack([sample["channel_ids"] for sample in samples]),
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "window_id": [sample["window_id"] for sample in samples],
    }
