"""HDF5 window datasets and deterministic PyTorch batching."""

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from pretrain.pipeline.dataset import (
    INDEX_DTYPE as INDEX_DTYPE,
    SPLITS as SPLITS,
    TASK_WINDOWS as TASK_WINDOWS,
    PreparedWriter as PreparedWriter,
    Windows as Windows,
    subject_split as subject_split,
)


class _StoppableBatchSampler(BatchSampler):
    """Stop scheduling new batches while existing worker tasks finish."""

    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=False)
        self.stopped = False

    def stop(self):
        self.stopped = True

    def __iter__(self):
        iterator = super().__iter__()
        while not self.stopped:
            try:
                yield next(iterator)
            except StopIteration:
                return


class _WindowBatches:
    """Own loader workers until scheduled batches have been collected."""

    def __init__(self, view, *, batch_size, seed, workers, pin_memory, shuffle):
        self.dataset = view
        self.batch_size = batch_size
        self.num_workers = workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sampler = (RandomSampler(self.dataset, generator=self.generator)
                   if self.shuffle else SequentialSampler(self.dataset))
        batches = _StoppableBatchSampler(sampler, self.batch_size)
        loader = DataLoader(
            self.dataset,
            batch_sampler=batches,
            collate_fn=collate,
            generator=self.generator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            multiprocessing_context="spawn" if self.num_workers else None,
        )
        iterator = iter(loader)
        try:
            yield from iterator
        except GeneratorExit:
            # Do not dispatch more indices. Collect only already scheduled work
            # so workers can finish serializing batches before they exit.
            batches.stop()
            for _ in iterator:
                pass
            raise


def batches(view, *, batch_size, seed, workers=0, pin_memory=False, shuffle=True):
    """Keep sample order deterministic and close worker queues on early exit."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if type(workers) is not int or workers < 0:
        raise ValueError("Worker count must be a nonnegative integer")
    return _WindowBatches(view, batch_size=batch_size, seed=seed, workers=workers,
                          pin_memory=pin_memory, shuffle=shuffle)


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
