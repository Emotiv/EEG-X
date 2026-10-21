"""Immutable prepared recordings, subject membership, and window views."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pretrain.channel_identity import load_channel_vocabulary, sha256_file
from pretrain.metadata import atomic_json, fingerprint


INDEX_DTYPE = np.dtype(
    [
        ("record", "<u4"),
        ("start", "<u8"),
        ("label", "<i4"),
        ("split", "u1"),
    ]
)
SPLITS = {"train": 0, "validation": 1, "test": 2}


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


def subject_split(records, seed=241):
    subjects = {record["subject"] for record in records}
    if len(subjects) < 3:
        raise ValueError("Independent train/validation/test require at least three subjects")
    ordered = sorted(subjects, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    # The revised contract pools old partitions, including official test subjects.
    test_count = max(1, round(len(ordered) * 0.2))
    test, pool = ordered[:test_count], ordered[test_count:]
    validation_count = max(1, round(len(ordered) * 0.1))
    validation = pool[: min(validation_count, len(pool) - 1)]
    train = pool[len(validation) :]
    return {
        s: role
        for role, group in (("train", train), ("validation", validation), ("test", test))
        for s in group
    }


class PreparedWriter:
    """Publish recordings and source-backed labels once, with explicit exclusions."""

    def __init__(self, output, *, task, window_samples, channels, label_evidence, preprocessing):
        if not label_evidence or label_evidence.get("status") != "verified":
            raise ValueError("Verified label evidence is required")
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.records, self.events, self.excluded = [], [], []
        self.task, self.samples = task, int(window_samples)
        if self.samples < 32 or self.samples % 32:
            raise ValueError("Window samples must be a positive patch multiple")
        vocabulary = load_channel_vocabulary()
        self.metadata = {
            "schema_version": 1,
            "task": task,
            "sample_rate": 128,
            "window_samples": self.samples,
            "channels": list(channels),
            "channel_ids": vocabulary.resolve_names(channels).tolist(),
            "vocabulary_sha256": vocabulary.fingerprint,
            "label_evidence": label_evidence,
            "preprocessing": preprocessing,
        }

    def add(
        self, signal, *, subject, recording, source, label=None, events=None, official_split=None
    ):
        if not subject or not recording or not source or not source.get("sha256"):
            raise ValueError("Source-backed subject, recording and source digest required")
        if label is None and events is None:
            raise ValueError("Recording needs verified labels or annotated events")
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[0] != len(self.metadata["channels"]):
            raise ValueError("Recording does not match ordered channel contract")
        if not np.isfinite(signal).all():
            raise ValueError("Nonfinite prepared signal")
        number = len(self.records)
        path = self.root / f"record-{number:06d}.npy"
        np.save(path, signal)
        record = {
            "path": path.name,
            "sha256": sha256_file(path),
            "subject": str(subject),
            "recording": str(recording),
            "source": source,
            "samples": signal.shape[1],
            "official_split": official_split,
        }
        candidates = (
            events
            if events is not None
            else [
                (start, int(label))
                for start in range(0, signal.shape[1] - self.samples + 1, self.samples)
            ]
        )
        accepted = []
        for start, target in candidates:
            if int(start) != start or int(target) != target or int(target) < 0:
                raise ValueError("Window offset and class must be nonnegative integers")
            if start < 0 or start + self.samples > signal.shape[1]:
                self.excluded.append(
                    {
                        "recording": str(recording),
                        "start": int(start),
                        "reason": "outside_recording",
                    }
                )
            else:
                accepted.append((number, int(start), int(target)))
        self.records.append(record)
        self.events.extend(accepted)

    def finish(self, *, seed=241):
        if not self.events:
            raise ValueError("No eligible windows")
        groups = {}
        for number, record in enumerate(self.records):
            groups.setdefault(record["sha256"], []).append(number)
        duplicate_records = {i for group in groups.values() if len(group) > 1 for i in group}
        valid_events = [event for event in self.events if event[0] not in duplicate_records]
        for number in sorted(duplicate_records):
            self.excluded.append(
                {"recording": self.records[number]["recording"], "reason": "duplicate_signal"}
            )
        if not valid_events:
            raise ValueError("All windows are duplicates")
        hashes = {}
        current_number, signal = None, None
        for position, (number, start, _) in enumerate(valid_events):
            if number != current_number:
                signal = np.load(self.root / self.records[number]["path"], mmap_mode="r")
                current_number = number
            payload = np.ascontiguousarray(signal[:, start : start + self.samples])
            digest = hashlib.sha256(payload.tobytes()).digest()
            hashes.setdefault(digest, []).append(position)
        duplicate_windows = {i for group in hashes.values() if len(group) > 1 for i in group}
        for position in sorted(duplicate_windows):
            number, start, _ = valid_events[position]
            self.excluded.append(
                {
                    "recording": self.records[number]["recording"],
                    "start": start,
                    "reason": "duplicate_window",
                }
            )
        valid_events = [event for i, event in enumerate(valid_events) if i not in duplicate_windows]
        if not valid_events:
            raise ValueError("All emitted windows are duplicates")
        used = {event[0] for event in valid_events}
        assignments = subject_split([r for i, r in enumerate(self.records) if i in used], seed)
        index = np.array(
            [
                (*event, SPLITS[assignments[self.records[event[0]]["subject"]]])
                for event in valid_events
            ],
            dtype=INDEX_DTYPE,
        )
        classes = np.unique(index["label"])
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Classes must use an explicit contiguous zero-based mapping")
        for name, split in SPLITS.items():
            labels = np.unique(index["label"][index["split"] == split])
            if len(labels) < 2 or not np.array_equal(labels, classes):
                raise ValueError(f"Split {name} lacks classification label coverage")
        np.save(self.root / "windows.npy", index)
        metadata = {
            **self.metadata,
            "records": self.records,
            "subjects": assignments,
            "split_seed": seed,
            "split_policy": "pooled_subject_70_10_20_v1",
            "windows_sha256": sha256_file(self.root / "windows.npy"),
            "excluded": self.excluded,
            "classes": classes.tolist(),
            "counts": {
                name: int((index["split"] == split).sum()) for name, split in SPLITS.items()
            },
        }
        metadata["fingerprint"] = fingerprint(metadata)
        atomic_json(self.root / "manifest.json", metadata)
        return metadata


class Windows(Dataset):
    def __init__(self, manifest, split, *, verify_signals=True):
        self.path = Path(manifest).resolve()
        self.metadata = data = json.loads(self.path.read_text())
        if data.get("schema_version") != 1 or data.get("fingerprint") != fingerprint(
            {k: v for k, v in data.items() if k != "fingerprint"}
        ):
            raise ValueError("Invalid paper dataset manifest")
        if data["label_evidence"].get("status") != "verified":
            raise ValueError("Unverified task labels")
        if data["vocabulary_sha256"] != load_channel_vocabulary().fingerprint:
            raise ValueError("Channel vocabulary changed")
        if sha256_file(self.path.parent / "windows.npy") != data["windows_sha256"]:
            raise ValueError("Window index changed")
        self.index = np.load(self.path.parent / "windows.npy", mmap_mode="r")
        if self.index.dtype != INDEX_DTYPE:
            raise ValueError("Wrong window index format")
        self.rows = np.flatnonzero(self.index["split"] == SPLITS[split])
        self.split, self.cache = split, OrderedDict()
        self.ids = torch.tensor(data["channel_ids"], dtype=torch.long)
        if verify_signals:
            for number in np.unique(self.index["record"][self.rows]):
                record = data["records"][int(number)]
                if sha256_file(self.path.parent / record["path"]) != record["sha256"]:
                    raise ValueError("Prepared recording changed")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.index[self.rows[index]]
        number, start = int(row["record"]), int(row["start"])
        record = self.metadata["records"][number]
        if self.metadata["subjects"][record["subject"]] != self.split:
            raise ValueError("Window crosses declared subject membership")
        if number not in self.cache:
            self.cache[number] = np.load(self.path.parent / record["path"], mmap_mode="r")
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        values = np.array(
            self.cache[number][:, start : start + self.metadata["window_samples"]], copy=True
        )
        expected = (len(self.ids), self.metadata["window_samples"])
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError("Invalid prepared window")
        identity = fingerprint([self.metadata["fingerprint"], record["recording"], start])
        return {
            "eeg": torch.from_numpy(values),
            "channel_ids": self.ids,
            "label": int(row["label"]),
            "window_id": identity,
        }

    def __getstate__(self):
        return {**self.__dict__, "cache": OrderedDict()}


def collate(samples):
    return {
        "eeg": torch.stack([s["eeg"] for s in samples]),
        "channel_ids": torch.stack([s["channel_ids"] for s in samples]),
        "label": torch.tensor([s["label"] for s in samples], dtype=torch.long),
        "window_id": [s["window_id"] for s in samples],
    }
