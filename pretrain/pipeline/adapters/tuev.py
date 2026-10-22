"""Native TUEV events with a binary map verified against archived labels."""

import hashlib
import os
import re
from pathlib import Path

import mne
import numpy as np
from scipy.signal import lfilter

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.adapters.tuab import CHANNELS
from pretrain.pipeline.artifacts import inspect_artifact


NATIVE_MAP = {1: 1, 2: 1, 3: 1, 6: 0}
LEGACY_CHANNELS = CHANNELS[:14] + ["T6", "T5"] + CHANNELS[16:]


def native_identity(root, path):
    """Retain the source folder ID across native numeric and alphabetic releases."""
    relative = path.relative_to(root)
    parts = relative.parts
    if (
        len(parts) != 3
        or parts[0] not in {"train", "eval"}
        or re.fullmatch(r"(?:[a-z]{8}|[0-9]{3})", parts[1]) is None
    ):
        raise ValueError(f"Unknown native TUEV participant layout: {relative}")
    return {
        "subject": parts[1],
        "recording": str(relative),
        "official_split": "test" if parts[0] == "eval" else "train",
    }


def annotations(path):
    rows = [
        [float(value) for value in line.split(",")]
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if (
        not np.isfinite(values).all()
        or (values[:, 1] < 0).any()
        or (values[:, 2] < values[:, 1]).any()
        or not np.isin(values[:, 3], [1, 2, 3, 4, 5, 6]).all()
    ):
        raise ValueError("Invalid native TUEV annotation")
    return values


def legacy_order(root):
    """Replay the archived producer order for evidence, not for new splits."""
    for subject in os.listdir(root):
        folder = Path(root) / subject
        if folder.is_dir():
            for name in os.listdir(folder):
                if name.endswith(".edf"):
                    yield folder / name


def archive_signal(path):
    """Historical Emotiv filter used only to identify archived raw waveforms.

    This reproduces datasets/data_preparation/filtering.py. Preparation of new
    training signals uses the common paper profile, not this historical filter.
    Archived TUEV X arrays contain volts, despite the producer's microvolt comment.
    """
    with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
        raw.resample(128, verbose=False)
        values = raw.get_data(picks=["EEG " + c.upper() + "-REF" for c in LEGACY_CHANNELS]).T
    deltas = np.diff(values, axis=0, prepend=values[:1])
    limited = np.cumsum(np.clip(deltas, -30, 30), axis=0)
    filtered = lfilter(
        [0.98279471, -1.96558942, 0.98279471], [1.0, -1.96529337, 0.96588546], limited, axis=0
    )
    quartile = filtered.shape[1] // 4
    reference = np.sort(filtered, axis=1)[:, quartile:-quartile].mean(axis=1, keepdims=True)
    return (filtered - reference).T


def recover_map(native, saved):
    native, saved = np.asarray(native), np.asarray(saved)
    if native.shape != saved.shape or set(np.unique(native)) != set(NATIVE_MAP):
        raise ValueError("Archived TUEV labels do not cover the native event map")
    mapping = {}
    for code in sorted(NATIVE_MAP):
        targets = np.unique(saved[native == code])
        if len(targets) != 1:
            raise ValueError("A native TUEV class has conflicting archived binary labels")
        mapping[code] = int(targets[0])
    if mapping != NATIVE_MAP:
        raise ValueError("Archived TUEV semantics differ from the verified epileptiform task")
    return mapping


def verify_archive(root, artifact):
    artifact = Path(artifact)
    saved = inspect_artifact(artifact)
    labels = saved["Y_test"].read(artifact)
    signals = saved["X_test"].read(artifact)
    if labels.ndim != 1 or signals.shape != (len(labels), 19, 256):
        raise ValueError("Invalid archived TUEV label or signal geometry")
    codes, waves, inventory = [], [], []
    checked, cursor = set(), 0
    for path in legacy_order(Path(root) / "eval"):
        event_path = path.with_suffix(".rec")
        rows = annotations(event_path)
        rows = rows[np.isin(rows[:, 3], list(NATIVE_MAP))]
        if cursor + len(rows) > len(labels):
            raise ValueError("Native TUEV inventory exceeds the archived labels")
        inventory.append(
            {
                "recording": str(path.relative_to(root)),
                "annotation_sha256": sha256_file(event_path),
                "windows": len(rows),
            }
        )
        if any(int(row[3]) not in checked for row in rows):
            values = archive_signal(path)
            for index, row in enumerate(rows):
                code = int(row[3])
                if code in checked:
                    continue
                start = row[1] if row[1] < 0.5 else row[1] - 0.5
                first = int(start * 128)
                candidate = values[:, first : first + 256]
                expected = np.asarray(signals[cursor + index])
                if candidate.shape != expected.shape or not np.allclose(
                    candidate, expected, rtol=1e-6, atol=1e-12
                ):
                    raise ValueError("Native TUEV waveform does not match its archived event order")
                waves.append(
                    {
                        "native_class": code,
                        "saved_class": int(labels[cursor + index]),
                        "recording": str(path.relative_to(root)),
                        "start": first,
                        "max_error_volts": float(np.max(np.abs(candidate - expected))),
                    }
                )
                checked.add(code)
        codes.extend(int(row[3]) for row in rows)
        cursor += len(rows)
    mapping = recover_map(codes, labels)
    return {
        "status": "verified",
        "kind": "exact_archived_binary_event_map",
        "source": "https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
        "native_map": {str(k): v for k, v in mapping.items()},
        "classes": {"0": "background", "1": "epileptiform"},
        "excluded_native_classes": [4, 5],
        "checked_labels": len(labels),
        "label_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
        "archive": str(artifact.resolve()),
        "historical_record_order": inventory,
        "waveform_checks": waves,
        "archive_signal_units": "volts",
        "window_rule": "start at event_start if below 0.5 seconds; otherwise event_start minus 0.5 seconds; floor to 128Hz; take 256 samples",
        "overlap": "source event windows may overlap; exact duplicates are excluded",
    }


class Tuev:
    task, window_samples, channels = "tuev_binary", 256, CHANNELS

    def __init__(self, root, artifact):
        self.root = Path(root).resolve()
        self.paths = sorted(self.root.rglob("*.edf"))
        if not self.paths:
            raise ValueError("TUEV contains no native EDF recordings")
        self.sources = [(path, native_identity(self.root, path)) for path in self.paths]
        self.label_evidence = verify_archive(self.root, artifact)

    def recordings(self):
        for path, identity in self.sources:
            rows = annotations(path.with_suffix(".rec"))
            rows = rows[np.isin(rows[:, 3], list(NATIVE_MAP))]
            if not len(rows):
                continue
            events = [
                (int((row[1] if row[1] < 0.5 else row[1] - 0.5) * 128), NATIVE_MAP[int(row[3])])
                for row in rows
            ]
            with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
                picks = ["EEG " + channel.upper() + "-REF" for channel in self.channels]
                yield {
                    "volts": raw.get_data(picks=picks),
                    "sample_rate": raw.info["sfreq"],
                    **identity,
                    "events": events,
                    "source": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "annotation_sha256": sha256_file(path.with_suffix(".rec")),
                        "native_units": "volts",
                        "version": "native_TUEV_EDF_REC",
                    },
                }
