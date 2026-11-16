"""Stream prepared windows to a single immutable, self-contained HDF5 file."""

from collections import Counter
import json
import os
from pathlib import Path

import h5py
import numpy as np

from pretrain.channel_identity import load_channel_vocabulary
from pretrain.metadata import fingerprint
from .catalog import validate_reference
from .groups import subject_groups
from .schema import (
    INDEX_DTYPE,
    SEEDS,
    array_hash,
    portable,
    recording_id,
    split_bank,
)


class PreparedWriter:
    def __init__(
        self,
        output,
        *,
        task,
        window_samples,
        channels,
        label_evidence,
        preprocessing,
        sampling_policy=None,
    ):
        if not label_evidence or label_evidence.get("status") != "verified":
            raise ValueError("Verified label evidence is required")
        self.path = Path(output).resolve()
        if self.path.suffix not in {".h5", ".hdf5"}:
            raise ValueError("Prepared dataset output must be one .h5 file")
        if self.path.exists():
            raise FileExistsError(f"Dataset already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.samples = int(window_samples)
        if self.samples < 32 or self.samples % 32:
            raise ValueError("Window samples must be a positive patch multiple")
        self.task, self.records, self.events, self.excluded = task, [], [], []
        vocabulary = load_channel_vocabulary()
        ids = vocabulary.resolve_names(channels).tolist()
        if len(set(ids)) != len(ids):
            raise ValueError("Physical electrode identities must be unique")
        positions = {record.index: record.position for record in vocabulary.channels}
        self.metadata = {
            "schema_version": 2,
            "task": task,
            "sample_rate": 128,
            "window_samples": self.samples,
            "channels": list(channels),
            "channel_ids": ids,
            "channel_xyz": [list(positions[i]) for i in ids],
            "vocabulary_sha256": vocabulary.fingerprint,
            "label_evidence": portable(label_evidence),
            "preprocessing": preprocessing,
            "sampling": sampling_policy
            or {
                "name": "explicit_windows_v1",
                "overlap": "preserved",
                "duplicates": "retained_with_grouped_subject_splits",
                "identity": "record_crop_label_occurrence_v1",
            },
        }
        self.temporary = self.path.with_suffix(self.path.suffix + ".working")
        self._owns_temporary = False
        self.file = h5py.File(self.temporary, "x")
        self._owns_temporary = True
        self.file.create_dataset(
            "X",
            shape=(0, len(ids), self.samples),
            maxshape=(None, len(ids), self.samples),
            chunks=(1, len(ids), self.samples),
            compression="lzf",
            shuffle=True,
            fletcher32=True,
            dtype="<f4",
        )

    def add(
        self, signal, *, subject, recording, source, label=None, events=None, official_split=None
    ):
        if not subject or not recording or not source or not source.get("sha256"):
            raise ValueError("Source-backed subject, recording and source digest required")
        if label is None and events is None:
            raise ValueError("Recording needs verified labels or annotated events")
        recording = recording_id(recording)
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[0] != len(self.metadata["channels"]):
            raise ValueError("Recording does not match ordered channel contract")
        if not np.isfinite(signal).all():
            raise ValueError("Nonfinite prepared signal")
        number = len(self.records)
        identity = fingerprint([self.task, str(subject), str(recording), source["sha256"]])
        record = {
            "id": identity,
            "subject": str(subject),
            "recording": str(recording),
            "source": portable(source),
            "samples": signal.shape[1],
            "sha256": array_hash(signal),
            "official_split": official_split,
            "label": int(label) if label is not None else None,
        }
        if any(record["id"] == previous["id"] for previous in self.records):
            raise ValueError(f"Duplicate recording identity: {recording}")
        self.records.append(record)
        stride = self.metadata["sampling"].get("stride_samples") or self.samples
        if int(stride) != stride or stride < 1:
            raise ValueError("Window stride must be a positive integer")
        candidates = (
            events
            if events is not None
            else [
                (start, int(label))
                for start in range(0, signal.shape[1] - self.samples + 1, stride)
            ]
        )
        record["event_count"] = len(candidates)
        covered = candidates[-1][0] + self.samples if candidates else 0
        if events is None and signal.shape[1] > covered:
            self.excluded.append(
                {
                    "recording": str(recording),
                    "start": covered,
                    "samples": signal.shape[1] - covered,
                    "reason": "incomplete_tail",
                }
            )
        eligible = []
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
                eligible.append((int(start), int(target)))
        if not eligible:
            return
        first = len(self.events)
        self.file["X"].resize(first + len(eligible), axis=0)
        occurrences = Counter()
        for offset, (start, target) in enumerate(eligible):
            values = np.ascontiguousarray(signal[:, start : start + self.samples])
            digest = array_hash(values)
            occurrence = occurrences[start, target]
            occurrences[start, target] += 1
            window_id = fingerprint(
                [identity, start, self.samples, self.metadata["sample_rate"], target, occurrence]
            )
            self.file["X"][first + offset] = values
            self.events.append((number, start, target, window_id.encode(), digest.encode()))

    def finish(self, *, seed=241, finalize=None):
        from .reader import DatasetReference, read_metadata

        publishing = self.path.with_suffix(self.path.suffix + ".partial")
        owns_publishing = False
        try:
            if not self.events:
                raise ValueError("No eligible windows")
            index = np.array(self.events, dtype=INDEX_DTYPE)
            old_rows = np.arange(len(index))
            order = sorted(
                range(len(self.records)),
                key=lambda i: (
                    self.records[i]["recording"],
                    self.records[i]["subject"],
                    self.records[i]["id"],
                ),
            )
            remap = {old: new for new, old in enumerate(order)}
            records = [self.records[i] for i in order]
            old_rows = np.asarray(
                sorted(
                    old_rows,
                    key=lambda row: (
                        remap[int(index[row]["record"])],
                        int(index[row]["start"]),
                        int(index[row]["label"]),
                        bytes(index[row]["window_id"]),
                    ),
                )
            )
            retained = index[old_rows].copy()
            retained["record"] = [remap[int(i)] for i in retained["record"]]
            classes = np.unique(retained["label"])
            if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 2:
                raise ValueError("Classes must use an explicit contiguous zero-based mapping")
            groups = subject_groups(records, retained)
            banks = {}
            for split_seed in sorted(set(SEEDS) | {seed}):
                banks[f"subject_holdout/{split_seed}"] = split_bank(
                    records, retained, split_seed, "subject_holdout", groups
                )
            if self.task == "tuab":
                banks["tuab_source/241"] = split_bank(records, retained, 241, "tuab_source", groups)
            assignments, selected, _ = banks[f"subject_holdout/{seed}"]
            metadata = {
                **self.metadata,
                "records": records,
                "subjects": assignments,
                "subject_groups": groups,
                "split_seed": seed,
                "source_subjects": sorted({record["subject"] for record in records}),
                "split_policy": "grouped_subject_70_10_20_v1",
                "classes": classes.tolist(),
                "windows_sha256": array_hash(retained),
                "counts": {role: len(rows) for role, rows in selected.items()},
                "excluded": sorted(self.excluded, key=lambda row: json.dumps(row, sort_keys=True)),
                "split_banks": {key: value[2] for key, value in sorted(banks.items())},
            }
            validate_reference(metadata, retained)
            if finalize:
                metadata = finalize(metadata)
            metadata = portable(metadata)
            metadata.pop("fingerprint", None)
            metadata["fingerprint"] = fingerprint(metadata)
            self.file.flush()
            output = h5py.File(publishing, "x")
            owns_publishing = True
            with output:
                x = output.create_dataset(
                    "X",
                    shape=(len(retained), len(self.metadata["channels"]), self.samples),
                    chunks=(1, len(self.metadata["channels"]), self.samples),
                    compression="lzf",
                    shuffle=True,
                    fletcher32=True,
                    dtype="<f4",
                )
                for begin in range(0, len(old_rows), 128):
                    rows = old_rows[begin : begin + 128]
                    permutation = np.argsort(rows)
                    values = self.file["X"][rows[permutation]]
                    x[begin : begin + len(rows)] = values[np.argsort(permutation)]
                output.create_dataset("y", data=retained["label"], fletcher32=True)
                output.create_dataset(
                    "window_index", data=retained, compression="lzf", fletcher32=True
                )
                strings = h5py.string_dtype("utf-8")
                output.create_dataset(
                    "metadata", data=json.dumps(metadata, sort_keys=True), dtype=strings
                )
                output.create_dataset(
                    "records",
                    data=[json.dumps(record, sort_keys=True) for record in records],
                    dtype=strings,
                )
                output.create_dataset("subjects", data=metadata["source_subjects"], dtype=strings)
                channels = output.create_group("channels")
                channels.create_dataset("names", data=metadata["channels"], dtype=strings)
                channels.create_dataset("ids", data=metadata["channel_ids"])
                channels.create_dataset("xyz", data=metadata["channel_xyz"])
                for key, (subjects, roles, digest) in banks.items():
                    group = output.require_group(f"splits/{key}")
                    group.attrs["subjects"] = json.dumps(subjects, sort_keys=True)
                    group.attrs["fingerprint"] = digest
                    for role, rows in roles.items():
                        group.create_dataset(f"{role}_idx", data=rows, fletcher32=True)
            validated = read_metadata(DatasetReference(publishing, seed))
            os.link(publishing, self.path)
            publishing.unlink()
            return validated
        finally:
            self.file.close()
            self.temporary.unlink(missing_ok=True)
            self._owns_temporary = False
            if owns_publishing:
                publishing.unlink(missing_ok=True)

    def close(self):
        if getattr(self, "file", None):
            self.file.close()
        if getattr(self, "_owns_temporary", False):
            self.temporary.unlink(missing_ok=True)
            self._owns_temporary = False

    def __del__(self):
        self.close()
