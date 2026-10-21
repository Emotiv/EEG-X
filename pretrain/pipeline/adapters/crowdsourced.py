"""Author-defined participants, native timestamps, and eyes-open/closed markers."""

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from pretrain.channel_identity import sha256_file


CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
LABELS = {"eyesclose_element": 0, "eyesopen_element": 1}


def intervals(metadata):
    result = []
    for marker in metadata["markers"]:
        data = marker.get("data", {})
        if data.get("label") not in LABELS:
            continue
        times = [
            datetime.fromisoformat(data[key].replace("Z", "+00:00"))
            for key in ("startDatetime", "endDatetime")
        ]
        if any(value.utcoffset() is None for value in times) or times[1] <= times[0]:
            raise ValueError("Ambiguous source event timezone or duration")
        result.append((times[0].timestamp(), times[1].timestamp(), LABELS[data["label"]]))
    result.sort()
    if len(result) != 2 or {row[2] for row in result} != {0, 1}:
        raise ValueError("Author recording does not contain both source conditions")
    if result[0][1] > result[1][0]:
        raise ValueError("Overlapping eyes-open/closed markers")
    return result


def read_recording(path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        names = next(csv.reader(handle))
        lower = [name.lower() for name in names]
        required = ["timestamp", "eeg.counter", "eeg.interpolated"] + [
            "eeg." + c.lower() for c in CHANNELS
        ]
        if len(lower) != len(set(lower)) or any(name not in lower for name in required):
            raise ValueError("Native EEG channel or clock fields changed")
        data = np.loadtxt(handle, delimiter=",", usecols=[lower.index(name) for name in required])
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Invalid native Crowdsourced signal or timestamp")
    delta = np.diff(data[:, 0])
    if len(data) < 2 or not (delta > 0).any():
        raise ValueError("Native source clock does not advance")
    median = float(np.median(delta[delta > 0]))
    rate = min((128, 256), key=lambda candidate: abs(median - 1 / candidate))
    if abs(median * rate - 1) > 0.01:
        raise ValueError("Timestamps do not support the native sampling contract")
    modulus = int(data[:, 1].max()) + 1
    # The original EPOC export includes counter 128; its counter cycle has
    # 129 values although the nominal signal sampling rate remains 128 Hz.
    if (
        modulus not in ({128, 129} if rate == 128 else {256})
        or (data[:, 1] < 0).any()
        or not np.equal(data[:, 1], np.floor(data[:, 1])).all()
    ):
        raise ValueError("Unknown native EEG counter cycle")
    breaks = (delta <= 0) | (delta > 1.5 / rate) | ((np.diff(data[:, 1]) % modulus) != 1)
    edges = np.r_[0, np.flatnonzero(breaks) + 1, len(data)]
    return data, rate, modulus, list(zip(edges[:-1], edges[1:], strict=True))


def event_windows(times, conditions, rate):
    output = []
    for begin, end, label in conditions:
        first, last = np.searchsorted(times, [begin, end], side="left")
        start, stop = int(np.ceil(first * 128 / rate)), int(np.ceil(last * 128 / rate))
        output.extend((offset, label) for offset in range(start, stop - 256 + 1, 256))
    return output


class Crowdsourced:
    task, window_samples, channels = "crowdsourced", 256, CHANNELS

    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = self.root / "manifest.json"
        authority = json.loads(manifest.read_text())
        if authority["record"] != "9bvgh":
            raise ValueError("Wrong author Crowdsourced deposit")
        files = authority["files"]
        for item in files:
            name = item["attributes"]["name"]
            if (
                Path(name).name != name
                or sha256_file(self.root / name) != item["attributes"]["extra"]["hashes"]["sha256"]
            ):
                raise ValueError("Crowdsourced source differs from its author digest")
        readme = (self.root / "Readme.txt").read_text(encoding="utf-8-sig")
        if "per recording (per subject)" not in readme:
            raise ValueError("Author participant-to-recording authority is missing")
        self.paths = sorted(
            self.root / item["attributes"]["name"]
            for item in files
            if item["attributes"]["name"].endswith(".csv.gz")
        )
        if not self.paths:
            raise ValueError("No author EPOC recordings")
        self.label_evidence = {
            "status": "verified",
            "kind": "author_deposit_markers_and_participant_definition",
            "source": "https://osf.io/9bvgh/",
            "source_manifest_sha256": sha256_file(manifest),
            "owner_authority_sha256": sha256_file(self.root / "Readme.txt"),
            "owner_rule": "one author recording per subject, identified by its native filename stem",
            "classes": {"0": "eyes_closed", "1": "eyes_open"},
            "native_labels": LABELS,
            "source_units": "microvolts",
            "unit_authority": "https://emotiv.gitbook.io/emotivpro-v3/managing-your-eeg-data-recordings/exporting-an-eeg-data-recording/csv-files",
            "sampling_authority": "nominal rate checked against native timestamps and counter cycle",
            "gaps": "split acquisition at timestamp or counter gaps; never insert samples or join blocks",
            "device_interpolation": "retain and count interpolation already present in the author raw export",
            "window_rule": "source marker intervals mapped through native sample indices to non-overlapping 128Hz windows",
            "normalization_unit": "complete uninterrupted acquisition block before windowing",
            "record_audit": [],
        }

    def recordings(self):
        for path in self.paths:
            data, rate, modulus, blocks = read_recording(path)
            stem = path.name.removesuffix(".csv.gz")
            marker = self.root / (stem + ".json")
            conditions = intervals(json.loads(marker.read_text()))
            self.label_evidence["record_audit"].append(
                {
                    "recording": stem,
                    "native_rate": rate,
                    "counter_modulus": modulus,
                    "samples": len(data),
                    "continuous_blocks": len(blocks),
                    "device_interpolated_samples": int((data[:, 2] > 0).sum()),
                }
            )
            source = {
                "path": str(path),
                "sha256": sha256_file(path),
                "marker_sha256": sha256_file(marker),
                "native_units": "microvolts",
                "version": "OSF_9bvgh",
            }
            for first, last in blocks:
                times = data[first:last, 0]
                if len(times) < rate * 2:
                    continue
                events = event_windows(times, conditions, rate)
                if not events:
                    continue
                yield {
                    "volts": data[first:last, 3:].T * 1e-6,
                    "sample_rate": rate,
                    "subject": stem,
                    "recording": f"{stem}/samples-{first}-{last}",
                    "events": events,
                    "source": {
                        **source,
                        "first_native_sample": int(first),
                        "last_native_sample": int(last),
                        "first_timestamp": float(times[0]),
                        "last_timestamp": float(times[-1]),
                    },
                }
