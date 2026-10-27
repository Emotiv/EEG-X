"""Native DREAMER arousal ratings with the selected greater-than-three rule."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.artifacts import inspect_artifact


def recover_labels(raw, artifact, experiment_config=None):
    data = inspect_artifact(artifact)
    participants = raw["Data"]
    scores, subjects = [], []
    for number, subject in enumerate(participants, 1):
        for signal, score in zip(subject["EEG"]["stimuli"], subject["ScoreArousal"], strict=True):
            count = max(0, (len(signal) - 256) // 64 + 1)
            scores.extend([int(score)] * count)
            subjects.extend([number] * count)
    scores, subjects = np.asarray(scores), np.asarray(subjects)
    # This is the archived producer's data order, not a new label threshold.
    order = np.random.RandomState(1234).permutation(len(scores))
    scores, subjects = scores[order], subjects[order]
    candidates = set(np.unique(scores).tolist())
    evidence_hash = hashlib.sha256()
    checked = 0
    ratings_match = True
    for split in ("train", "val", "test"):
        ids = data[f"id_{split}"].read(artifact)
        labels = data[f"Y_{split}"].read(artifact)
        chosen = np.isin(subjects, np.unique(ids))
        if not np.array_equal(subjects[chosen], ids):
            raise ValueError("DREAMER artifact membership/order does not match the raw producer")
        candidates = {
            cut for cut in candidates if np.array_equal((scores[chosen] > cut).astype(int), labels)
        }
        ratings_match = ratings_match and np.array_equal(scores[chosen], labels)
        evidence_hash.update(np.ascontiguousarray(ids).tobytes())
        evidence_hash.update(np.ascontiguousarray(labels).tobytes())
        checked += len(labels)
    attribution = None
    if ratings_match and experiment_config is not None:
        configuration = json.loads(Path(experiment_config).read_text())
        expected = Path(configuration["data_dir"]) / f"{configuration['problem']}.npy"
        if (
            configuration["problem"] != "DREAMER_Arousal"
            or expected.resolve() != Path(artifact).resolve()
            or configuration.get("data_provider") != "local"
        ):
            raise ValueError("Archived experiment does not identify this arousal artifact")
        loader = Path(__file__).resolve().parents[2] / "datasets" / "loader.py"
        candidates = {3}
        attribution = {
            "configuration_sha256": sha256_file(experiment_config),
            "loader_sha256": sha256_file(loader),
            "scope": "archived ratings audit; threshold three is the selected study rule",
        }
    if len(candidates) != 1:
        raise ValueError("No unique arousal mapping matches every archived label")
    cutoff = candidates.pop()
    return {
        "status": "verified",
        "field": "ScoreArousal",
        "rule": "greater_than",
        "threshold": cutoff,
        "classes": {"0": "low_arousal", "1": "high_arousal"},
        "evidence": "exact match of producer ordering, subject IDs and saved labels or ratings",
        "experiment_attribution": attribution,
        "checked_windows": checked,
        "label_payload_sha256": evidence_hash.hexdigest(),
        "artifact": str(Path(artifact).resolve()),
    }


class Dreamer:
    task, window_samples = "dreamer_arousal", 256

    def __init__(self, raw_path, artifact=None, experiment_config=None):
        self.path = Path(raw_path)
        self.raw = loadmat(self.path, simplify_cells=True)["DREAMER"]
        self.channels = [str(name) for name in self.raw["EEG_Electrodes"]]
        self.label_evidence = {
            "status": "verified",
            "kind": "native_ratings_and_selected_study_rule",
            "field": "ScoreArousal",
            "rule": "greater_than",
            "threshold": 3,
            "classes": {"0": "low_arousal", "1": "high_arousal"},
            "source": "https://zenodo.org/records/546113",
            "scope": "threshold selected by user; not an asserted manuscript threshold",
        }
        if artifact is not None:
            audit = recover_labels(self.raw, artifact, experiment_config)
            if audit["threshold"] != 3:
                raise ValueError("Archived DREAMER labels differ from the selected native rule")
            self.label_evidence["optional_archive_audit"] = audit
        self.source = {
            "path": str(self.path.resolve()),
            "sha256": sha256_file(self.path),
            "version": str(self.raw.get("Version", "source_mat")),
            "native_units": "microvolts",
        }

    def recordings(self):
        threshold = self.label_evidence["threshold"]
        for subject, participant in enumerate(self.raw["Data"], 1):
            for video, (signal, rating) in enumerate(
                zip(participant["EEG"]["stimuli"], participant["ScoreArousal"], strict=True), 1
            ):
                if not np.isfinite(rating) or int(rating) != rating or not 1 <= int(rating) <= 5:
                    raise ValueError("Invalid native DREAMER arousal rating")
                yield {
                    "volts": np.asarray(signal).T * 1e-6,
                    "sample_rate": float(self.raw["EEG_SamplingRate"]),
                    "subject": str(subject),
                    "recording": f"subject-{subject}/video-{video}",
                    "source": self.source,
                    "label": int(rating > threshold),
                }
