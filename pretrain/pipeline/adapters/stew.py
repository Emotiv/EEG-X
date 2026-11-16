"""Native STEW TXT recordings with stable participant and condition identities."""

from pathlib import Path
import re

import numpy as np

from pretrain.channel_identity import EMOTIV_EPOC_CHANNEL_NAMES, sha256_file


class Stew:
    task, window_samples, channels = "stew", 256, list(EMOTIV_EPOC_CHANNEL_NAMES)

    def __init__(self, root, subjects=None):
        self.root = Path(root).resolve()
        selected = list(range(1, 49)) if subjects is None else sorted(set(subjects))
        found = {}
        for path in sorted(self.root.iterdir()):
            match = re.fullmatch(r"sub(\d{2})_(hi|lo)\.txt", path.name, flags=re.IGNORECASE)
            if match:
                found[int(match[1]), match[2].lower()] = path
        missing = [
            (subject, condition)
            for subject in selected
            for condition in ("hi", "lo")
            if (subject, condition) not in found
        ]
        if missing:
            raise ValueError(
                f"STEW raw recordings missing for {len(missing)} participant/condition pairs"
            )
        self.paths = [found[key] for key in sorted(found) if key[0] in selected]
        self.label_evidence = {
            "status": "verified",
            "kind": "native_recording_condition",
            "source": "https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset",
            "classes": {"0": "low_workload", "1": "high_workload"},
            "subjects": selected,
            "complete_subject_scope": selected == list(range(1, 49)),
        }

    def recordings(self):
        for path in self.paths:
            match = re.fullmatch(r"sub(\d{2})_(hi|lo)\.txt", path.name, flags=re.IGNORECASE)
            signal = np.loadtxt(path, dtype=np.float64)
            if signal.ndim != 2 or signal.shape[1] != len(self.channels):
                raise ValueError(
                    f"STEW recording does not contain the ordered EEG channels: {path}"
                )
            yield {
                "volts": signal.T * 1e-6,
                "sample_rate": 128.0,
                "subject": match[1],
                "recording": path.stem,
                "label": int(match[2].lower() == "hi"),
                "source": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "native_units": "microvolts",
                    "version": "native_STEW_TXT",
                },
            }
