"""Source-aligned motor imagery trials from the BNCI Competition IV-2a MAT files."""

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pretrain.channel_identity import sha256_file


CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


class Bciciv:
    task, window_samples, channels = "bciciv_2a", 256, CHANNELS
    label_evidence = {
        "status": "verified",
        "kind": "official_trial_annotations",
        "classes": {"0": "feet", "1": "left_hand", "2": "right_hand", "3": "tongue"},
        "source": "https://www.bbci.de/competition/iv/desc_2a.pdf",
        "mat_authority": "https://raw.githubusercontent.com/NeuroTechX/moabb/v1.2.0/moabb/datasets/bnci.py",
        "crop_seconds_after_trial_start": [2, 4],
        "trial_indices": "one_based",
        "sessions": "T and E remain with the same native participant",
        "artifact_flags": "retain all annotated trials; no primary artifact rejection",
    }

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = sorted(self.root.glob("A??[TE].mat"))
        if len(self.paths) != 18:
            raise ValueError("BCICIV-2a requires both sessions of every native participant")

    def recordings(self):
        for path in self.paths:
            match = re.fullmatch(r"A(0[1-9])([TE])\.mat", path.name)
            if match is None:
                raise ValueError("Unknown native participant or session")
            runs = np.atleast_1d(loadmat(path, simplify_cells=True)["data"])
            source = {
                "path": str(path),
                "sha256": sha256_file(path),
                "native_units": "microvolts",
                "version": "BNCI2014_001_mat",
            }
            for number, run in enumerate(runs):
                trials = np.atleast_1d(run["trial"]).astype(int)
                if not trials.size:
                    continue
                labels = np.atleast_1d(run["y"]).astype(int)
                rate = float(run["fs"])
                signal = np.asarray(run["X"])
                if rate != 250 or signal.shape[1] != 25 or len(labels) != len(trials):
                    raise ValueError("Native BCI signal or trial contract changed")
                if not np.isin(labels, [1, 2, 3, 4]).all() or (trials < 1).any():
                    raise ValueError("Invalid native BCI class or trial onset")
                # Preprocess each acquisition run before cutting windows. Offsets map
                # native one-based trial starts onto the pinned resampled timeline.
                events = [
                    (
                        int(round(((onset - 1) / rate + 2) * 128)),
                        {1: 1, 2: 2, 3: 0, 4: 3}[int(label)],
                    )
                    for onset, label in zip(trials, labels, strict=True)
                ]
                yield {
                    "volts": signal[:, :22].T * 1e-6,
                    "sample_rate": rate,
                    "subject": match[1],
                    "recording": f"{path.stem}/run-{number}",
                    "source": source,
                    "events": events,
                }
