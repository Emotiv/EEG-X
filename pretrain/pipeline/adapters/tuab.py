"""Official TUAB directory labels and native participant identifiers."""

import re
from pathlib import Path

import mne

from pretrain.channel_identity import sha256_file


CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]


class Tuab:
    task, window_samples, channels = "tuab", 1280, CHANNELS
    label_evidence = {
        "status": "verified",
        "kind": "official_tuab_directory_labels",
        "classes": {"0": "normal", "1": "abnormal"},
        "source": "https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
        "version": "3.0.1",
        "subject_authority": "native TUAB EDF participant prefix",
    }

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = sorted(self.root.rglob("*.edf"))
        if not self.paths:
            raise ValueError("TUAB contains no readable EDF recordings")

    def recordings(self):
        for path in self.paths:
            parts = path.relative_to(self.root).parts
            if (
                len(parts) != 4
                or parts[0] not in {"train", "eval"}
                or parts[1] not in {"normal", "abnormal"}
            ):
                raise ValueError(f"Unexpected TUAB source layout: {path}")
            match = re.fullmatch(r"([a-z]{8})_s\d+_t\d+\.edf", path.name)
            if match is None:
                raise ValueError("TUAB participant identity does not match native naming contract")
            with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
                names = {
                    name.removeprefix("EEG ").removesuffix("-REF").upper(): name
                    for name in raw.ch_names
                }
                if any(name.upper() not in names for name in self.channels):
                    raise ValueError(f"TUAB recording lacks the verified reference montage: {path}")
                picks = [names[name.upper()] for name in self.channels]
                yield {
                    "volts": raw.get_data(picks=picks),
                    "sample_rate": raw.info["sfreq"],
                    "subject": match[1],
                    "recording": str(path.relative_to(self.root)),
                    "official_split": "test" if parts[0] == "eval" else "train",
                    "source": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "native_units": "volts",
                    },
                    "label": int(parts[1] == "abnormal"),
                }
