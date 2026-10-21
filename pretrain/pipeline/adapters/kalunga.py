"""Author-labelled SSVEP recordings, including rest and corrected event meanings."""

import json
import re
from pathlib import Path

import mne
import numpy as np

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.download import digest


CHANNELS = ["Oz", "O1", "O2", "PO3", "POz", "PO7", "PO8", "PO4"]


class Kalunga:
    task, window_samples, channels = "kalunga", 256, CHANNELS

    def __init__(self, root):
        self.root = Path(root).resolve()
        record = self.root / "record.json"
        self.metadata = json.loads(record.read_text())
        if self.metadata["id"] != 2392979:
            raise ValueError("Kalunga author deposit identity changed")
        self.label_evidence = {
            "status": "verified",
            "kind": "author_deposit_fif_stimulus_codes",
            "source": "https://zenodo.org/records/2392979",
            "corrected_loader": "https://raw.githubusercontent.com/NeuroTechX/moabb/555cc3b/moabb/datasets/ssvep_exo.py",
            "author_metadata_sha256": sha256_file(record),
            "classes": {"0": "rest", "1": "13_Hz", "2": "21_Hz", "3": "17_Hz"},
            "native_codes": {"1": "rest", "2": "13_Hz", "3": "21_Hz", "4": "17_Hz"},
            "crop_seconds_after_event": [2, 4],
            "subject_authority": "author deposit participant-labelled filenames",
            "duplicates": "exclude every copy of exact recording or emitted-window duplicate groups",
            "scope": "new reduced-data evaluation; not the original paper sample contract",
        }

    def recordings(self):
        for item in sorted(self.metadata["files"], key=lambda f: f["key"]):
            match = re.fullmatch(r"subject(0[1-9]|1[0-2])_run[1-5]_raw\.fif", item["key"])
            if match is None:
                raise ValueError("Unknown author participant filename")
            path = self.root / item["key"]
            if path.stat().st_size != item["size"] or digest(path) != item["checksum"]:
                raise ValueError("Kalunga raw recording differs from author deposit")
            with mne.io.read_raw_fif(path, preload=False, verbose=False) as raw:
                eeg = [
                    name
                    for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
                    if kind == "eeg"
                ]
                if eeg != self.channels or raw.info["sfreq"] != 256:
                    raise ValueError("Native Kalunga EEG order or sampling changed")
                events = mne.find_events(
                    raw, stim_channel="STI 014", shortest_event=1, initial_event=True, verbose=False
                )
                if not len(events) or not np.isin(events[:, 2], [1, 2, 3, 4]).all():
                    raise ValueError("Unknown native Kalunga event code")
                windows = [
                    (int(round(((start - raw.first_samp) / 256 + 2) * 128)), int(code - 1))
                    for start, _, code in events
                ]
                yield {
                    "volts": raw.get_data(picks=self.channels),
                    "sample_rate": 256,
                    "subject": match[1],
                    "recording": path.stem,
                    "events": windows,
                    "source": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "native_units": "volts",
                        "version": "zenodo_2392979",
                    },
                }
