"""PhysioNet imagery EDF runs, retaining task-rest and native subject identities."""

from pathlib import Path
import re

import mne

from pretrain.channel_identity import sha256_file


CHANNELS = "FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz".split()
RUNS = (4, 6, 8, 10, 12, 14)


class Physionet:
    task, window_samples, channels = "physionet_mi", 256, CHANNELS

    def __init__(self, root, subjects=None):
        self.root = Path(root).resolve()
        selected = list(range(1, 110)) if subjects is None else sorted(set(subjects))
        found = {}
        for path in sorted(self.root.rglob("S???R??.edf")):
            match = re.fullmatch(r"S(\d{3})R(\d{2})\.edf", path.name)
            key = int(match[1]), int(match[2])
            if key[0] in selected and key[1] in RUNS:
                if key in found:
                    raise ValueError(f"Multiple PhysioNet source files for {path.name}")
                found[key] = path
        missing = [
            (subject, run) for subject in selected for run in RUNS if (subject, run) not in found
        ]
        if missing:
            raise ValueError(
                f"PhysioNet raw corpus is incomplete: {len(missing)} imagery EDFs missing"
            )
        self.paths = [(key, found[key]) for key in sorted(found)]
        self.label_evidence = {
            "status": "verified",
            "kind": "official_edf_annotations",
            "source": "https://physionet.org/content/eegmmidb/1.0.0/",
            "classes": {
                "0": "feet",
                "1": "left_hand",
                "2": "right_hand",
                "3": "hands",
                "4": "rest",
            },
            "subjects": selected,
            "imagery_runs": list(RUNS),
            "complete_subject_scope": selected == list(range(1, 110)),
            "crop_seconds_after_annotation": [0, 2],
            "model_crop": "true_two_seconds_at_actual_native_rate",
            "reference_catalog": {
                "native_window_samples": 320,
                "initial_event": "omit_native_sample_zero",
                "short_annotations": "retain_later_reference_events",
                "timing_correction": "legacy320 samples span2.5s at128Hz; new model crops remain2s",
                "evidence": "complete_legacy_waveform_label_and_multiplicity_replay",
            },
            "baseline_runs": "separate baseline runs are excluded; rest comes from task T0 annotations",
        }

    def recordings(self):
        for (subject, run), path in self.paths:
            with mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
                names = {name.rstrip(".").upper(): name for name in raw.ch_names}
                try:
                    picks = [names[channel.upper()] for channel in self.channels]
                except KeyError as error:
                    raise ValueError(f"PhysioNet physical channel missing: {error}") from error
                mapping = (
                    {"T0": 4, "T1": 1, "T2": 2}
                    if run in (4, 8, 12)
                    else {"T0": 4, "T1": 3, "T2": 0}
                )
                events, native_events = [], []
                short = initial = incomplete = 0
                rate = float(raw.info["sfreq"])
                for onset, duration, description in zip(
                    raw.annotations.onset,
                    raw.annotations.duration,
                    raw.annotations.description,
                    strict=True,
                ):
                    if description not in mapping:
                        continue
                    native_start = int(round((float(onset) - raw.first_time) * rate))
                    if native_start == 0:
                        initial += 1
                        continue
                    if native_start < 0 or native_start + 320 > raw.n_times:
                        incomplete += 1
                        continue
                    short += int(duration < 2)
                    native_events.append([native_start, mapping[description]])
                    events.append((int(round(native_start * 128 / rate)), mapping[description]))
                yield {
                    "volts": raw.get_data(picks=picks),
                    "sample_rate": rate,
                    "subject": f"S{subject:03d}",
                    "recording": path.stem,
                    "events": events,
                    "source": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "native_units": "volts",
                        "version": "EEGMMIDB_1.0.0",
                        "short_annotations_retained": short,
                        "initial_events_excluded": initial,
                        "incomplete_reference_crops": incomplete,
                        "legacy_native_crop_samples": 320,
                        "legacy_crop_seconds": 320 / rate,
                        "model_crop_seconds": 2,
                        "reference_native_events": native_events,
                    },
                }
