"""Native participant and annotation parsing for the added raw adapters."""

import mne
import numpy as np
import pytest

from pretrain.pipeline.adapters.physionet import CHANNELS, RUNS, Physionet
from pretrain.pipeline.adapters.stew import Stew


def test_stew_preserves_subjects_conditions_and_units(tmp_path):
    for subject in (1, 2, 3):
        for condition in ("hi", "lo"):
            np.savetxt(tmp_path / f"sub{subject:02d}_{condition}.txt", np.ones((512, 14)) * subject)
    with pytest.raises(ValueError, match="missing"):
        Stew(tmp_path)
    adapter = Stew(tmp_path, subjects=[1, 2, 3])
    rows = list(adapter.recordings())
    assert len(rows) == 6
    assert {(r["subject"], r["label"]) for r in rows} == {
        (f"{s:02d}", label) for s in (1, 2, 3) for label in (0, 1)
    }
    np.testing.assert_allclose(rows[0]["volts"], 1e-6)
    assert adapter.label_evidence["complete_subject_scope"] is False


def test_physionet_uses_imagery_annotations_native_rates_and_task_rest(tmp_path, monkeypatch):
    for subject in (1, 2):
        for run in RUNS:
            (tmp_path / f"S{subject:03d}R{run:02d}.edf").write_text(f"{subject}:{run}")

    def read(path, **kwargs):
        rate = 160 if "S001" in path.name else 128
        raw = mne.io.RawArray(
            np.zeros((64, rate * 12)),
            mne.create_info([name + "." for name in CHANNELS], rate, "eeg"),
            verbose=False,
        )
        raw.set_annotations(mne.Annotations([0, 4, 8], [3, 3, 1], ["T0", "T1", "T2"]))
        return raw

    monkeypatch.setattr(mne.io, "read_raw_edf", read)
    with pytest.raises(ValueError, match="incomplete"):
        Physionet(tmp_path)
    adapter = Physionet(tmp_path, subjects=[1, 2])
    rows = list(adapter.recordings())
    assert len(rows) == 12
    assert {row["sample_rate"] for row in rows} == {128, 160}
    assert {row["subject"] for row in rows} == {"S001", "S002"}
    for row in rows:
        run = int(row["recording"][-2:])
        assert row["events"] == [
            (512, 1 if run in (4, 8, 12) else 3),
            (1024, 2 if run in (4, 8, 12) else 0),
        ]
        assert row["source"]["short_annotations_retained"] == 1
        assert row["source"]["initial_events_excluded"] == 1
        assert row["source"]["legacy_native_crop_samples"] == 320
    assert adapter.label_evidence["classes"]["4"] == "rest"
    assert adapter.label_evidence["complete_subject_scope"] is False


def test_dreamer_native_ratings_do_not_require_a_processed_artifact(tmp_path, monkeypatch):
    from pretrain.pipeline.adapters import dreamer

    path = tmp_path / "DREAMER.mat"
    path.write_bytes(b"native fixture")
    data = {
        "EEG_Electrodes": ["C1", "Pz"],
        "EEG_SamplingRate": 128,
        "Data": [
            {"EEG": {"stimuli": [np.ones((512, 2)), np.ones((512, 2))]}, "ScoreArousal": [3, 4]}
        ],
    }
    monkeypatch.setattr(dreamer, "loadmat", lambda *args, **kwargs: {"DREAMER": data})
    adapter = dreamer.Dreamer(path)
    rows = list(adapter.recordings())
    assert [row["label"] for row in rows] == [0, 1]
    assert adapter.label_evidence["threshold"] == 3
    assert "optional_archive_audit" not in adapter.label_evidence


def test_tuev_native_annotations_do_not_require_a_processed_artifact(tmp_path, monkeypatch):
    from pretrain.pipeline.adapters import tuev

    folder = tmp_path / "train/001"
    folder.mkdir(parents=True)
    path = folder / "native.edf"
    path.write_bytes(b"native fixture")
    path.with_suffix(".rec").write_text("0,1,2,1\n0,5,6,6\n")

    def read(*args, **kwargs):
        return mne.io.RawArray(
            np.ones((19, 2000)),
            mne.create_info(
                ["EEG " + channel.upper() + "-REF" for channel in tuev.CHANNELS], 250, "eeg"
            ),
            verbose=False,
        )

    monkeypatch.setattr(mne.io, "read_raw_edf", read)
    adapter = tuev.Tuev(tmp_path)
    row = next(adapter.recordings())
    assert row["subject"] == "001"
    assert row["events"] == [(64, 1), (576, 0)]
    assert adapter.label_evidence["native_map"] == {"1": 1, "2": 1, "3": 1, "6": 0}
    assert "optional_archive_audit" not in adapter.label_evidence
