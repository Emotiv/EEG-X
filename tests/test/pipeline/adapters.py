import json
import csv
import gzip
from datetime import datetime, timezone

import mne
import numpy as np
from scipy.io import savemat

from pretrain.pipeline.adapters.bciciv import Bciciv
from pretrain.pipeline.adapters.kalunga import CHANNELS, Kalunga
from pretrain.pipeline.download import digest
from pretrain.channel_identity import sha256_file
from pretrain.pipeline.adapters.crowdsourced import Crowdsourced, CHANNELS as EPOC_CHANNELS
from pretrain.pipeline.adapters.tuev import recover_map


def test_tuev_keeps_native_subject_ids_in_both_source_partitions(tmp_path, monkeypatch):
    from pretrain.pipeline.adapters import tuev
    from pretrain.pipeline.adapters.tuab import CHANNELS as TUEV_CHANNELS

    paths = [
        tmp_path / "train/aaaaaaar/aaaaaaar_00000001.edf",
        tmp_path / "eval/000/bckg_000_a_.edf",
    ]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.touch()
        path.with_suffix(".rec").write_text("0,0,2,6\n")
    raw = mne.io.RawArray(
        np.zeros((19, 512)),
        mne.create_info(["EEG " + c.upper() + "-REF" for c in TUEV_CHANNELS], 128, "eeg"),
        verbose=False,
    )
    monkeypatch.setattr(tuev, "verify_archive", lambda *_: {"status": "verified"})
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *_, **__: raw.copy())
    recordings = list(tuev.Tuev(tmp_path, "archive.npy").recordings())
    assert {item["subject"]: item["official_split"] for item in recordings} == {
        "aaaaaaar": "train",
        "000": "test",
    }
    assert all(item["events"] == [(0, 0)] for item in recordings)


def test_tuev_rejects_unknown_ownership_before_archive_or_signal_work(tmp_path, monkeypatch):
    import pytest
    from pretrain.pipeline.adapters import tuev

    path = tmp_path / "train/unknown-owner/record.edf"
    path.parent.mkdir(parents=True)
    path.touch()

    def unexpected_work(*_):
        pytest.fail("Source ownership must be checked before archive verification")

    monkeypatch.setattr(tuev, "verify_archive", unexpected_work)
    with pytest.raises(ValueError, match="participant layout"):
        tuev.Tuev(tmp_path, "archive.npy")


def test_bci_native_units_labels_and_one_based_source_onsets(tmp_path):
    signal = np.random.default_rng(1).normal(size=(4000, 25))
    run = {"X": signal, "fs": 250, "trial": np.array([1, 1001]), "y": np.array([1, 4])}
    for subject in range(1, 10):
        for session in ("T", "E"):
            savemat(tmp_path / f"A{subject:02d}{session}.mat", {"data": run})
    adapter = Bciciv(tmp_path)
    item = next(adapter.recordings())
    assert item["subject"] == "01"
    assert item["events"] == [(256, 0), (768, 3)]
    np.testing.assert_allclose(item["volts"], signal[:, :22].T * 1e-6)


def test_kalunga_preserves_author_codes_and_source_event_offsets(tmp_path):
    signal = np.random.default_rng(1).normal(scale=1e-6, size=(9, 8192))
    signal[-1] = 0
    for start, code in zip([256, 2304, 4352, 6400], [1, 2, 3, 4], strict=True):
        signal[-1, start] = code
    raw = mne.io.RawArray(
        signal, mne.create_info(CHANNELS + ["STI 014"], 256, ["eeg"] * 8 + ["stim"]), verbose=False
    )
    path = tmp_path / "subject01_run1_raw.fif"
    raw.save(path, overwrite=False, verbose=False)
    (tmp_path / "record.json").write_text(
        json.dumps(
            {
                "id": 2392979,
                "files": [
                    {"key": path.name, "size": path.stat().st_size, "checksum": digest(path)}
                ],
            }
        )
    )
    adapter = Kalunga(tmp_path)
    item = next(adapter.recordings())
    assert item["subject"] == "01"
    assert item["events"] == [(384, 0), (1408, 1), (2432, 2), (3456, 3)]
    assert adapter.label_evidence["classes"] == {
        "0": "rest",
        "1": "13_Hz",
        "2": "21_Hz",
        "3": "17_Hz",
    }
    np.testing.assert_allclose(item["volts"], signal[:8], rtol=1e-6)


def test_tuev_map_is_derived_from_every_archived_label():
    import pytest

    native = np.array([6, 1, 2, 3, 6, 1, 2, 3])
    saved = np.array([0, 1, 1, 1, 0, 1, 1, 1])
    assert recover_map(native, saved) == {1: 1, 2: 1, 3: 1, 6: 0}
    saved[-1] = 0
    with pytest.raises(ValueError, match="conflicting archived"):
        recover_map(native, saved)


def test_crowdsourced_gaps_keep_subject_identity_and_never_join_signal(tmp_path):
    rate, count = 128, 5120
    clock = 1_630_000_000 + np.arange(count) / rate
    clock[2000:] += 2.5
    eeg = np.random.default_rng(12).normal(4000, 20, (count, 14))
    values = np.column_stack((clock, np.arange(count) % 129, np.zeros(count), eeg))
    stem = "Alpha Supression_EPOC_2020_11_27T22_20"
    signal = tmp_path / (stem + ".csv.gz")
    with gzip.open(signal, "wt") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Timestamp", "EEG.Counter", "EEG.Interpolated"] + ["EEG." + c for c in EPOC_CHANNELS]
        )
        writer.writerows(values)
    markers = []
    for begin, end, label in [(2, 12, "eyesopen_element"), (20, 32, "eyesclose_element")]:
        markers.append(
            {
                "data": {
                    "label": label,
                    "startDatetime": datetime.fromtimestamp(
                        clock[0] + begin, timezone.utc
                    ).isoformat(),
                    "endDatetime": datetime.fromtimestamp(clock[0] + end, timezone.utc).isoformat(),
                }
            }
        )
    (tmp_path / (stem + ".json")).write_text(json.dumps({"markers": markers}))
    (tmp_path / (stem + "_markers.csv")).write_text("timestamp,label\n")
    (tmp_path / "Readme.txt").write_text("Three files per recording (per subject)")
    files = [
        {"attributes": {"name": path.name, "extra": {"hashes": {"sha256": sha256_file(path)}}}}
        for path in tmp_path.iterdir()
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"record": "9bvgh", "files": files}))
    adapter = Crowdsourced(tmp_path)
    recordings = list(adapter.recordings())
    assert len(recordings) == 2
    assert {record["subject"] for record in recordings} == {stem}
    assert sum(record["volts"].shape[1] for record in recordings) == count
    assert {label for record in recordings for _, label in record["events"]} == {0, 1}
    for record in recordings:
        first, last = (
            record["source"]["first_native_sample"],
            record["source"]["last_native_sample"],
        )
        np.testing.assert_allclose(record["volts"], eeg[first:last].T * 1e-6)
    assert adapter.label_evidence["record_audit"][0]["counter_modulus"] == 129
