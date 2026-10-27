"""Exercise the existing raw preparation command with the EEG-X2 signal recipe."""

import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import mne
import numpy as np
import pytest
import yaml
from sklearn.preprocessing import StandardScaler

from pretrain.pipeline.adapters.stew import Stew
from pretrain.pipeline.preprocessing import preprocess, profile
from pretrain.pipeline.dataset import read_metadata
from pretrain.pipeline.splits import validate_run_dataset

CHANNELS = Stew.channels


ROOT = Path(__file__).resolve().parents[3]


def reference_signal(values, sample_rate, channels):
    raw = mne.io.RawArray(
        values.copy(), mne.create_info(channels, sample_rate, "eeg"), verbose=False
    )
    raw.filter(0.5, 45.0, n_jobs=1, verbose=False)
    raw.resample(128, n_jobs=1, verbose=False)
    scaled = StandardScaler().fit_transform(raw.get_data().T).T
    return np.clip(scaled, -15, 15).astype(np.float32)


def test_existing_prepare_command_keeps_native_reference_and_uses_new_profile(tmp_path):
    native = tmp_path / "raw"
    native.mkdir()
    originals = {}
    for subject in range(1, 10):
        for condition, offset in (("hi", 0), ("lo", 1)):
            generator = np.random.default_rng(subject * 10 + offset)
            time = np.arange(2560) / 128
            common = np.sin(2 * np.pi * 10 * time) + 0.2 * generator.standard_normal(len(time))
            values = np.tile(common[:, None], (1, 14))
            name = f"sub{subject:02d}_{condition}"
            originals[name] = values.T * 1e-6
            np.savetxt(native / f"{name}.txt", values)
    output = tmp_path / "prepared.h5"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pretrain.pipeline.prepare",
            "--task",
            "stew",
            "--raw",
            str(native),
            "--output",
            str(output),
            "--subjects",
            *map(str, range(1, 10)),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file(), "Preparation must produce one H5 file, not a directory of NPY files"
    with h5py.File(output, "r") as handle:
        metadata = json.loads(handle["metadata"].asstr()[()])
        assert {"X", "y", "window_index", "records", "subjects", "channels", "splits"} <= set(
            handle
        )
        assert handle["X"].shape == (1314, 14, 256)
        signal = handle["X"][0]
        assert set(handle["splits/subject_holdout"]) == {"241", "242", "243", "244", "245"}
    saved = read_metadata(output)
    assert saved["reference_comparison"]["status"] == "pilot"
    with pytest.raises(ValueError, match="verified complete reference catalog"):
        validate_run_dataset({"purpose": "paper_pretrain", "seed": 241}, saved)
    config = tmp_path / "paper.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": str(output),
                "output": str(tmp_path / "paper-run"),
                "device": "cpu",
                "precision": "fp32",
                "epochs": 1,
                "depth": 1,
                "width": 16,
                "wandb": {"enabled": False},
            }
        )
    )
    rejected = subprocess.run(
        [sys.executable, "Main.py", "--config", str(config)],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "verified complete reference catalog" in rejected.stderr
    assert not (tmp_path / "paper-run/training").exists()
    assert metadata["preprocessing"] == profile()
    assert metadata["preprocessing"]["reference"] == "native"
    assert metadata["preprocessing"]["notch"] is None
    assert sum(metadata["counts"].values()) == 1314
    record = metadata["records"][0]
    expected_record = reference_signal(
        originals[Path(record["source"]["path"]).stem], 128, CHANNELS
    )
    expected = expected_record[:, :256]
    np.testing.assert_allclose(signal, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(signal[0], signal[1])
    assert signal.std() > 0.5


@pytest.mark.parametrize("amplitude", [1e-6, 1e-15])
def test_signal_recipe_matches_eegx2_standard_scaler(amplitude):
    generator = np.random.default_rng(17)
    time = np.arange(5000) / 250
    values = amplitude * np.stack([np.sin(2 * np.pi * 10 * time), np.sin(2 * np.pi * 50 * time)])
    values += generator.normal(scale=amplitude * 0.1, size=values.shape)
    channels = ["C1", "Pz"]
    np.testing.assert_allclose(
        preprocess(values, 250, channels),
        reference_signal(values, 250, channels),
        rtol=1e-6,
        atol=1e-6,
    )
