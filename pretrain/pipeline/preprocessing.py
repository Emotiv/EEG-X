"""One versioned raw-to-model signal contract for the paper task matrix."""

from importlib.metadata import version

import mne
import numpy as np

from pretrain.channel_identity import load_channel_vocabulary


def profile():
    return {
        "name": "paper_eegx_tuh_v1",
        "mne": version("mne"),
        "sample_rate": 128,
        "reference": "average",
        "notch": 60.0,
        "notch_width": 1.0,
        "notch_transition": 1.0,
        "bandpass": [0.5, 45.0],
        "filter_method": "fir",
        "filter_design": "firwin",
        "filter_window": "hamming",
        "filter_phase": "zero",
        "filter_length": "auto",
        "filter_transition": "auto",
        "padding": "reflect_limited",
        "resample_method": "fft",
        "resample_npad": "auto",
        "resample_window": "auto",
        "normalization": "recording_channel_zscore",
        "ddof": 0,
        "std_floor_volts": 1e-12,
        "clip": 15.0,
        "output_dtype": "float32",
        "ica": False,
        "rasr": False,
    }


def processed_profile():
    """Describe processed release values preserved without signal transforms."""
    return {
        "name": "processed_release_as_is_v1",
        "sample_rate": 128,
        "output_dtype": "float32",
        "local_operations": ["cast_float32"],
        "upstream_history": "not_verified",
    }


def preprocess(volts, sample_rate, channels):
    """Filter a complete recording, then standardize; input units are volts."""
    signal = np.asarray(volts, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != len(channels) or signal.shape[1] < 2:
        raise ValueError("Expected recording [physical channels, time]")
    if not np.isfinite(signal).all() or sample_rate / 2 <= 45:
        raise ValueError("Invalid signal values or acquisition sampling rate")
    ids = load_channel_vocabulary().resolve_names(channels)
    if len(set(ids.tolist())) != len(channels):
        raise ValueError("Physical electrode identities must be unique")
    settings = profile()
    raw = mne.io.RawArray(
        signal.copy(), mne.create_info(channels, sample_rate, "eeg"), verbose=False
    )
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if sample_rate / 2 > settings["notch"]:
        raw.notch_filter(
            [60.0],
            notch_widths=1.0,
            trans_bandwidth=1.0,
            method="fir",
            fir_design="firwin",
            fir_window="hamming",
            phase="zero",
            filter_length="auto",
            pad="reflect_limited",
            n_jobs=1,
            verbose=False,
        )
    raw.filter(
        0.5,
        45.0,
        method="fir",
        fir_design="firwin",
        fir_window="hamming",
        phase="zero",
        filter_length="auto",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        pad="reflect_limited",
        n_jobs=1,
        verbose=False,
    )
    if sample_rate != 128:
        raw.resample(128, method="fft", npad="auto", window="auto", n_jobs=1, verbose=False)
    values = raw.get_data()
    mean = values.mean(axis=-1, keepdims=True, dtype=np.float64)
    std = values.std(axis=-1, keepdims=True, dtype=np.float64, ddof=0)
    output = (values - mean) / np.maximum(std, settings["std_floor_volts"])
    return np.clip(output, -15, 15).astype(np.float32)
