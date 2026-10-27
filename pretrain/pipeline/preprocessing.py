"""EEG-X2 QuantPlus signal preparation shared by the native dataset adapters."""

from importlib.metadata import version

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

from pretrain.channel_identity import load_channel_vocabulary


def profile():
    return {
        "name": "eegx2_quantplus_v1",
        "mne": version("mne"),
        "scikit_learn": version("scikit-learn"),
        "sample_rate": 128,
        "reference": "native",
        "notch": None,
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
        "normalization": "recording_channel_standard_scaler",
        "scaler_with_mean": True,
        "scaler_with_std": True,
        "ddof": 0,
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
    """Preserve native reference, filter and standardize a recording given in volts."""
    settings = profile()
    highpass, lowpass = settings["bandpass"]
    signal = np.asarray(volts, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != len(channels) or signal.shape[1] < 2:
        raise ValueError("Expected recording [physical channels, time]")
    if not np.isfinite(signal).all() or sample_rate / 2 <= lowpass:
        raise ValueError("Invalid signal values or acquisition sampling rate")
    ids = load_channel_vocabulary().resolve_names(channels)
    if len(set(ids.tolist())) != len(channels):
        raise ValueError("Physical electrode identities must be unique")
    raw = mne.io.RawArray(
        signal.copy(), mne.create_info(channels, sample_rate, "eeg"), verbose=False
    )
    raw.filter(
        highpass,
        lowpass,
        method=settings["filter_method"],
        fir_design=settings["filter_design"],
        fir_window=settings["filter_window"],
        phase=settings["filter_phase"],
        filter_length=settings["filter_length"],
        l_trans_bandwidth=settings["filter_transition"],
        h_trans_bandwidth=settings["filter_transition"],
        pad=settings["padding"],
        n_jobs=1,
        verbose=False,
    )
    if sample_rate != settings["sample_rate"]:
        raw.resample(
            settings["sample_rate"],
            method=settings["resample_method"],
            npad=settings["resample_npad"],
            window=settings["resample_window"],
            n_jobs=1,
            verbose=False,
        )
    output = (
        StandardScaler(with_mean=settings["scaler_with_mean"], with_std=settings["scaler_with_std"])
        .fit_transform(raw.get_data().T)
        .T
    )
    return np.clip(output, -settings["clip"], settings["clip"]).astype(settings["output_dtype"])
