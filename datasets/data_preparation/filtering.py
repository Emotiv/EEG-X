import numpy as np
from scipy import signal

DEFAULT_SLEW_LIMIT = 30

# Bandpass filter function (1-40 Hz)
def bandpass_filter(eeg_data, lowcut=1.0, highcut=40.0, fs=128, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, eeg_data, axis=0)

# Slew rate limiting
def limit_slew_rate(eeg_data, slew_limit=DEFAULT_SLEW_LIMIT):
    last_in = eeg_data[0, :].reshape(1, -1).copy()
    eeg_data = np.vstack((last_in, eeg_data))
    deltas = eeg_data[1:, :] - eeg_data[:-1, :]
    max_deltas = np.ones(deltas.shape) * slew_limit
    deltas = np.clip(deltas, -max_deltas, max_deltas)
    return np.cumsum(deltas, axis=0)

# High-pass filter (removes DC drift)
def ab_filter(eeg_data, filter_coeffs={}):
    a = filter_coeffs.get('a', [1., -1.96529337, 0.96588546])
    b = filter_coeffs.get('b', [0.98279471, -1.96558942, 0.98279471])
    filtfilt = filter_coeffs.get('filtfilt', False)
    return signal.filtfilt(b, a, eeg_data, axis=0) if filtfilt else signal.lfilter(b, a, eeg_data, axis=0)

# Interquartile Mean Filter (removes common noise across channels)
def iqm_filter(eeg_data):
    n_chans = eeg_data.shape[1]
    if n_chans <= 2:
        return eeg_data
    quart = n_chans // 4
    iq_idx = slice(quart, n_chans - quart)
    sorted_data = np.sort(eeg_data, axis=1)
    medians = np.mean(sorted_data[:, iq_idx], axis=1).reshape(-1, 1)
    return eeg_data - medians

# Updated Emotiv EEG Cleaning Pipeline (Correct Order)
def emotiv_clean(raw, sfreq=128):
    x = raw.copy()
    x = limit_slew_rate(x)
    x = ab_filter(x)  # Step 1: High-pass filtering (DC drift removal)
    # x = bandpass_filter(x, lowcut=1.0, highcut=40.0, fs=sfreq)  # Step 2: Bandpass filtering (1-40 Hz)
    x = iqm_filter(x)  # Step 3: Artifact removal across channels
    return x
