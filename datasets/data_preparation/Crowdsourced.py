import os
import pandas as pd
import argparse
import numpy as np
import pickle
from scipy import stats
import json
from dateutil import parser as pars
from filtering import emotiv_clean
from mne_ICA import MNE_ICA

def Crowdsourced(raw_data_path):
    X_datas = []
    y_datas = []
    id_datas = []
    i = 0
    # Reading only 14 Channel EEG (Epoch, EpochPlus, EpochX)
    files = [file for file in os.listdir(raw_data_path) if file.endswith('.csv.gz') and 'Alpha Supression_EPOC' in file]
    for file in files:
        print("Processing file:", file)
        data_path = os.path.join(raw_data_path, file)
        df = pd.read_csv(data_path)
        rec_details = json.load(open(data_path[0:-6] + 'json'))
        X_data, y_data = Emotiv_process(df, rec_details)
        id_data = i
        i += 1
        # Append each numpy array to the respective list
        X_datas.append(X_data)
        y_datas.append(y_data)
        id_datas.append(id_data)
    return X_datas, y_datas, id_datas

def Emotiv_process(df, rec_details):
    df.columns = [col.lower() for col in df.columns]
    COUNTER_NAME = "eeg.counter"
    assert (COUNTER_NAME in df.columns)
    channels = [c for c in df.columns if c.startswith("eeg.") and c not in ["eeg.interpolated",
                                                                            "eeg.counter",
                                                                            "eeg.rawcq",
                                                                            "eeg.battery",
                                                                            "eeg.batterypercent",
                                                                            "eeg.markerhardware"]]
    # processed_data = check_counter(df, channels)
    eeg_data = np.array(df[channels])
    t = np.array(df.timestamp)
    marker_labels = ["eyesopen_element", "eyesclose_element"]
    labels = make_label_column(t, marker_labels, rec_details)
    # Convert to NumPy array
    labels_array = labels.to_numpy()
    valid_indices = np.where((labels_array == 'eyesopen_element') | (labels_array == 'eyesclose_element'))[0]
    X_data = eeg_data[valid_indices]
    y_data = labels_array[valid_indices]
    # Step 2: Map 'eyesopen_element' to 1 and 'eyesclose_element' to 0
    y_data = np.where(y_data == 'eyesopen_element', 1, 0)
    return X_data, y_data


def check_counter(frame, eeg_channels):
    # Check and mark any missing data
    counter = np.array(frame['eeg.counter'])
    mod_number = np.max(counter) + 1
    #print(mod_number)
    if mod_number > 129:
        frame = subsample(frame, eeg_channels)
        counter = np.array(frame['eeg.counter'])

    interpolated_flag = frame['eeg.interpolated']
    interpolated_idx, = np.where(interpolated_flag > 0)
    timestamp = np.array(frame['timestamp'])
    large_time_steps, = np.where((timestamp[1:] - timestamp[0:-1]) > 1.0)
    # insert function just adds a zero to keep the length the correct length
    counter_jumps = np.insert(
        (counter[1:] - counter[0:-1]) % mod_number - 1, 0, 0)
    for i in interpolated_idx:
        counter_jumps[i] = max(1, counter_jumps[i])
    for step_idx in large_time_steps:
        counter_jumps[step_idx + 1] = 129
    frame['eeg.interpolated'] = counter_jumps
    return frame


def subsample(frame, eeg_channels):
    n_data = frame.shape[0]
    marker_channels = [ch for ch in frame.columns if 'marker' in ch]
    pm_channels = [ch for ch in frame.columns if ch.startswith('pm.')]
    pow_channels = [ch for ch in frame.columns if ch.startswith('pow.')]
    mot_channels = [ch for ch in frame.columns if ch.startswith('mot.')]
    sq_channels = [ch for ch in frame.columns if ch.startswith('sq.')]

    marker_data = np.array(frame[marker_channels])
    other_channels = [c for c in frame.columns
                      if (c not in eeg_channels and
                          c != "eeg.counter") and
                      'marker' not in c and
                      not c.startswith('pm.') and
                      not c.startswith('pow.') and
                      not c.startswith('mot.') and
                      not c.startswith('sq.')]
    other_channel_data = np.array(frame[other_channels])
    keep_idx, = np.where(frame["eeg.counter"] % 2 == 0)
    new_markers = downsample_marker_channels(marker_data, keep_idx)
    pm_data = frame[pm_channels]
    pm_data = downsample_type_data(pm_data, keep_idx, 'pm.')
    pow_data = frame[pow_channels]
    pow_data = downsample_type_data(pow_data, keep_idx, 'pow.')
    mot_data = frame[mot_channels]
    mot_data = downsample_type_data(mot_data, keep_idx, 'mot.')
    sq_data = frame[sq_channels]
    sq_data = downsample_type_data(sq_data, keep_idx, 'sq.')
    frame_temp = frame.iloc[keep_idx, :]
    other_channel_data = other_channel_data[keep_idx, :]
    # frame_temp = frame[frame[COUNTER_NAME] % 2 == 0] ## keep only even counter
    eeg_data = np.array(frame_temp[eeg_channels])
    counter = (np.array(frame_temp["eeg.counter"]) // 2).reshape(-1, 1)
    columns = ["eeg.counter"] + other_channels + eeg_channels + \
              marker_channels + pm_channels + pow_channels + \
              mot_channels + sq_channels
    data = np.concatenate([counter, other_channel_data, eeg_data, new_markers,
                           pm_data, pow_data, mot_data, sq_data], axis=1)
    frame_out = pd.DataFrame(data, columns=columns)
    return frame_out


def downsample_marker_channels(marker_data, keep_idx=None):
    """
    Need to ensure all markers are caught not just the markers that are x%2 == 0
    Note if consecutive markers are even and odd the odd marker will be
    displaced by one sample
    :param marker_data:
    :param start_idx: To match with cortex the EEG data is subsampled on the
    odd counters
    :return:
    """
    n_data = marker_data.shape[0]
    if keep_idx is None:
        keep_idx = list(range(0, n_data, 2))
    new_markers = []
    for ch in range(marker_data.shape[1]):
        new_marker = np.zeros(len(keep_idx))
        raw_marker_idxs, = np.where(marker_data[:, ch] != 0)
        marker_idxs = (raw_marker_idxs / 2).astype(int)
        last_idx = -1
        i = 0
        for idx in marker_idxs:
            if idx == last_idx:
                marker_idxs[i] += 1
                last_idx = idx + 1
            else:
                last_idx = idx
            i += 1
        new_marker[marker_idxs] = marker_data[raw_marker_idxs.tolist(), ch]
        new_markers.append(new_marker.tolist())
    return np.array(new_markers).T


def downsample_type_data(pm_data, keep_idx, prefix):
    if pm_data.shape[1] == 0:
        return np.array(pm_data)[keep_idx, :]
    ch = ""
    for c in pm_data.columns:
        if c.startswith(prefix):
            ch = c
            break
    if not ch:
        return np.zeros((len(keep_idx), 0))
    pm_idx, = np.where(np.logical_not(np.isnan(pm_data[ch])))
    odd_idx = pm_idx % 2 == 1
    new_pm_idx = np.copy(pm_idx)
    new_pm_idx[odd_idx] = new_pm_idx[odd_idx] + 1
    if new_pm_idx[-1] == pm_data.shape[0]:
        new_pm_idx[-1] -= 2
    pm_data = np.array(pm_data)
    pm_data[new_pm_idx, :] = pm_data[pm_idx, :]
    return pm_data[keep_idx, :]



def make_label_column(t, marker_labels, rec_details):
    labels = pd.Series([""] * len(t))
    for m in rec_details['markers']:
        label = m['data']['label']
        #print(m['data']['label'])
        if label in marker_labels:
            start_time = pars.parse(m['data']['startDatetime']).timestamp()
            end_time = pars.parse(m['data']['endDatetime']).timestamp()
            label_idx, = np.where(np.logical_and(t > start_time, t < end_time))
            labels[label_idx] = label
    return labels


def data_writer(args, Raw_data, reconst_raw_array, data_labels, data_id):
    w = args.window_size
    s = args.stride
    X_Raws = np.empty((0, 14, w))
    X_cleans = np.empty((0, 14, w))
    y_datas = np.empty(0)
    id_datas = np.array([], dtype=int)
    for i in range(len(Raw_data)):
        X_raw, y_data, id_data = Windowed_majority_labeling(Raw_data[i], data_labels[i], data_id[i], w, s)
        X_clean, _, _ = Windowed_majority_labeling(reconst_raw_array[i], data_labels[i], data_id[i], w, s)
        X_Raws = np.vstack((X_Raws, X_raw))
        X_cleans = np.vstack((X_cleans, X_clean))
        y_datas = np.append(y_datas, y_data)
        id_datas = np.append(id_datas, id_data)
    
    # Shuffle the data
    np.random.seed(1234)  # For reproducibility
    shuffled_indices = np.random.permutation(len(y_datas))
    X_Raws = X_Raws[shuffled_indices]
    X_cleans = X_cleans[shuffled_indices]
    y_datas = y_datas[shuffled_indices]
    id_datas = id_datas[shuffled_indices]

    # Split Subject-Wise
    # Get unique subject IDs
    unique_subjects = np.unique(id_datas)
    # Shuffle the subjects
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_subjects)
    # Calculate the number of subjects for each set
    n_total_subjects = len(unique_subjects)
    n_test_subjects = int(n_total_subjects * 0.2)
    n_val_subjects = int(n_total_subjects * 0.1)

    # Split the subjects
    test_subjects = unique_subjects[:n_test_subjects]
    val_subjects = unique_subjects[n_test_subjects:n_test_subjects + n_val_subjects]
    train_subjects = unique_subjects[n_test_subjects + n_val_subjects:]

    # Assign data points to each set based on the subject IDs
    train_idx = np.isin(id_datas, train_subjects)
    val_idx = np.isin(id_datas, val_subjects)
    test_idx = np.isin(id_datas, test_subjects)


    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # Save the data using a higher pickle protocol
    data_dict = {
        'X_train': X_Raws[train_idx], 'X_train_clean': X_cleans[train_idx], 'Y_train': y_datas[train_idx], 'id_train': id_datas[train_idx],
        'X_val': X_Raws[val_idx], 'X_val_clean': X_cleans[val_idx], 'Y_val': y_datas[val_idx], 'id_val': id_datas[val_idx],
        'X_test': X_Raws[test_idx], 'X_test_clean': X_cleans[test_idx], 'Y_test': y_datas[test_idx], 'id_test': id_datas[test_idx]
    }
    save_path = args.save_path + '/Crowdsourced.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return 


def Windowed_majority_labeling(values, labels, ids, window_size, step):
    # Initialize empty lists to store windowed samples and labels
    windowed_samples = []
    window_labels = []
    window_ids = []
    for i in range(0, values.shape[-1] - window_size + 1, step):
        # Extract the windowed sample
        windowed_sample = values[:,i:i + window_size]

         # Extract the corresponding labels for the current window
        windowed_labels = labels[i:i + window_size]
         # Perform majority voting on the labels in the window
        majority_label = stats.mode(windowed_labels, axis=None)[0]

        windowed_samples.append(list(windowed_sample))
        window_labels.append(majority_label)
        window_ids.append(ids)

    # Convert the windowed samples and labels to numpy arrays
    windowed_samples = np.array(windowed_samples) # (sample, channel, lenght)
    # windowed_samples = np.array(windowed_samples)
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)
    return windowed_samples, window_labels, window_ids


if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the Crowdsourced dataset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/EEG-X/RAW/Crowdsourced/Raw', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/Crowdsourced', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=32, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'Non'], help='Cleaning method to apply')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_datas, y_datas, id_datas = Crowdsourced(args.raw_data_path) # X_data.shape = (sample, channel)
    Filterd_data = [emotiv_clean(X_datas[i], sfreq=128).T for i in range(len(X_datas))] # emotiv_clean(lenght, channel)
    # Artifact removal
    ica = MNE_ICA(device='EPOC', sfreq=128)
    clean_data = []
    for data in Filterd_data:
        cleaned = ica.clean_data(data)
        clean_data.append(cleaned)

    data_writer(args, Filterd_data, clean_data, y_datas, id_datas)