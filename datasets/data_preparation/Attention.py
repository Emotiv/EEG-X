import os
import pandas as pd
import argparse
import numpy as np
import pickle
from scipy import stats
from filtering import emotiv_clean
from mne_ICA import MNE_ICA

def Attention(Data_path):
    Raw_Data = pd.read_csv(Data_path)
    Data_X, Data_Y, subject_ids = Preprocessed_At(Raw_Data)
    return Data_X, Data_Y, subject_ids


def find_col(data):
    eeg_columns = ['eeg.af3', 'eeg.f7', 'eeg.f3', 'eeg.fc5', 'eeg.t7', 'eeg.p7',
                   'eeg.o1', 'eeg.o2', 'eeg.p8', 'eeg.t8', 'eeg.fc6', 'eeg.f4',
                   'eeg.f8', 'eeg.af4']
    return data[eeg_columns]

def Preprocessed_At(all_data):
    cq_boolean = np.array(all_data['minimum_cq__desc'] > 3)
    all_data = all_data[cq_boolean]

    Distraction = all_data.marker_label__desc.unique()
    Distraction = np.setdiff1d(Distraction, 'long_baseline')

    all_data.marker_label__desc = all_data.marker_label__desc.replace(Distraction, 1)
    all_data.marker_label__desc = all_data.marker_label__desc.replace('long_baseline', 0)
    all_labels = all_data.marker_label__desc.values

    subject_index = all_data.subject_id__desc

    X_datas = find_col(all_data)  # Remove description
    return X_datas, all_labels, subject_index.values


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
    np.random.seed(4321)  # For reproducibility
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
    save_path = args.save_path + '/Attention.npy'
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
        window_ids.append(ids[i])

    # Convert the windowed samples and labels to numpy arrays
    windowed_samples = np.array(windowed_samples) # (sample, channel, lenght)
    # windowed_samples = np.array(windowed_samples)
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)
    return windowed_samples, window_labels, window_ids


if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the Attention dataset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/EEG-X/RAW/Attention/A3_Attention_Benchmark_data_raw.csv.gz', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/Attention', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=64, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'Non'], help='Cleaning method to apply')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_datas, y_datas, id_datas = Attention(args.raw_data_path) # X_data.shape = (sample, channel)
    Filterd_data = []
    Filterd_labels = []
    Filterd_ids = []
    unique_subjects = np.unique(id_datas)  # Get the unique subject IDs
    for subject in unique_subjects:
        subject_mask = id_datas == subject
        subject_values = X_datas[subject_mask].values
        subject_labels = y_datas[subject_mask]
        ids = id_datas[subject_mask]
        Filterd_data.append(emotiv_clean(subject_values).T)  # emotiv_clean(input_shape : length, channel)
        Filterd_labels.append(subject_labels)
        Filterd_ids.append(ids)

    # Artifact removal
    ica = MNE_ICA(device='EPOC', sfreq=128)
    clean_data = []
    for data in Filterd_data:
        cleaned = ica.clean_data(data)
        clean_data.append(cleaned)
    data_writer(args, Filterd_data, clean_data, Filterd_labels, Filterd_ids)
