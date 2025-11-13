"""
Author: Navid Foumani

Dataset: STEW - Simultaneous Task EEG Workload Dataset  
Download Link: https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset

Default raw data path: "/data/EEG-X/RAW/STEW/Raw"  
Cleaned data path: "/data/EEG-X/Processed/STEW/"

Description:  
This script loads the raw EEG data, applies the Emotiv Filtering, and then applies either ICA, rASR, or none based on the specified arguments.  
The cleaned data is segmented into windows of a predefined size (default: 256 samples with 75% overlap).  
The dataset is then split into training and test sets using subject-wise partitioning, ensuring that no subject appears in both sets.  
(20% of the subjects are randomly selected for the test set.)

Inputs:  
- `raw_data_path`: Path to the raw EEG data (default: "/data/EEG-X/RAW/STEW/Raw")  
- `save_data_path`: Path to save the cleaned EEG data (default: "/data/EEG-X/Processed/STEW/")  
- `window_size`: Size of the data windows for segmentation (default: 256)  
- `overlap_ratio`: Overlap between consecutive windows (default: 32 (82.5%))  
- `apply_ica`: Boolean flag to apply ICA (default: False)  
- `apply_rasr`: Boolean flag to apply rASR (default: False)  

Output:  
- Processed EEG data saved in the specified `save_data_path`, ready for downstream tasks.  
"""


import os
import pandas as pd
import argparse
import numpy as np
import mne
import pickle
from filtering import emotiv_clean
from mne_ICA import MNE_ICA

def STEW(data_path):
    # Initialize lists to hold the data, labels, and IDs
    X_datas = []
    y_datas = []
    id_datas = []
    # Loop through the files in the directory
    for file_name in os.listdir(data_path):
        if file_name.startswith('sub') and file_name.endswith('.txt'):
            # Determine the label based on the file name
            label = 1 if 'hi' in file_name else 0
            # Extract the subject ID from the file name
            subject_id = int(file_name[3:5])

            # Construct the full path to the file
            file_path = os.path.join(data_path, file_name)

            # Read the data file into a dataframe
            values = pd.read_csv(file_path, delimiter=',', header=None).values

            # Append the data, label, and ID to the respective lists
            X_datas.append(values.T)
            y_datas.append(label)
            id_datas.append(subject_id)

    # Verify the lengths of the lists
    print(f"Number of Recording: {len(X_datas)}")  # Expected length: 96
    print(f"Number of Subjects: {len(list(set(id_datas)))}")  # Expected length: 48
    return X_datas, y_datas, id_datas


def Windowed_majority_labeling(values, labels, ids, window_size, step):
    # Initialize empty lists to store windowed samples and labels
    windowed_samples = []
    window_labels = []
    window_ids = []
    for i in range(0, values.shape[-1] - window_size + 1, step):
        # Extract the windowed sample
        windowed_sample = values[:,i:i + window_size]
        # Append the windowed sample and label to the lists
        windowed_samples.append(list(windowed_sample))
        window_labels.append(labels)
        window_ids.append(ids)

    # Convert the windowed samples and labels to numpy arrays
    windowed_samples = np.array(windowed_samples) # (sample, channel, lenght)
    # windowed_samples = np.array(windowed_samples)
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)
    return windowed_samples, window_labels, window_ids


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
    np.random.seed(1234)  # For reproducibility
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
    save_path = args.save_path + '/STEW.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return 


if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the STEW dataset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/EEG-X/RAW/STEW/Raw', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/STEW', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=32, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'Non'], help='Cleaning method to apply')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_datas, y_datas, id_datas = STEW(args.raw_data_path) # X_data.shape =   (sample, channel, lenght)
    Filterd_data = [emotiv_clean(X_datas[i].T, sfreq=128).T for i in range(len(X_datas))] # emotiv_clean(lenght, channel)
    # Artifact removal
    ica = MNE_ICA(device='EPOC', sfreq=128)
    clean_data = []
    for data in Filterd_data:
        cleaned = ica.clean_data(data)
        clean_data.append(cleaned)
    clean_data = np.array(clean_data)
    data_writer(args, Filterd_data, clean_data, y_datas, id_datas)
