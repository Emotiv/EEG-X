import os
import pandas as pd
import argparse
import numpy as np
import pickle
from scipy import stats
from filtering import emotiv_clean

def Attention_EpocMN8(Data_path):
    Raw_Data = pd.read_csv(Data_path)
    Data_X_epoch, Data_X_MN8, Data_Y, subject_ids = Preprocessed_At(Raw_Data)
    return Data_X_epoch, Data_X_MN8, Data_Y, subject_ids


def Preprocessed_At(all_data):
    Distraction = all_data.marker_label__desc.unique()
    Distraction = np.setdiff1d(Distraction, 'long_baseline')

    all_data.marker_label__desc = all_data.marker_label__desc.replace(Distraction, 1)
    all_data.marker_label__desc = all_data.marker_label__desc.replace('long_baseline', 0)
    all_labels = all_data.marker_label__desc.values

    subject_index = all_data.subject_id__desc

    eeg_columns = [col for col in all_data.columns if col.endswith('__feat')]  # Remove description
    X_datas = all_data[eeg_columns].values
    X_datas_reshaped = X_datas.reshape(X_datas.shape[0], 16, 256)
    X_data_epoch = X_datas_reshaped[:, :14, :]    # Shape: (21183, 14, 256)
    X_data_MN8 = X_datas_reshaped[:, 14:, :]       # Shape: (21183, 2, 256)
    return X_data_epoch, X_data_MN8, all_labels, subject_index.values


def data_writer(args, X_data_epoch, X_data_MN8, y_datas, id_datas):
    # Shuffle the data
    np.random.seed(41)  # For reproducibility
    shuffled_indices = np.random.permutation(len(y_datas))
    X_data_epoch = X_data_epoch[shuffled_indices]
    X_data_MN8 = X_data_MN8[shuffled_indices]
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
        'X_train': X_data_epoch[train_idx], 'X_train_MN8': X_data_MN8[train_idx], 'Y_train': y_datas[train_idx], 'id_train': id_datas[train_idx],
        'X_val': X_data_epoch[val_idx], 'X_val_MN8': X_data_MN8[val_idx], 'Y_val': y_datas[val_idx], 'id_val': id_datas[val_idx],
        'X_test': X_data_epoch[test_idx], 'X_test_MN8': X_data_MN8[test_idx], 'Y_test': y_datas[test_idx], 'id_test': id_datas[test_idx]
    }
    save_path = args.save_path + '/Attention_EpocMN8.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return 


if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the Attention dataset from Epoch and MN8 Headset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/benchmark_data/A3_Attention_Epoc_MN8_Benchmark_data.csv.gz', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/Attention_EpocMN8', help='Path to save the cleaned EEG data')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_data_epoch, X_data_MN8, y_datas, id_datas = Attention_EpocMN8(args.raw_data_path) # X_data.shape = (sample, channel)

    data_writer(args, X_data_epoch, X_data_MN8, y_datas, id_datas)
