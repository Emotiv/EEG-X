"""
Author: Navid Foumani

Dataset: DREAMER - A Database for Emotion Recognition through EEG
Download Link: https://zenodo.org/records/546113

Default raw data path: "/data/EEG-X/RAW/DREAMER/Raw"  
Cleaned data path: "/data/EEG-X/Processed/DREAMER/"

Description:  
This script loads the raw EEG data, applies the Emotiv Filtering, and then applies either ICA, rASR, or none based on the specified arguments.  
The cleaned data is segmented into windows of a predefined size (default: 256 samples with 75% overlap).  
The dataset is then split into training and test sets using subject-wise partitioning, ensuring that no subject appears in both sets.  
(20% of the subjects are randomly selected for the test set.)

Inputs:  
- `raw_data_path`: Path to the raw EEG data (default: "/data/EEG-X/RAW/DREAMER/Raw")  
- `save_data_path`: Path to save the cleaned EEG data (default: "/data/EEG-X/Processed/DREAMER/")  
- `window_size`: Size of the data windows for segmentation (default: 256)  
- `overlap_ratio`: Overlap between consecutive windows (default: 75%)  
- `apply_ica`: Boolean flag to apply ICA (default: False)  
- `apply_rasr`: Boolean flag to apply rASR (default: False)  

Output:  
- Processed EEG data saved in the specified `save_data_path`, ready for downstream tasks.  
"""


import os
import pandas as pd
import argparse
import numpy as np
# import mne
import pickle
import scipy.io as sio
from filtering import emotiv_clean
# from mne_ICA import MNE_ICA

def DREAMER(data_path):
    dat = sio.loadmat(data_path + '/DREAMER.mat')
    dat = dat['DREAMER']
    X_datas = []
    Y = np.empty((414, 3))
    trial = np.empty((414), dtype=np.int8)
    subject = np.empty((414,), dtype=np.int8)
    age = np.empty((414,), dtype=np.int8)
    for participant_id in range(0, 23):
        for video_id in range(0, 18):
            experiment_id = participant_id * 18 + video_id
            X_temp = np.transpose(dat[0, 0]['Data'][0, participant_id]['EEG'][0, 0]['stimuli'][0, 0][video_id, 0][:, :])
            X_datas.append(X_temp)
            Y[experiment_id, 0] = dat[0, 0]['Data'][0, participant_id]['ScoreArousal'][0, 0][video_id, 0]
            Y[experiment_id, 1] = dat[0, 0]['Data'][0, participant_id]['ScoreValence'][0, 0][video_id, 0]
            Y[experiment_id, 2] = dat[0, 0]['Data'][0, participant_id]['ScoreDominance'][0, 0][video_id, 0]
            subject[experiment_id] = participant_id + 1
            trial[experiment_id] = video_id + 1
            age[experiment_id] = dat[0, 0]['Data'][0, participant_id]['Age'][0, 0]

    y_datas = (Y).astype(int)
    # y_datas = (Y[:, 1] > 3).astype(int)
    # y_datas_d = (Y[:, 2] > 3).astype(int)
    
    return X_datas, y_datas, subject, age


def Windowed_majority_labeling(values, labels, ids, window_size, step):
    # Initialize empty lists to store windowed samples and labels
    windowed_samples = []
    window_labels_a = []
    window_labels_v = []
    window_labels_d = []
    window_ids = []
    for i in range(0, values.shape[-1] - window_size + 1, step):
        # Extract the windowed sample
        windowed_sample = values[:,i:i + window_size]
        # Append the windowed sample and label to the lists
        windowed_samples.append(list(windowed_sample))
        window_labels_a.append(labels[0])
        window_labels_v.append(labels[1])
        window_labels_d.append(labels[2])
        window_ids.append(ids)

    # Convert the windowed samples and labels to numpy arrays
    windowed_samples = np.array(windowed_samples) # (sample, channel, lenght)
    # windowed_samples = np.array(windowed_samples)
    window_labels_a = np.array(window_labels_a)
    window_labels_v = np.array(window_labels_v)
    window_labels_d = np.array(window_labels_d)
    window_ids = np.array(window_ids)
    return windowed_samples, window_labels_a, window_labels_v, window_labels_d, window_ids  


def data_writer(args, Raw_data, data_labels, data_id, age):
    w = args.window_size
    s = args.stride
    X_Raws = np.empty((0, 14, w))
    # X_cleans = np.empty((0, 14, w))
    y_datas_a = np.empty(0)
    y_datas_v = np.empty(0)
    y_datas_d = np.empty(0)
    age_datas = np.empty(0)
    id_datas = np.array([], dtype=int)
    for i in range(len(Raw_data)):
        X_raw, y_data_a, y_data_v, y_data_d, id_data = Windowed_majority_labeling(Raw_data[i], data_labels[i], data_id[i], w, s)
        # X_clean, _, _, _, _ = Windowed_majority_labeling(reconst_raw_array[i], data_labels, data_id[i], w, s)
        X_Raws = np.vstack((X_Raws, X_raw))
        # X_cleans = np.vstack((X_cleans, X_clean))
        y_datas_a = np.hstack((y_datas_a, y_data_a)) 
        y_datas_v = np.hstack((y_datas_v, y_data_v))
        y_datas_d = np.hstack((y_datas_d, y_data_d))
        id_datas = np.hstack((id_datas, id_data))
        # age_datas = np.hstack((age_datas, age[i]))
    # Shuffle the data
    np.random.seed(1234)  # For reproducibility
    shuffled_indices = np.random.permutation(len(X_Raws))
    X_Raws = X_Raws[shuffled_indices]
    # X_cleans = X_cleans[shuffled_indices]
    y_datas_a = y_datas_a[shuffled_indices]
    y_datas_v = y_datas_v[shuffled_indices]
    y_datas_d = y_datas_d[shuffled_indices]
    # age_datas = age_datas[shuffled_indices]
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
    data_dict_a = {
        'X_train': X_Raws[train_idx], 'Y_train': y_datas_a[train_idx], 'id_train': id_datas[train_idx],
        'X_val': X_Raws[val_idx], 'Y_val': y_datas_a[val_idx], 'id_val': id_datas[val_idx],
        'X_test': X_Raws[test_idx], 'Y_test': y_datas_a[test_idx], 'id_test': id_datas[test_idx]
    }
    if not os.path.exists(args.save_path + '_Arousal'):
        os.makedirs(args.save_path + '_Arousal')
    save_path = args.save_path + '_Arousal/DREAMER_Arousal.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict_a, f)
    data_dict_v = {
        'X_train': X_Raws[train_idx], 'Y_train': y_datas_v[train_idx], 'id_train': id_datas[train_idx],
        'X_val': X_Raws[val_idx], 'Y_val': y_datas_v[val_idx], 'id_val': id_datas[val_idx],
        'X_test': X_Raws[test_idx], 'Y_test': y_datas_v[test_idx], 'id_test': id_datas[test_idx]
    }
    if not os.path.exists(args.save_path + '_Valence'):
        os.makedirs(args.save_path + '_Valence')
    save_path = args.save_path + '_Valence/DREAMER_Valence.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict_v, f)
    data_dict_d = { 
        'X_train': X_Raws[train_idx], 'Y_train': y_datas_d[train_idx], 'id_train': id_datas[train_idx],
        'X_val': X_Raws[val_idx], 'Y_val': y_datas_d[val_idx], 'id_val': id_datas[val_idx],
        'X_test': X_Raws[test_idx], 'Y_test': y_datas_d[test_idx], 'id_test': id_datas[test_idx]
    }
    if not os.path.exists(args.save_path + '_Dominance'):
        os.makedirs(args.save_path + '_Dominance')
    save_path = args.save_path + '_Dominance/DREAMER_Dominance.npy'
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict_d, f)

    return 

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the DREAMER dataset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/EEG-X/RAW/DREAMER/Raw', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/DREAMER', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=64, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'Non'], help='Cleaning method to apply')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_datas, y_datas, id_datas, age = DREAMER(args.raw_data_path) # X_data.shape =   (sample, channel, lenght)
    Filterd_data = [emotiv_clean(X_datas[i].T, sfreq=128).T for i in range(len(X_datas))]
    # Artifact removal
    # ica = MNE_ICA(device='EPOC', sfreq=128)
    # clean_data = []
    # for data in Filterd_data:
    #     cleaned = ica.clean_data(data)
    #     clean_data.append(cleaned)
    # clean_data = np.array(clean_data)
    # data_writer(args, Filterd_data, clean_data, y_datas, id_datas, age) 
    data_writer(args, Filterd_data, y_datas, id_datas, age) 
