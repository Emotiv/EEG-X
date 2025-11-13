"""
Author: Navid Foumani

Dataset: BCICIV_2A - Simultaneous Task EEG Workload Dataset  
Download Link: https://ieee-dataport.org/open-access/BCICIV_2A-simultaneous-task-eeg-workload-dataset

Default raw data path: "/data/EEG-X/RAW/BCICIV_2A/Raw"  
Cleaned data path: "/data/EEG-X/Processed/BCICIV_2A/"

Description:  
This script loads the raw EEG data, applies the Emotiv Filtering, and then applies either ICA, rASR, or none based on the specified arguments.  
The cleaned data is segmented into windows of a predefined size (default: 256 samples with 75% overlap).  
The dataset is then split into training and test sets using subject-wise partitioning, ensuring that no subject appears in both sets.  
(20% of the subjects are randomly selected for the test set.)

Inputs:  
- `raw_data_path`: Path to the raw EEG data (default: "/data/EEG-X/RAW/BCICIV_2A/Raw")  
- `save_data_path`: Path to save the cleaned EEG data (default: "/data/EEG-X/Processed/BCICIV_2A/")  
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

def BCICIV_2A(data_path):
    '''
    data_path = 'datasets/ICA_BCICIV_2a/BCICIV_2a_raw'

NO_channels = 22
No_Trial = 6*48*2*9 	
Window_Length = 6*250 
offset = 2*250 
class_return = np.zeros(No_Trial)
subject_return = np.zeros(No_Trial, dtype=object)
data_return = np.zeros((No_Trial,NO_channels,Window_Length-offset))



NO_valid_trial = 0
for file_name in os.listdir(data_path):
    if file_name.endswith('.mat'):
        raw_data = sio.loadmat(os.path.join(data_path, file_name))
        raw_data = raw_data['data']
        for i in range(0,raw_data.size): # for loop in the number of trials for each subject
            data = raw_data[0, i][0, 0]
            X        = data[0][:,:NO_channels]
            a_trial    = data[1]
            y        = data[2]
            classes  = data[4]
            # artifacts = data[5]
            # gender   = data[6]
            # age      = data[7]
            # fs       = data[3]
            for trial in range(0,a_trial.size):
                data_return[NO_valid_trial,:,:] = np.transpose(X[int(a_trial[trial]+offset):(int(a_trial[trial])+Window_Length)])
                class_return[NO_valid_trial] = int(y[trial])
                subject_return[NO_valid_trial] = file_name[1:4]
                NO_valid_trial +=1

data_dict = {
    'X': data_return,
    'Y': class_return,
    'id': subject_return
}

np.save(os.path.join(data_path, 'BCICIV_2a_raw_4s_defult.npy'), data_dict)

import numpy as np
data = np.load('other/BCICIV_2a_raw_4s.npy', allow_pickle=True).item()
X_Raws = data['X']
from filtering import emotiv_clean
Filterd_data = [emotiv_clean(X_Raws[i].T).T for i in range(len(X_Raws))]
    
    '''

    # Verify the lengths of the lists
    print(f"Number of Recording: {len(X_datas)}")  # Expected length: 96
    print(f"Number of Subjects: {len(list(set(id_datas)))}")  # Expected length: 48
    return X_datas, y_datas, id_datas

def create_slices(data, window_size=256, stride=32):
    """
    Create slices from the data with specified dimensions (1, 14, 256)
    
    Args:
        data: Input data array
        window_size: Size of each window (default: 256)
        stride: Stride between windows (default: 32)
    
    Returns:
        numpy array of shape (N, 1, 14, 256) containing all slices
    """
    slices = []
    for i in range(0, data.shape[-1] - window_size + 1, stride):
        slice_data = data[:, i:i + window_size]
        # Reshape to (1, 14, 256)
        slice_data = slice_data.reshape(1, 14, window_size)
        slices.append(slice_data)
    
    # Stack all slices into a single array
    return np.stack(slices)

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the BCICIV_2A dataset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/EEG-X/RAW/BCICIV_2A/Raw', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/BCICIV_2A', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=32, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'Non'], help='Cleaning method to apply')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_datas, y_datas, id_datas = BCICIV_2A(args.raw_data_path) # X_data.shape = (sample, channel, length)
    
    # Create slices from the data
    all_slices = []
    for data in X_datas:
        slices = create_slices(data, args.window_size, args.stride)
        all_slices.append(slices)
    
    # Stack all slices from all recordings
    all_slices = np.concatenate(all_slices, axis=0)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Save the slices
    save_path = os.path.join(args.save_path, 'BCICIV_2A_slices.npy')
    np.save(save_path, all_slices)
    print(f"Saved {len(all_slices)} slices to {save_path}")
    print(f"Shape of saved slices: {all_slices.shape}")
    
    # Continue with the rest of the processing if needed
    Filterd_data = [emotiv_clean(X_datas[i].T, sfreq=128).T for i in range(len(X_datas))]
    ica = MNE_ICA(device='EPOC', sfreq=128)
    clean_data = []
    for data in Filterd_data:
        cleaned = ica.clean_data(data)
        clean_data.append(cleaned)
    # clean_data = np.array(clean_data)
    # data_writer(args, Filterd_data, clean_data, y_datas, id_datas)
