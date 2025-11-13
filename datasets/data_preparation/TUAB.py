'''
Author: Navid

This script prepares the TUAB dataset for analysis. 
TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
The raw data is stored in /data/datasets_public/TUAB/edf.

The code reads the data, applies Emotiv filtering, segments it into 2-second windows, 
and stores the results in /data/datasets_public/TUAB/processed_128hz_256seqlen.
This preprocessing makes the TUAB dataset ready for downstream tasks and model compatibility.

'''
import os
import logging
import numpy as np
import mne
import scipy.io as sio
import pickle
from tqdm import trange
from scipy.signal import butter, sosfilt
from sklearn import model_selection
from filtering import emotiv_clean
from mne.io import RawArray
from mne_icalabel import label_components


logger = logging.getLogger(__name__)



def TUAB(raw_data_path, save_path, window_size, stride):

    generate_data(raw_data_path, save_path)

    return


def generate_data(root, save_path):
    
    # Train abnormal subjects
    train_abnormal_path = os.path.join(root, "train", "abnormal/01_tcp_ar")
    train_abnormal_list = list(set([item.split("_")[0] for item in os.listdir(train_abnormal_path)]))
    # Split the list into training and validation sets (10% validation)
    train_abnormal_list, val_abnormal_list = model_selection.train_test_split(train_abnormal_list, test_size=0.1, random_state=42)
    print("Train abnormal # of subject:" + str(len(train_abnormal_list)))
    print("Validation abnormal # of subject:" + str(len(val_abnormal_list)))

    # Train normal subjects
    train_normal_path = os.path.join(root, "train", "normal/01_tcp_ar")
    train_normal_list = list(set([item.split("_")[0] for item in os.listdir(train_normal_path)]))
    # Split the list into training and validation sets (10% validation)
    train_normal_list, val_normal_list = model_selection.train_test_split(train_normal_list, test_size=0.1, random_state=42)
    print("Train normal # of subject:" + str(len(train_normal_list)))
    print("Validation normal # of subject:" + str(len(val_normal_list)))



    # Test abnormal subjects
    test_abnormal_path = os.path.join(root, "eval", "abnormal/01_tcp_ar")
    test_abnormal_list = list(set([item.split("_")[0] for item in os.listdir(test_abnormal_path)]))
    print("Test abnormal # of subject:" + str(len(test_abnormal_list)))
    # Test normal subjects
    test_normal_path = os.path.join(root, "eval", "normal/01_tcp_ar")
    test_normal_list = list(set([item.split("_")[0] for item in os.listdir(test_normal_path)]))
    print("Test normal # of subject:" + str(len(test_normal_list)))

    # ---------------------------------------------------------------------------------------------------------
    # create the raw train, val, test sample folder
    if not os.path.exists(os.path.join(save_path, "processed_Raw")):
        os.makedirs(os.path.join(save_path, "processed_Raw"))

        if not os.path.exists(os.path.join(save_path, "processed_Raw", "val")):
            os.makedirs(os.path.join(save_path, "processed_Raw", "val"))
        val_dump_folder = os.path.join(save_path, "processed_Raw", "val")

        pre_processing(train_abnormal_path, val_abnormal_list, val_dump_folder, 1)
        pre_processing(train_normal_path,  val_normal_list, val_dump_folder, 0)

        
        # --------------------------------------------------------------------------------------------------------------
        if not os.path.exists(os.path.join(save_path, "processed_Raw", "train")):
            os.makedirs(os.path.join(save_path, "processed_Raw", "train"))
        train_dump_folder = os.path.join(save_path, "processed_Raw", "train")

        pre_processing(train_abnormal_path, train_abnormal_list, train_dump_folder, 1)
        pre_processing(train_normal_path,  train_normal_list, train_dump_folder, 0)


        # --------------------------------------------------------------------------------------------------------------
        if not os.path.exists(os.path.join(save_path, "processed_Raw", "test")):
            os.makedirs(os.path.join(save_path, "processed_Raw", "test"))
        test_dump_folder = os.path.join(save_path, "processed_Raw", "test")

        pre_processing(test_abnormal_path, test_abnormal_list, test_dump_folder, 1)
        pre_processing(test_normal_path, test_normal_list, test_dump_folder, 0)
    
    # --------------------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(save_path, "windowed_10S")):
        os.makedirs(os.path.join(save_path, "windowed_10S"))

        if not os.path.exists(os.path.join(save_path, "windowed_10S", "val")):
            os.makedirs(os.path.join(save_path, "windowed_10S", "val"))
        val_chunk_folder = os.path.join(save_path, "windowed_10S", "val")
        segment_save(os.path.join(save_path, "processed_Raw", "val"), val_chunk_folder)
        print ("Validation samples were done")

        if not os.path.exists(os.path.join(save_path, "windowed_10S", "train")):
            os.makedirs(os.path.join(save_path, "windowed_10S", "train"))
        train_chunk_folder = os.path.join(save_path, "windowed_10S", "train")
        segment_save(os.path.join(save_path, "processed_Raw", "train"), train_chunk_folder)
        print ("Training samples were done")


        if not os.path.exists(os.path.join(save_path, "windowed_10S", "test")):
            os.makedirs(os.path.join(save_path, "windowed_10S", "test"))
        test_chunk_folder = os.path.join(save_path, "windowed_10S", "test")
        segment_save(os.path.join(save_path, "processed_Raw", "test"), test_chunk_folder)
        print ("Testing samples were done")

    return


def pre_processing(data_path, data_list, save_path, label):
    # Initialize lists to store data, labels, and IDs
    for file in os.listdir(data_path):
        if file.split("_")[0] in data_list:
            print("process", file)
            # loading *.edf file
            file_path = os.path.join(data_path, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            # downsample from 250 Hz to 128 Hz
            raw.resample(128)
            ch_name = raw.ch_names
            raw_data = raw.get_data()
            channeled_data = raw_data.copy()[:19]
            # LE_data = Linked_Ears_Reference(channeled_data, raw_data, ch_name)
            # Match_data = Channel_selection (channeled_data, raw_data, ch_name)
            Match_data = Channel_Order(channeled_data, raw_data, ch_name)
            # Emotiv Filtering
            Match_data_filtered = emotiv_clean(Match_data) 

            # ICA Cleaning ---------------------------------------------------------------
            Match_data_clean =  ICA_Clean(Match_data_filtered)           
            # Save Match Data
            # Save processed data for the current file
            file_id = os.path.splitext(file)[0]  # Remove file extension
            save_file_path = os.path.join(save_path, f"{file_id}.pkl")
            # Create the data dictionary
            data_dict = {'X': Match_data_filtered, 'X_Clean':Match_data_clean ,'y': label, 'id': file_id.split("_")[0]}
            with open(save_file_path, 'wb') as f:
                pickle.dump(data_dict, f)
    return


def ICA_Clean(data):

    analyze_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                    'O2', 'F7', 'F8', 'T7', 'T8', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    # Load the standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    # Keep only the specified channels
    channel_positions = {ch: pos for ch, pos in montage.get_positions()['ch_pos'].items() if ch in analyze_channels}
    mont = mne.channels.make_dig_montage(ch_pos=channel_positions, coord_frame='head')
    Epoch_info = mne.create_info(ch_names=mont.ch_names, sfreq=128.,ch_types='eeg')

    ica = mne.preprocessing.ICA(
    n_components=18,
    max_iter="auto",
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),)   
    Filterd_mne = RawArray(data.copy(), info=Epoch_info, verbose=0)
    Filterd_mne = Filterd_mne.set_montage(mont)
    Filterd_mne = Filterd_mne.set_eeg_reference("average", verbose=0)

    ica.fit(Filterd_mne)
    ic_labels = label_components(Filterd_mne, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    reconst = Filterd_mne.copy()
    clean_data = ica.apply(reconst, exclude=exclude_idx)
    clean_data = clean_data.get_data()
    return clean_data


def Channel_Order(channeled_data, raw_data, ch_name):

    '''
    Temple Cap:
                            Fp1   |Fpz|   Fp2
                F7    F3          |Fz|        F4    F8
    A1 ---  T3    C3              |Cz|          C4      T4 --- A2
                T5    P3          |Pz|         P4   T6
                          O1      |Oz|     O2

    '''
    channeled_data[0] = raw_data[ch_name.index("EEG FP1-REF")]
    channeled_data[1] = raw_data[ch_name.index("EEG FP2-REF")] 
    channeled_data[2] = raw_data[ch_name.index("EEG F3-REF")]
    channeled_data[3] = raw_data[ch_name.index("EEG F4-REF")]
    channeled_data[4] = raw_data[ch_name.index("EEG C3-REF")]
    channeled_data[5] = raw_data[ch_name.index("EEG C4-REF")]
    channeled_data[6] = raw_data[ch_name.index("EEG P3-REF")]
    channeled_data[7] = raw_data[ch_name.index("EEG P4-REF")]
    channeled_data[8] = raw_data[ch_name.index("EEG O1-REF")]
    channeled_data[9] = raw_data[ch_name.index("EEG O2-REF")]
    channeled_data[10] = raw_data[ch_name.index("EEG F7-REF")]
    channeled_data[11] = raw_data[ch_name.index("EEG F8-REF")]
    channeled_data[12] = raw_data[ch_name.index("EEG T3-REF")] # ~ T7
    channeled_data[13] = raw_data[ch_name.index("EEG T4-REF")] # ~ T8
    channeled_data[14] = raw_data[ch_name.index("EEG T6-REF")] # ~ P6
    channeled_data[15] = raw_data[ch_name.index("EEG T5-REF")] # ~ P5
    channeled_data[16] = raw_data[ch_name.index("EEG FZ-REF")]
    channeled_data[17] = raw_data[ch_name.index("EEG CZ-REF")]
    channeled_data[18] = raw_data[ch_name.index("EEG PZ-REF")]
    
    return channeled_data


def Channel_selection(channeled_data, raw_data, ch_name):

    '''
    Epoch:
                            eeg.af3', 'eeg.af4'           
                 'eeg.f7', 'eeg.f3',   'eeg.f4', 'eeg.f8'
                 
         'eeg.fc5',                                     'eeg.fc6'
          
    'eeg.t7',                                              'eeg.t8'
            'eeg.p7',                                  'eeg.p8'

                            'eeg.o1', 'eeg.o2'
    
    '''
    '''
    Temple Cap:
                            Fp1   |Fpz|   Fp2
                F7    F3          |Fz|        F4    F8
    A1 ---  T3    C3              |Cz|          C4      T4 --- A2
                T5    P3          |Pz|         P4   T6
                          O1      |Oz|     O2

    '''
    # Subtract A1 from left-side electrodes
    channeled_data[0] = raw_data[ch_name.index("EEG FP1-REF")] # ~ AF3
    channeled_data[1] = raw_data[ch_name.index("EEG F7-REF")] # F7
    channeled_data[2] = raw_data[ch_name.index("EEG F3-REF")] # F3
    channeled_data[3] = raw_data[ch_name.index("EEG C3-REF")] # ~ FC5
    channeled_data[4] = raw_data[ch_name.index("EEG T3-REF")] # ~ T7
    channeled_data[5] = raw_data[ch_name.index("EEG T5-REF")] # ~ P7
    channeled_data[6] = raw_data[ch_name.index("EEG O1-REF")] # O1
    channeled_data[7] = raw_data[ch_name.index("EEG O2-REF")] # O2
    channeled_data[8] = raw_data[ch_name.index("EEG T6-REF")] # ~ P8
    channeled_data[9] = raw_data[ch_name.index("EEG T4-REF")] # ~ T8
    channeled_data[10] = raw_data[ch_name.index("EEG C4-REF")] # ~ FC6
    channeled_data[11] = raw_data[ch_name.index("EEG F4-REF")] # F4
    channeled_data[12] = raw_data[ch_name.index("EEG F8-REF")] # F8
    channeled_data[13] = raw_data[ch_name.index("EEG FP2-REF")] # ~ AF4

    return channeled_data


def Linked_Ears_Reference(channeled_data, raw_data, ch_name):

    '''
                            Fp1   |Fpz|   Fp2
                F7    F3          |Fz|        F4    F8
    A1 ---  T3    C3              |Cz|          C4      T4 --- A2
                T5    P3          |Pz|         P4   T6
                          O1      |Oz|     O2

    '''
    # Subtract A1 from left-side electrodes
    channeled_data[0] = raw_data[ch_name.index("EEG FP1-REF")] - raw_data[ch_name.index("EEG A1-REF")] # ~ AF3
    channeled_data[1] = raw_data[ch_name.index("EEG F7-REF")] - raw_data[ch_name.index("EEG A1-REF")] # F7
    channeled_data[2] = raw_data[ch_name.index("EEG F3-REF")] - raw_data[ch_name.index("EEG A1-REF")] # F3
    channeled_data[3] = raw_data[ch_name.index("EEG C3-REF")] - raw_data[ch_name.index("EEG A1-REF")] # ~ FC5
    channeled_data[4] = raw_data[ch_name.index("EEG T3-REF")] - raw_data[ch_name.index("EEG A1-REF")]# ~ T7
    channeled_data[5] = raw_data[ch_name.index("EEG T5-REF")] - raw_data[ch_name.index("EEG A1-REF")] # ~ P7
    channeled_data[6] = raw_data[ch_name.index("EEG O1-REF")] - raw_data[ch_name.index("EEG A1-REF")] # O1

    # Subtract A2 from right-side electrodes
    channeled_data[7] = raw_data[ch_name.index("EEG O2-REF")] - raw_data[ch_name.index("EEG A2-REF")] # O2
    channeled_data[8] = raw_data[ch_name.index("EEG T6-REF")] - raw_data[ch_name.index("EEG A2-REF")] # ~ P8
    channeled_data[9] = raw_data[ch_name.index("EEG T4-REF")] - raw_data[ch_name.index("EEG A2-REF")] # ~ T8
    channeled_data[10] = raw_data[ch_name.index("EEG C4-REF")] - raw_data[ch_name.index("EEG A2-REF")] # ~ FC6
    channeled_data[11] = raw_data[ch_name.index("EEG F4-REF")] - raw_data[ch_name.index("EEG A2-REF")] # F4
    channeled_data[12] = raw_data[ch_name.index("EEG F8-REF")] - raw_data[ch_name.index("EEG A2-REF")] # F8
    channeled_data[13] = raw_data[ch_name.index("EEG FP2-REF")] - raw_data[ch_name.index("EEG A2-REF")] # ~ AF4

    return channeled_data


def windowed_majority_labeling(values, values_clean, label, ids, window_size, step):
    windowed_samples = []
    windowed_samples_clean = []
    window_labels = []
    window_ids = []

    num_samples  = values.shape[1]
    # values = emotiv_clean(values)
    for i in range(0, num_samples - window_size + 1, step):
        windowed_sample = values[:, i:i + window_size]
        windowed_sample_clean = values_clean[:, i:i + window_size]
        windowed_samples.append(windowed_sample)
        windowed_samples_clean.append(windowed_sample_clean)
        window_labels.append(label)
        window_ids.append(ids)

    windowed_samples = np.array(windowed_samples)
    windowed_samples_clean = np.array(windowed_samples_clean)
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)

    return windowed_samples, windowed_samples_clean, window_labels, window_ids


def windowed_majority_labeling_list(values_list, label, ids, window_size, step):
    windowed_samples = []
    window_labels = []
    window_ids = []

    for subj in range(len(values_list)):
        num_samples  = values_list[subj].shape[1]
        values = values_list[subj]
        values = emotiv_clean(values)
        for i in range(0, num_samples - window_size + 1, step):
            windowed_sample = values[:, i:i + window_size]
            windowed_samples.append(windowed_sample)
            window_labels.append(label[subj])
            window_ids.append(ids[subj])

    windowed_samples = np.array(windowed_samples)
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)

    return windowed_samples, window_labels, window_ids


def segment_save(source_path, target_path):
    for file in os.listdir(source_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(source_path, file)
                # Load the pickle file
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                windowed_samples, windowed_samples_Clean, window_labels, window_ids = windowed_majority_labeling (data['X'], data['X_Clean'], data['y'], data['id'], 
                                                                                              window_size, stride)
                file_id = os.path.splitext(file)[0]  # Remove file extension

                # Save each windowed sample in a separate file
                for idx, (sample, sample_clean, label, sample_id) in enumerate(zip(windowed_samples, windowed_samples_Clean, window_labels, window_ids)):
                    # Create a unique filename for each sample
                    individual_file_path = os.path.join(target_path, f"{file_id}_{idx}.pkl")
                    # Create the individual data dictionary
                    data_dict = {'X': sample, 'X_Clean':sample_clean, 'y': label, 'id': sample_id}
                    # Save to a separate file
                    with open(individual_file_path, 'wb') as f:
                        pickle.dump(data_dict, f)


def shuffle_save(data, labels, ids, save_path):
    """
    Shuffle the data, labels, and IDs while maintaining their alignment,
    and save them as a dictionary in a .npy file.

    Parameters:
    - data (np.ndarray): Data array of shape (N, ...)
    - labels (np.ndarray): Label array of shape (N,)
    - ids (np.ndarray): ID array of shape (N,)
    - save_path (str): Path to save the .npy file

    Returns:
    - None
    """
    # Ensure the arrays have the same length
    assert data.shape[0] == labels.shape[0] == ids.shape[0], "Arrays must have the same length."

    # Generate a permutation of indices
    permutation = np.random.permutation(data.shape[0])

    # Shuffle the arrays using the permutation
    data_shuffled = data[permutation]
    labels_shuffled = labels[permutation]
    ids_shuffled = ids[permutation]

    # Create a dictionary to store the data, labels, and IDs
    data_dict = {
        "data": data_shuffled,
        "label": labels_shuffled,
        "id": ids_shuffled
    }
    # Save the dictionary using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Shuffled data saved to {save_path}")



if __name__ == '__main__':
    raw_data_path = '/data/datasets_public/TUAB/edf'
    save_path = '/data/datasets_public/TUAB/processed_128hz_1280seqlen'
    window_size = 1280
    stride = 1280
    TUAB(raw_data_path, save_path, window_size, stride)



