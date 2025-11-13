'''
Author: Navid

This script prepares the TUEV dataset for analysis. 
TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
The raw data is stored in /data/datasets_public/TUEV/edf.

The code reads the data, applies Emotiv filtering, segments it into 2-second windows, 
and stores the results in data/datasets_public/TUEV/processed_128hz_256seqlen.
This preprocessing makes the TUEV dataset ready for downstream tasks and model compatibility.

'''
import os
import logging
import numpy as np
import argparse
import mne
import pickle
from sklearn import model_selection
from filtering import emotiv_clean
from mne_ICA import MNE_ICA

import sys
sys.path.append('/home/navid/emotiv-ml/')

logger = logging.getLogger(__name__)


def TUEV(args):
    root = args.raw_data_path
    save_path = args.save_path

    train_path = os.path.join(root, "train")
    train_list = list(set([item.split("_")[0] for item in os.listdir(train_path)]))
     # Split the list into training and validation sets (10% validation)
    train_list, val_list = model_selection.train_test_split(train_list, test_size=0.1, random_state=1234)
    print("Train # of subject:" + str(len(train_list)))
    print("Validation # of subject:" + str(len(val_list)))


    test_path = os.path.join(root, "eval")
    test_list = list(set([item.split("_")[0] for item in os.listdir(test_path)]))
    print("Test # of subject:" + str(len(test_list)))

    # ---------------------------------------------------------------------------------------------------------
    # create the raw train, val, test sample folder
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path))
    val_dump_folder = save_path + "/TUEV_val.npy"
    pre_processing(train_path, val_list, val_dump_folder, 'val')
    # --------------------------------------------------------------------------------------------------------------
    train_dump_folder = save_path + "/TUEV_train.npy"
    pre_processing(train_path, train_list, train_dump_folder, 'train')

    # --------------------------------------------------------------------------------------------------------------
    test_dump_folder = save_path + "/TUEV_test.npy"
    pre_processing(test_path, test_list, test_dump_folder, 'test')
    # --------------------------------------------------------------------------------------------------------------
    return


def pre_processing(data_path, data_list, save_path, split):
    # Initialize empty lists to accumulate data
    ica = MNE_ICA(device='TUH', sfreq=128)
    # pc = pickle.load(open(f'/home/navid/emotiv-ml/axon/pickle/prep_c_{2}_{0.75}_{9}.pickle', 'rb'))
    all_segments = []
    all_segments_clean = []
    all_labels = []
    # Initialize lists to store data, labels, and IDs
    for file in os.listdir(data_path):
        if file.split("_")[0] in data_list:
            print("process", file)
            # loading *.edf file
            file_path = os.path.join(data_path, file)
            for edf_file in os.listdir(file_path):
                if edf_file[-4:] == ".edf":
                    try:
                        event, raw = readEDF(file_path + "/" + edf_file )  # event is the .rec file in the form of an array
                    except (ValueError, KeyError):
                        print("something funky happened in " + file_path + "/" + edf_file)
                        continue
                    raw.resample(128)
                    ch_name = raw.ch_names
                    raw_data = raw.get_data()
                    channeled_data = raw_data.copy()[:19]
                    # LE_data = Linked_Ears_Reference(channeled_data, raw_data, ch_name)
                    # Match_data = Channel_selection (channeled_data, raw_data, ch_name)
                    Match_data = Channel_Order(channeled_data, raw_data, ch_name)
                    # Emotiv Filtering ------------------------------------------------------------
                    Match_data_filtered = emotiv_clean(Match_data.T).T # emotiv_clean(lenght, channel)
                    # scale the data from V to uV
                    Match_data_filtered = Match_data_filtered * 1e6
                    # rASR cleaning ---------------------------------------------------------------
                    # Match_data_clean = pc.clean(Match_data_filtered.transpose(1,0)).transpose(1,0)
                    # ICA Cleaning ---------------------------------------------------------------
                    Match_data_clean =  ica.clean_data(Match_data_filtered)
                    selected_segments, selected_segments_clean, selected_labels = select_signal_by_event(Match_data_filtered, Match_data_clean, event)
                    # Append segments and labels to global lists
                    all_segments.extend(selected_segments)  # Use extend to avoid nesting lists
                    all_segments_clean.extend(selected_segments_clean)
                    all_labels.extend(selected_labels)

    # Convert to numpy arrays
    X = np.array(all_segments) 
    X_clean = np.array(all_segments_clean ) 
    y = np.array(all_labels) 
    # Create the data dictionary
    data_dict = {f'X_{split}': X, f'X_{split}_clean': X_clean, f'y_{split}': y}
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName)
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return eventData, Rawdata


def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 250.0
    [numChan, numPoints] = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return features, offending_channel, labels


def select_signal_by_event(signal, signal_clean, event,  sampling_rate=128, excluded_artifacts=None):
    """
    Selects signal segments based on the given event data.
    
    Parameters:
        signal (numpy.ndarray): The signal data, with shape (channels, samples).
        event (numpy.ndarray): Array containing event data.
        sampling_rate (int): Sampling frequency of the signal (default: 128 Hz).
        excluded_channels (list): List of channels to exclude (default: None).
        
    Returns:
        list: A list of signal segments corresponding to the event intervals.
    """
    

    if excluded_artifacts is None:
        excluded_artifacts = [4, 5]
    # Filter events to exclude rows with unwanted classes
    filtered_events = event[~np.isin(event[:, 3], excluded_artifacts)]
    
    selected_segments = []
    selected_segments_clean = []
    selected_label = []
    
    for row in filtered_events:
        channel, start_time, end_time, label = row
        # Adjust the start and end times based on the conditions
        if start_time < 0.5: # Ensure no negative start time
            extended_start_time = start_time
            extended_end_time = start_time + 2    
        else:
            extended_start_time = start_time - 0.5  
            extended_end_time = start_time + 1.5
        '''
        if (extended_end_time * sampling_rate) > signal.shape[1]:
            extended_start_time = start_time - 1  
            extended_end_time = end_time
        '''



        start_idx = int(extended_start_time * sampling_rate)
        end_idx = int(extended_end_time * sampling_rate)
        
        # Select the segment from the signal
        segment = signal[:, start_idx:end_idx]
        segment_clean = signal_clean[:, start_idx:end_idx]
        # Binary classification
        '''
        if segment_clean.shape == (19, 256) and label in [1, 2, 3, 6]:
            class_label = 1 if label in [1, 2, 3] else 0
            selected_segments.append(segment)
            selected_segments_clean.append(segment_clean)
            selected_label.append(class_label)
        '''
        # Multi-class classification
        label_mapping = {1: 0, 2: 1, 3: 2, 6: 3}
        if segment_clean.shape == (19, 256) and label in label_mapping:
            class_label = label_mapping[label]
            selected_segments.append(segment)
            selected_segments_clean.append(segment_clean)
            selected_label.append(class_label)


    return selected_segments, selected_segments_clean, selected_label


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


def windowed_majority_labeling(values, label, ids, window_size, step):
    windowed_samples = []
    window_labels = []
    window_ids = []

    num_samples  = values.shape[1]
    values = emotiv_clean(values)
    for i in range(0, num_samples - window_size + 1, step):
        windowed_sample = values[:, i:i + window_size]
        windowed_samples.append(windowed_sample)
        window_labels.append(label)
        window_ids.append(ids)

    windowed_samples = np.array(windowed_samples)
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)

    return windowed_samples, window_labels, window_ids


if __name__ == '__main__':
        # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the STEW dataset.')
    parser.add_argument('--raw_data_path', type=str, default='/data/EEG-X/RAW/TUEV', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='/data/EEG-X/Processed/TUEV', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=256, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'Non'], help='Cleaning method to apply')
    dash_line = '-'.join('' for x in range(100))
    # Parse the arguments
    args = parser.parse_args()
    print(dash_line)
    print("Loading the raw data")
    X_datas, y_datas, id_datas = TUEV(args)
