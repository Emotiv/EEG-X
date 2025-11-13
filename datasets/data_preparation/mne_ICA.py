import os
import mne
from mne.io import RawArray
from mne_icalabel import label_components
import numpy as np

class MNE_ICA:
    def __init__(self, device, sfreq):
        # based on the device we define the montage
        if device == 'EPOC':
            montage_file = '/data/EEG-X/RAW/Cap_Configs/Cap_locations/Standard-10-20-Cap14.locs'
            self.mont = mne.channels.read_custom_montage(montage_file)
            self.n_components = 13
        elif device == 'Insight':
            analyze_channels = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
            # Load the standard 10-20 montage
            montage = mne.channels.make_standard_montage('standard_1020')
            # Keep only the specified channels
            channel_positions = {ch: pos for ch, pos in montage.get_positions()['ch_pos'].items() if ch in analyze_channels}
            self.mont = mne.channels.make_dig_montage(ch_pos=channel_positions, coord_frame='head')
            self.n_components = 4
        elif device == 'TUH':
            analyze_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                    'O2', 'F7', 'F8', 'T7', 'T8', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
            # Load the standard 10-20 montage
            montage = mne.channels.make_standard_montage('standard_1020')
            # Keep only the specified channels
            channel_positions = {ch: pos for ch, pos in montage.get_positions()['ch_pos'].items() if ch in analyze_channels}
            self.mont = mne.channels.make_dig_montage(ch_pos=channel_positions, coord_frame='head')
            self.n_components = 18
        elif device == 'BCICIV':
            montage_file = '/data/EEG-X/RAW/Cap_Configs/Cap_locations/Standard-10-20-Cap22.locs'
            self.mont = mne.channels.read_custom_montage(montage_file)
            self.n_components = 21
        elif device == 'MN8' or 'MW20':
            print ('To Do')
        self.Epoch_info = mne.create_info(ch_names=self.mont.ch_names, sfreq=128.,ch_types='eeg')


    def clean_data(self, data):
        ica = mne.preprocessing.ICA(
                n_components=self.n_components,
                max_iter="auto",
                method="infomax",
                random_state=97,
                fit_params=dict(extended=True),
                )   
        Filterd_mne = RawArray(data.copy(), info=self.Epoch_info, verbose=0)
        Filterd_mne = Filterd_mne.set_montage(self.mont)
        Filterd_mne = Filterd_mne.set_eeg_reference("average", verbose=0)

        ica.fit(Filterd_mne)
        ic_labels = label_components(Filterd_mne, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        reconst = Filterd_mne.copy()
        clean_data = ica.apply(reconst, exclude=exclude_idx)
        clean_data = clean_data.get_data()
        return clean_data


# Example usage
if __name__ == "__main__":
    # Create dummy data with shape (14, 5000)
    dummy_data = np.random.randn(14, 5000)
    ica = MNE_ICA(device='EPOC', sfreq=128)
    clean_data = ica.clean_data(dummy_data)