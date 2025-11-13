import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from braindecode.datautil import load_concat_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from braindecode.datasets.base import BaseConcatDataset

import logging
logger = logging.getLogger('__main__')

class EEGChallengeDownstreamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "/data/eeg_challenge/challenge1_processed_all_CCD",
        batch_size: int = 128,
        valid_frac=0.0,
        test_frac=0.2,
        seed=2025,
        num_workers: int = 0,
    ):
        """PyTorch Lightning DataModule for EEG Challenge downstream tasks.
        data_dir: Directory containing the preprocessed EEG data.
        batch_size: Batch size for DataLoader.
        valid_frac: Fraction of data to use for validation. Current implementation does not use a validation set.
        test_frac: Fraction of data to use for testing.
        seed: Random seed for reproducibility.
        num_workers: Number of workers for DataLoader.
        """

        super().__init__()

        self.data_dir = Path(data_dir)
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.seed = seed

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        single_windows_loaded = load_concat_dataset(self.data_dir, preload=True, n_jobs=self.num_workers)
        meta_information = single_windows_loaded.get_metadata()
        subjects = meta_information["subject"].unique()
        train_subj, valid_test_subject = train_test_split(
            subjects, test_size=(self.valid_frac + self.test_frac), random_state=check_random_state(self.seed), shuffle=True
        )
        valid_subj, test_subj = train_test_split(valid_test_subject, test_size=self.test_frac, random_state=check_random_state(self.seed + 1), shuffle=True)

        assert (set(valid_subj) | set(test_subj) | set(train_subj)) == set(subjects)
        logger.info(f"Total subjects: {len(subjects)} | Train subjects: {train_subj} | Valid subjects: {valid_subj} | Test subjects: {test_subj}")
        subject_split = single_windows_loaded.split("subject")
        train_set = []
        valid_set = []
        test_set = []

        for s in subject_split:
            if s in train_subj:
                train_set.append(subject_split[s])
            elif s in valid_subj:
                valid_set.append(subject_split[s])
            elif s in test_subj:
                test_set.append(subject_split[s])

        self.train_set = BaseConcatDataset(train_set)
        self.val_set = BaseConcatDataset(valid_set)
        self.test_set = BaseConcatDataset(test_set)
        logging.info(
            f"Loaded dataset with train: {len(self.train_set)} samples | valid: {len(self.val_set)} samples | test: {len(self.test_set)} samples from {self.data_dir}"
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_set is None:
            raise ValueError("train_set is not initialized.")
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_set is None:
            raise ValueError("val_set is not initialized.")
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_set is None:
            raise ValueError("test_set is not initialized.")
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def get_metadata(self):
        """Return metadata for the dataset."""
        if self.train_set is not None:
            return self.train_set.get_metadata()
        return None

    @property
    def n_channels(self) -> int:
        """Return number of EEG channels."""
        return 129  # Default for this dataset

def load_CCD(config):
    processed_data_dir = config['data_dir']
    logger.info(f"Loading CCD data from {processed_data_dir}")
    data_module = EEGChallengeDownstreamDataModule(
                    data_dir=processed_data_dir,
                    batch_size=32,
                    valid_frac=0.0, # we dont use val set for now
                    test_frac=0.2,
                    seed=2025,
                    num_workers=8)
    data_module.setup()
    logger.info(f"CCD data loaded successfully")
    return data_module


def dataloader_to_numpy_dict(dataloader):
    X_list, Y_list = [], []
    for batch in dataloader:
        # Assuming batch = (X_raw, Y, ids)
        X, Y, ids = batch
        X_list.append(X.cpu().numpy())
        Y_list.append(Y.cpu().numpy().astype(np.float32).reshape(-1, 1))
        # id_list.append(ids.cpu().numpy())

    return (np.concatenate(X_list, axis=0),
            np.concatenate(Y_list, axis=0))


if __name__ == '__main__':
    # /data/eeg_challenge/challenge1_processed_all_CCD
    config = {
        'data_dir': '/data/eeg_challenge/challenge1_processed_mini_CCD',
        'batch_size': 32,
        'valid_frac': 0.0,
        'test_frac': 0.2,
        'seed': 2025,
        'num_workers': 8
    }
    Data = load_CCD(config)
    train_loader = DataLoader(Data.train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(Data.val_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(Data.test_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # Extract data for each split
    X_train, Y_train = dataloader_to_numpy_dict(train_loader)
    X_val, Y_val = dataloader_to_numpy_dict(val_loader)
    X_test, Y_test = dataloader_to_numpy_dict(test_loader)

    # Create dictionary
    data_dict = {
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'X_test': X_test, 'Y_test': Y_test,
    }

    # For NAM
    # X_train, X_train_clean, Y_train, id_train = dataloader_to_numpy_dict(train_loader)
    # X_val, X_val_clean, Y_val, id_val = dataloader_to_numpy_dict(val_loader)
    # X_test, X_test_clean, Y_test, id_test = dataloader_to_numpy_dict(test_loader)

    # # Create dictionary
    # data_dict = {
    #     'X_train': X_train, 'X_train_clean': X_train_clean, 'Y_train': Y_train, 'id_train': id_train,
    #     'X_val': X_val, 'X_val_clean': X_val_clean, 'Y_val': Y_val, 'id_val': id_val,
    #     'X_test': X_test, 'X_test_clean': X_test_clean, 'Y_test': Y_test, 'id_test': id_test
    # }

    # Save with pickle
    save_path = '/data/EEG-X/Processed/CCD/CCD.npy'  
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"✅ Data saved to {save_path}")
    
