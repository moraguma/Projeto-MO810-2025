import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L
import numpy as np
from pydub import AudioSegment

import json, gdown, os, zipfile

from api.MO810_API import MO810DataModule, MO810Dataset

# Folders
DATASET_URL = "https://drive.google.com/uc?export=download&id=1TyZmku25CZdeUd09He_Ddqcnm27OyyhS"
ROOT_DIR = "../data/PrunedBirdCLEF"

# Data
MAX_LENGTH = 60


class PrunedBirdCLEFDataset(MO810Dataset):
    """
    The PrunedBirdCLEF dataset, modified from BirdCLEF2020 based on MetaAudio's approach, 
    but with a friendlier size of ~5GB.
    
    This class defines the interface that all MO810-compatible datasets should implement,
    including methods for downloading, accessing, and representing samples.
    """

    def __init__(self, dataset_path, split, load_mode="disk", transform=None, *args, **kwargs):
        """
        Arguments:
            dataset_path: String - Path to the dataset
            split: String - The split to be used. Either 'train', 'val', 'test' or 'unlabeled'
            load_mode: String - Either 'disk' or 'ram'. When 'disk', will load each sample from the
                disk as it is requested. When in 'ram', will load all samples into RAM on init
            transform: Callable, optional - Optional transform to be applied on a sample
        """
        super().__init__(*args, **kwargs)

        self.split = split
        self.load_mode = load_mode
        self.transform = transform

        # Download data
        self.dataset_path = dataset_path
        self.download_data()

        # Load dataset
        with open(f"{dataset_path}/dataset_specs.json", "r") as f:
            self.dataset_specs = json.load(f)

        # Load into RAM
        if self.load_mode == "ram":
            self.ram_data = []
            for i in range(len(self.dataset_specs[self.split])):
                self.ram_data.append(self.load_item(i))

    def download_data(self):
        """
        Downloads the dataset.
        """
        if not os.path.isdir(self.dataset_path):
            os.mkdir(self.dataset_path)
            gdown.download(DATASET_URL, f"{self.dataset_path}/temp.zip", quiet=False)

            with zipfile.ZipFile(f"{self.dataset_path}/temp.zip", 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)

            os.remove(f"{ROOT_DIR}/temp.zip")

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset. Must be implemented in subclasses.
        """
        return len(self.dataset_specs[self.split])
    
    def load_item(self, index):
        """
        Returns:
            Tensor: waveform [channel, time] loaded from memory
            int: label of item
            int: frame rate of audio
        """
        if torch.is_tensor(index):
            index = index.item()
        
        local_path, label = self.dataset_specs[self.split][index]

        audio = AudioSegment.from_mp3(f"{self.dataset_path}/{local_path}")
        audio_tensor = Tensor(audio.get_array_of_samples())
        audio_tensor = F.pad(audio_tensor, (0, audio.frame_rate * MAX_LENGTH - audio_tensor.shape[0])).unsqueeze(0)

        return audio_tensor, label, audio.frame_rate

    def __getitem__(self, index):
        """
        Fetches the sample at the given index. Must be implemented in subclasses.

        Args:
            index (int): Index of the sample.

        Returns:
            Sample from the dataset.
        """
        if self.load_mode == "disk":
            return self.load_item(index)
        elif self.load_mode == "ram":
            return self.ram_data[index]
    
    def __str__(self):
        """
        Returns:
            str: Human-readable description of the dataset.
        """
        s = f"Pruned BirdCLEF dataset 🐦\n  Split: {self.split}\n  Number of samples: {len(self.dataset_specs[self.split])}"
        if self.split != "unlabeled":
            s += f"\n  Number of classes: {self.dataset_specs[self.split][-1][1] + 1}"
        return s

class MO810DataModule(L.LightningDataModule):
    """
    Abstract base class for PyTorch Lightning DataModules in the MO810 course work.
    
    Provides the expected interface for training, validation, and test dataloaders.
    """

    def train_dataloader(self, fraction=None, samples_per_class=None):
        """
        Returns:
            DataLoader: Training dataloader. Should support sampling options.
        """
        raise NotImplemented

    def val_dataloader(self):
        """
        Returns:
            DataLoader: Validation dataloader.
        """
        raise NotImplemented

    def test_dataloader(self):
        """
        Returns:
            DataLoader: Test dataloader.
        """
        raise NotImplemented