import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, random_split
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

        return audio_tensor, label

    def __getitem__(self, index):
        """
        Fetches the sample at the given index. Must be implemented in subclasses.

        Args:
            index (int): Index of the sample.

        Returns:
            Sample from the dataset.
        """
        if self.load_mode == "disk":
            audio_tensor, label = self.load_item(index)
        elif self.load_mode == "ram":
            audio_tensor, label = self.ram_data[index]

        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        
        return audio_tensor, label

    def __str__(self):
        """
        Returns:
            str: Human-readable description of the dataset.
        """
        s = f"Pruned BirdCLEF dataset 🐦\n  Split: {self.split}\n  Number of samples: {len(self.dataset_specs[self.split])}"
        if self.split != "unlabeled":
            s += f"\n  Number of classes: {self.dataset_specs[self.split][-1][1] + 1}"
        return s


class TransformedSubset(Dataset):
    """
    A dataset wrapper for applying a different transform to a subset of a dataset.

    This is useful when you want to change the transformation pipeline (e.g., data augmentation)
    for a specific subset, such as using a different transform for validation or testing
    while keeping the original dataset unchanged.

    Attributes:
        subset (torch.utils.data.Subset): The original subset of the dataset.
        transform (callable, optional): A function/transform that takes in a data sample and returns a transformed version.
    """

    def __init__(self, subset, transform=None):
        """
        Initialize the TransformedSubset.
        Args:
            subset (torch.utils.data.Subset): The subset to wrap.
            transform (callable, optional): Transform to apply to the input data.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieve an item from the subset and apply the transform to the data.
        Args:
            idx (int): Index of the data sample to retrieve.
        Returns:
            tuple: (transformed_data, target), where target is the label or ground truth.
        """
        data, target = self.subset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        """
        Get the number of samples in the subset.
        Returns:
            int: Length of the dataset.
        """
        return len(self.subset)


class PrunedBirdCLEFDataModule(L.LightningDataModule):
    """
    Abstract base class for PyTorch Lightning DataModules in the MO810 course work.
    
    Provides the expected interface for training, validation, and test dataloaders.
    """
    def __init__(self, root_dir: str = "../data/PrunedBirdCLEF", load_mode: str = "disk", train_transform=None, val_transform=None, test_transform=None, batch_size: int = 32, num_workers : int = 4):
        super().__init__()

        self.train_dataset = PrunedBirdCLEFDataset(root_dir, "train", load_mode, train_transform)
        self.val_dataset = PrunedBirdCLEFDataset(root_dir, "val", load_mode, val_transform)
        self.test_dataset = PrunedBirdCLEFDataset(root_dir, "test", load_mode, test_transform)

        self.batch_size = batch_size
        self.num_workers = num_workers

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