import os
import pickle
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from torch import Tensor


class ImageNetDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.data, self.labels = self.load_batches()

    def load_batches(self) -> Tuple[np.ndarray, List[int]]:
        data, labels = [], []
        for i in range(1, 11):  # Assuming 10 batches
            batch_path = os.path.join(self.root_dir, f"train_data_batch_{i}")
            with open(batch_path, "rb") as f:
                batch = pickle.load(f)
            data.append(batch["data"])
            labels.extend(batch["labels"])
        data = np.concatenate(data, axis=0)
        return data, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
