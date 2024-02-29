import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

# -----------------------------------------------------------------------------
# Utility functions. Would be nice to move them in another file called 'utils.py' or something...
def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
# -----------------------------------------------------------------------------

# ----------------------------- VIT CONFIG ------------------------------------
@dataclass
class VITConfig:
    block_size: int = None # TODO: Choose block_size
    n_head: int = 2  # TODO: Choose n_head
    n_embed: int = 8 # TODO: Choose n_embed
    dropout: float = 0.0 # TODO: Do we want dropout? Just change the value
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # TODO: Add other configs ? 🤔

# -----------------------------------------------------------------------------
    
# ----------------------------- VIT MODEL -------------------------------------
    
# NOTE: Kassi is implementing VIT Multi head attention module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0, f"Can't divide dimension {config.n_embed} into {config.n_head} heads"

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        # TODO: Other properties...

    def forward(self, sequences):
        # TODO: Implement forward pass
        pass

# Trevor
class ViTBlock(nn.Module):
    pass
    
# Trevor
class ViT(nn.Module):
    pass