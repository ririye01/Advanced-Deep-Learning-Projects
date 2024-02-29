import numpy as np

from tqdm import tqdm, trange

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
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
        super(MultiHeadSelfAttention).__init__()
        assert config.n_embed % config.n_head == 0, f"Can't divide dimension {config.n_embed} into {config.n_head} heads"

        # I am using this linear layer to initialize K, Q, V matrices for ALL heads in ONE go, instead of using a for-loop. More efficient.
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)

        # This linear layer will produce the output of this module
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        
        # SUGGESTION: Use Flash Attention? PyTorch >= 2.0 has it built-in. Would be nice if we want to go BRRRRR 🚀
        self.flash = None

    def forward(self, sequences):
        B, T, C = sequences.size() # batch size, sequence length, embedding dimensionality (n_embed)

        # Generate the Q, K, V matrices in ONE go... THEN separate them in three matrices 🙂
        # TORCH SPLIT REFERENCE: https://pytorch.org/docs/stable/generated/torch.split.html
        q, k, v = self.c_attn(sequences).split(self.n_embed, dim=2)

        # NOTE: Instead of using a for-loop, like in the blog we can introduce a "head" dimension. Why? To process all the heads in parallel
        # REFERENCE: Learned that from Andrej's NanoGPT. It's pretty smart 🧠
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # TODO: Flash Attention line would come here
            pass
        else:
            # Manual implementation of attention.
            # 1 - Dot product all the queries with the keys. NOTE: It does it for all the heads in ONE go.
            attn_scr = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # 2 - Softmax so that the attention scores get mapped between 0 and 1
            attn_scr = F.softmax(attn_scr, dim=-1)

            # 3 - Compute weighted sum of 'v' vectors using attention scores
            y = attn_scr @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.c_proj(y)

        return y

# Trevor
class ViTBlock(nn.Module):
    pass
    
# Trevor
class ViT(nn.Module):
    pass