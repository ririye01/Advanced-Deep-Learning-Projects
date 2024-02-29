# External Imports
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
from torchvision import transforms, datasets

# Seeding (Not Needed)
# np.random.seed(0)
# torch.manual_seed(0)

# Vit Config Dataclass
@dataclass
class VITConfig:
    block_size: int = None # TODO: Choose block_size
    n_head: int = 2  # TODO: Choose n_head
    n_embed: int = 8 # TODO: Choose n_embed
    dropout: float = 0.0 # TODO: Do we want dropout? Just change the value
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # TODO: Add other configs ? 🤔

# Utility Functions: Patchify, Positional Embeddings

# Patchify: Divide Images Into Patches
def patchify(images, n_patches):

    # Obtain Image Shape
    n, c, h, w = images.shape
    assert h == w, "Patchify Method Implemented For Square Images Only"

    # Initialize Patches With Zeros, Compute Patch Size
    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    # Iterate Over Images, Compute Patches
    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

# Positional Embeddings: Generate Positional Embeddings For Transformer
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

# Simple Multi-Head Self-Attention Module
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

# ViT Block Defining Architecture
class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio = 4):
        super(ViTBlock, self).__init__()

        # Hidden Dim, No. Heads
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        # Norm, Attention, Norm, MLP
        self.norm1 = nn.LayerNorm(hidden_d)
        self.attn = MultiHeadSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )
    
    # Attention, Then Linear Layer
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        out = out + self.mlp(self.norm2(x))
        return out
    
# ViT Model (Pass Images Through Transformer Encoder Blocks, Then Classification MLP)
class ViT(nn.Module):
    def __init__(self, chw, n_patches = 7, n_blocks = 2, hidden_d = 8, n_heads = 2, out_d = 10):
        super(ViT, self).__init__()

         # Attributes
        self.chw = chw  # ( C Channels, H Height, W Width )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input, Patch Sizes
        assert (
            chw[1] % n_patches == 0
        ), "Input Shape Not Entirely Divisible By No. Patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input Shape Not Entirely Divisible By No. Patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear Mapper (Embedding)
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable Classification Token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional Embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent = False,
        )

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # Classification MLP
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim = -1))

    def forward(self, images):

        # Dividing Images Into Patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running Linear Layer Tokenization
        # Map Vector Corresponding To Each Patch To Hidden Size Dim
        tokens = self.linear_mapper(patches)

        # Adding Class Token
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim = 1)

        # Adding Positional Embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting Class Token
        out = out[:, 0]

        # Map To Output Dim, Output Category Distribution
        return self.mlp(out)
    
# Training
def main():

    # Transform Initialization
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # Load Data, Transform
    train_set = datasets.ImageFolder(root = "../data_imagenet/train", transform = transform)
    test_set = datasets.ImageFolder(root = "../data_imagenet/val", transform = transform)
    train_loader = DataLoader(train_set, shuffle = True, batch_size = 128)
    test_loader = DataLoader(test_set, shuffle = False, batch_size = 128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )
    model = ViT(
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)
    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()