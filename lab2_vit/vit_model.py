# External Imports
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
import os
import pickle
import wandb

# Seeding (Not Needed)
# np.random.seed(0)
# torch.manual_seed(0)

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
    def __init__(self, d, n_heads = 2):
        super(MultiHeadSelfAttention, self).__init__()

        # Set Attributes
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't Divide Dimension {d} Into {n_heads} Heads"
        d_head = int(d / n_heads)

        # Projections
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, sequences):
        result = []
        for sequence in sequences: # (N, seq_length, token_dim)
            seq_result = []
            for head in range(self.n_heads): # (N, seq_length, n_heads, token_dim / n_heads)

                # Attn Mechanism (Project, Softmax, Multiply, Sum, Concatenate)
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim = 0) for r in result]) # (N, seq_length, item_dim) 

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
        out = x + self.attn(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
    
# ViT Model (Pass Images Through Transformer Encoder Blocks, Then Classification MLP)
class ViT(nn.Module):
    def __init__(self, chw, n_patches = 8, n_blocks = 2, hidden_d = 8, n_heads = 2, out_d = 10):
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

# Load ImageNet => PyTorch Tensors
class CustomImageNetDataset(Dataset):

    # Train True For Train Dataset, False For Test
    def __init__(self, root_dir, train = True):
        self.root_dir = root_dir
        self.train = train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
        self.data, self.labels = self.load_data()

    # Load Train / Test Data - Adjust Based On Data Structure
    def load_data(self):
        data, labels = [], []
        num_batches = 10 if self.train else 1
        for i in range(1, num_batches + 1):
            batch_path = os.path.join(self.root_dir, f'train_data_batch_{i}') if self.train else os.path.join(self.root_dir, 'val_data')
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f)
            data.append(batch['data'])
            labels += batch['labels']
        data = np.concatenate(data, axis = 0)
        labels = [label - 1 for label in labels]
        return data, labels

    # Length Attribute
    def __len__(self):
        return len(self.labels)

    # Convert Flattened Image -> C X H X W -> NumPy Array
    def __getitem__(self, idx):
        img_flat, label = self.data[idx], self.labels[idx]
        
        # Reshape Flat Array Into RGB Image (64 X 64 X 3)
        red_channel = img_flat[0:4096].reshape(64, 64)
        green_channel = img_flat[4096:4096*2].reshape(64, 64)
        blue_channel = img_flat[4096*2:4096*3].reshape(64, 64)
        img_rgb = np.stack((red_channel, green_channel, blue_channel), axis = -1)

        # Apply Transformations, Return
        img_rgb = self.transform(img_rgb)
        return img_rgb, label

# Training
def main():

    # Load Data, Transform (Can Add `num_workers = 4` To DataLoader)
    # train_dataset = CustomImageNetDataset('../data_imagenet/train')
    # test_dataset = CustomImageNetDataset('../data_imagenet/val', train = False)
    # train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    # test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

    # Loading Data
    transform = ToTensor()
    train_set = MNIST(root = "./../datasets", train = True, download = True, transform = transform)
    test_set = MNIST(root = "./../datasets", train = False, download = True, transform = transform)
    train_loader = DataLoader(train_set, shuffle = True, batch_size = 128)
    test_loader = DataLoader(test_set, shuffle = False, batch_size = 128)

    # Defining Model, Training Options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT(
        (1, 28, 28), 
        n_patches = 7, 
        n_blocks = 4, 
        hidden_d = 8, 
        n_heads = 4, 
        out_d = 10
        # (3, 64, 64),
        # n_patches = 16,
        # n_blocks = 4,
        # hidden_d = 512,
        # n_heads = 8,
        # out_d = 1000
    ).to(device)
    N_EPOCHS = 25
    LR = 0.005

    # Initialize WandB
    # wandb.init(project = "ViT_Scratch", entity = "trevorous", id = None)
    # wandb.watch(model, log = "all")

    # Log Hyperparameters
    wandb.config = {
        "learning_rate": 0.005,
        "n_patches": 8,
        "n_blocks": 2,
        "hidden_d": 8,
        "n_heads": 2,
    }

    # Optimizer, Loss
    optimizer = Adam(model.parameters(), lr = LR)
    criterion = CrossEntropyLoss()

    # Run Epochs
    for epoch in trange(N_EPOCHS):

        # Initialize Loss
        train_loss = 0.0

        # Batch Loop
        for batch in tqdm(train_loader, desc = "Training", leave = False):

            # Get Batch, Move To Device
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward Pass, Loss, Backward Pass
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Tell Me Information UwU!
        print(f"Epoch {epoch + 1} / {N_EPOCHS} Loss: {train_loss:.2f}")

        # Logging
        wandb.log({
            "train_loss": train_loss, 
            "epoch": epoch,
        })

    # Test Loop (Same As Before)
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc = "Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            # Need To Argmax And Stuff For Prediction
            correct += torch.sum(torch.argmax(y_hat, dim = 1) == y).detach().cpu().item()
            total += len(x)

        # Tell Me How Bad I Am :)
        print(f"Test Loss: {test_loss:.2f}")
        print(f"Test Accuracy: {correct / total * 100:.2f}%")

        # Log Test Loss, Accuracy
        wandb.log({
            "test_loss": test_loss, 
            "test_accuracy": correct / total * 100, 
            "epoch": epoch
        })
    
    # Save Model
    model_save_path = "./vit_mnist_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model Saved!")

if __name__ == "__main__":
    main()