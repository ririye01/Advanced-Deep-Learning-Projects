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
from torch.optim.lr_scheduler import StepLR
from huggingface_hub import PyTorchModelHubMixin
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
class ViT(nn.Module, PyTorchModelHubMixin):
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
def main(info = None):

    # Device Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Push Model
    if info["push"]:
        model = ViT(info["chw"], info["n_patches"], info["n_blocks"], info["hidden_d"], info["n_heads"], info["out_d"]).to(device)
        model.load_state_dict(torch.load(info["checkpoint_path"], map_location = torch.device(device)))
        model.eval()
        model.push_to_hub(info["hf_dataset_repo_name"], token = info["hf_token"])
        print(f"Model Pushed To {info['hf_dataset_repo_name']}")
        return None

    # Load Data, Transform
    # train_dataset = CustomImageNetDataset('../data_imagenet/train')
    # test_dataset = CustomImageNetDataset('../data_imagenet/val', train = False)
    # train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    # test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

    # Loading Data
    transform = ToTensor()
    train_set = MNIST(root = "./../datasets", train = True, download = True, transform = transform) # num_workers = 4, pin_memory = True)
    test_set = MNIST(root = "./../datasets", train = False, download = True, transform = transform) # num_workers = 4, pin_memory = True)
    train_loader = DataLoader(train_set, shuffle = True, batch_size = info["batch_size"])
    test_loader = DataLoader(test_set, shuffle = False, batch_size = info["batch_size"])

    # Model Configuration
    model = ViT(info["chw"], info["n_patches"], info["n_blocks"], info["hidden_d"], info["n_heads"], info["out_d"]).to(device)

    # Initialize WandB
    wandb.init(project = info["wandb_project"], entity = info["wandb_entity"], id = None)
    wandb.watch(model, log = "all")

    # Log Hyperparameters
    wandb.config = {"learning_rate": info["learning_rate"], "n_patches": info["n_patches"],
        "n_blocks": info["n_blocks"], "hidden_d": info["hidden_d"], "n_heads": info["n_heads"]}

    # Optimizer, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr = info["learning_rate"], betas = (0.9, 0.999), weight_decay = info["weight_decay"], eps = 1e-7)
    scheduler = StepLR(optimizer, step_size = info["step_lr_stepsize"], gamma = 0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Learning Rate Warmup
    WARMUP_EPOCHS = info["warmup_epochs"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, total_iters = WARMUP_EPOCHS)

    # Run Epochs
    for epoch in trange(info["num_epochs"]):

        # Initialize Information
        train_loss, train_correct, train_total = 0, 0, 0

        # Batch Loop
        for x, y in tqdm(train_loader, desc = "Training", leave = False):

            # Move Batch To Device
            x, y = x.to(device), y.to(device)

            # Forward Pass, Loss, Backward Pass
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()
            train_loss += loss.item()
            train_correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
            train_total += y.size(0)

        # Apply Warmup For First Epochs, Then Scheduler
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Compute Loss, Accuracy
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # Model Evaluation (Same For Training)
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x_val, y_val in tqdm(test_loader, desc = "Testing"):
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_val_pred = model(x_val)
                val_loss += criterion(y_val_pred, y_val).item()
                val_correct += (y_val_pred.argmax(1) == y_val).type(torch.float).sum().item()
                val_total += y_val.size(0)

        # Compute Loss, Accuracy
        val_loss /= len(test_loader)
        val_accuracy = val_correct / val_total

        # Logging
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Print Information
        print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_accuracy:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_accuracy:.4f}")
        
    # Save Model
    model_save_path = "./vit_mnist_model_bigger.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model Saved!")

if __name__ == "__main__":

    # Configurations
    info = {
        "chw": (1, 28, 28),
        "n_patches": 7,
        "n_blocks": 6, # 4
        "hidden_d": 64, # 8
        "n_heads": 8, # 4
        "out_d": 10, # 10
        "batch_size": 128,
        "learning_rate": 1e-3,
        "warmup_epochs": 5,
        "num_epochs": 50,
        "wandb_project": "",
        "wandb_entity": "",
        "weight_decay": 0.01,
        "step_lr_stepsize": 10,
        "push": False,
        "checkpoint_path": "",
        "hf_dataset_repo_name": "",
        "hf_token": "",
    }

    main(info)
