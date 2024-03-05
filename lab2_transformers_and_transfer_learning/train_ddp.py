import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import transforms
from torchvision.datasets import MNIST
from transformers import ViTConfig, ViTModel, ViTForImageClassification

import os
from typing import Tuple


def init_distributed() -> None:
    """
    Sets up distributed trining ENV variables and backend.
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    distributed_url = "env://"  # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Try the NCCL backend (has to do with CUDA)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=distributed_url,
            world_size=world_size,
            rank=rank,
        )
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
            backend="gloo",
            init_method=distributed_url,
            world_size=world_size,
            rank=rank,
        )

    # this will make all `.cuda()` calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def _transform_mnist_image() -> transforms.Compose:
    """
    Returns a composed transform to convert MNIST images to the format expected by ViT.
    This typically includes resizing and normalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by ViT
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize for a single channel (gray)
    ])


def _load_mnist_data() -> Tuple[MNIST, MNIST]:
    transform = _transform_mnist_image()
    mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_valset = MNIST(root='./data', train=False, download=True, transform=transform)
    return mnist_trainset, mnist_valset


def _load_vit_model() -> ViTModel:
    configuration = ViTConfig()
    return ViTModel(config=configuration)


def _modify_vit_model_for_mnist(model: ViTModel) -> nn.Module:
    """
    Adjusts the ViT model for MNIST dataset (10 classes).
    """
    num_classes = 10
    # Replace the classifier head with a new one for 10 classes (MNIST)
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    return model


def train(model, data_loader, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)}] Loss: {loss.item():.6f}')

    return total_loss / len(data_loader.dataset)


def validate(model, data_loader, criterion, device):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += criterion(output.logits, target).item()  # Sum up batch loss
            pred = output.logits.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(data_loader.dataset)
    print(f'\nValidation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)\n')


def main() -> None:
    mnist_trainset, mnist_valset = _load_mnist_data()
    model = _load_vit_model()
    model = _modify_vit_model_for_mnist(model).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Wrap the model for distributed training
    model = DDP(model)

    train_sampler = DistributedSampler(mnist_trainset)
    val_sampler = DistributedSampler(mnist_valset)

    train_loader = DataLoader(mnist_trainset, batch_size=64, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(mnist_valset, batch_size=64, shuffle=False, sampler=val_sampler)

    epochs = 15
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train(model, train_loader, criterion, optimizer, epoch, torch.device('cuda'))
        validate(model, val_loader, criterion, torch.device('cuda'))



if __name__ == "__main__":
    init_distributed()
    main()
