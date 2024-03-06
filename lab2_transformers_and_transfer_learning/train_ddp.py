import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from transformers import ViTConfig, ViTModel, ViTForImageClassification

import os
from typing import Tuple, Literal


def _transform_mnist_image() -> transforms.Compose:
    """
    Returns a composed transform to convert MNIST images to the format expected by ViT.
    This typically includes resizing and normalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by ViT
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize across 3 output channels
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


def train(model, data_loader, criterion, optimizer, epoch, device) -> float:
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        # print(outputs)
        # print(type(outputs))
        # input()
        loss = criterion(outputs.logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)}] Loss: {loss.item():.6f}')

    return total_loss / len(data_loader.dataset)


def validate(model, data_loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output)
            print(type(output))
            input()
            validation_loss += criterion(output.logits, target).item()  # Sum up batch loss
            pred = output.logits.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss = validation_loss / len(data_loader.dataset)
    validation_accuracy = 100. * correct / len(data_loader.dataset)
    print(f'\nValidation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({validation_accuracy:.0f}%)\n')
    return validation_loss, validation_accuracy


def main() -> None:
    DEVICE: Literal["cuda", "mps", "cpu"] = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    mnist_trainset, mnist_valset = _load_mnist_data()

    model = _load_vit_model()
    model = _modify_vit_model_for_mnist(model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(mnist_trainset, batch_size=32, shuffle=True, sampler=None)
    val_loader = DataLoader(mnist_valset, batch_size=32, shuffle=True, sampler=None)

    NUM_EPOCHS = 15
    for epoch in range(NUM_EPOCHS):
        train_loss = train(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=DEVICE,
        )
        val_loss = validate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )



if __name__ == "__main__":
    main()
