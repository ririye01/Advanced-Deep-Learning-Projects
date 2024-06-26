from struct import Struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from transformers import ViTConfig, ViTModel, ViTForImageClassification
import wandb

import os
from typing import Tuple, Literal


def wandb_init() -> None:
    wandb.init(project="mnist-vit-finetune")


def _transform_mnist_image() -> transforms.Compose:
    """
    Returns a composed transform to convert MNIST images to the format expected by ViT.
    Handles the grayscaling of images, resizing, and normalization.
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


def _freeze_entire_model_except_ending_linear_classifier(
    model: nn.Module,
    num_classes: int = 10,
) -> nn.Module:
    """
    Freeze all parameters in a Vision Transformer (ViT) model except for the linear
    classifier at the end, and adjust the classifier for a specified number of classes.

    This function is particularly useful for transfer learning, where the pretrained
    layers of the ViT model are used as feature extractors, and only the final classifier
    is trained for a specific task.

    Parameters
    ----------
    model : ViTModel
        The pre-trained Vision Transformer model.
    num_classes : int, optional
        The number of classes for the final linear classifier, by default 10.

    Returns
    -------
    nn.Module
        The modified ViT model with all layers frozen except the ending linear classifier.
    """
    # Freeze everything except the last layer
    for name, param in model.named_parameters():
        # Required to keep the `requires_grad` setting true for the pooler parameter.
        # This issue is handled later by setting the gradients equal to None
        # immediately after backpropogation every single time.
        if not name.startswith("pooler"):
          param.requires_grad = False

    # Replace the classifier head with a new one for `num_classes` classes
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    return model


def train_on_epoch(
    model: nn.Module,
    training_data_loader: DataLoader,
    val_data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    epoch: int,
    device: Literal["cuda", "mps", "cpu"],
) -> float:
    """
    Run through an epoch.
    """
    total_training_loss = 0
    for batch_idx, (data, target) in enumerate(training_data_loader):
        model.train()
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # ViT encoder has 2 outputs, final embedding vectors for all image tokens and a stack of attention weights.
        # Here we are not using attention weights during training/validation.
        # embeddings: [batch_size, n_tokens, embedding dim]  e.g.[16, 197, 768]
        outputs = model(data)
        pooler_output = model.classifier(outputs.pooler_output)
        training_loss = criterion(pooler_output, target)

        # BACKPROPOGATE
        training_loss.backward()

        # ZERO OUT THE GRADIENTS TO MAKE SURE THAT THEY ARE NOT UPDATED IN BACKPROPOGATION
        # WE NEEDED TO DO THIS BECAUSE MODEL REFUSED TO TRAIN WITHOUT IT
        model.pooler.dense.weight.grad = model.pooler.dense.bias.grad = None

        # TAKE A STEP THROUGH THE LOSS LANDSCAPE.
        optimizer.step()
        total_training_loss += training_loss.item()

        # LOG BATCH TRAINING LOSS.
        wandb.log({
            "training_loss": training_loss.item(),
        })

        # OUTPUT VALIDATION RESULTS AND LOG THEM IN WEIGHTS AND BIASES.
        if batch_idx % 500 == 0 and batch_idx != 0:
            val_loss, val_accuracy = validate_on_epoch(
                model=model,
                data_loader=val_data_loader,
                criterion=criterion,
                device=device,
                in_train_loop=False,
            )

            wandb.log({
                "training_loss": training_loss.item(),
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            })

            print(
                f"Epoch: {epoch} [{batch_idx * len(data)}/{len(training_data_loader.dataset)}]" +
                f"Training Loss: {training_loss.item():.6f} " +
                f"**Validation Accuracy: {val_accuracy:.3f}**"
            )

        print(
            f"Epoch: {epoch} [{batch_idx * len(data)}/{len(training_data_loader.dataset)}]" +
            f"Training Loss: {training_loss.item():.6f} "
        )

    return total_training_loss / len(training_data_loader.dataset)


def validate_on_epoch(
    model: ViTModel,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: Literal[""],
    in_train_loop=False,
) -> Tuple[float, float]:
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # ViT encoder has 2 outputs, final embedding vectors for all image tokens and a stack of attention weights.
            # Here we are not using attention weights during training/validation.
            # embeddings: [batch_size, n_tokens, embedding dim]  e.g.[16, 197, 768]
            outputs = model(data)
            pooler_output = model.classifier(outputs.pooler_output)
            last_hidden_state = outputs.last_hidden_state
            validation_loss += criterion(pooler_output, target).item()  # Sum up batch loss
            pred = pooler_output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 10 == 0 and in_train_loop:
                print(f"Validation Loss: {validation_loss / len(data_loader.dataset):.6f} ")

    # Calculate overall validation loss
    validation_loss = validation_loss / len(data_loader.dataset)
    validation_accuracy = 100. * correct / len(data_loader.dataset)
    print(f"\nValidation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({validation_accuracy:.0f}%)\n")

    # Log metrics to wandb
    wandb.log({
        "val_accuracy_per_epoch": validation_accuracy,
        "val_loss_per_epoch": validation_loss,
    })

    return validation_loss, validation_accuracy


def main() -> None:
    # Cuda is ideal, but 3 of our team members own Macs.
    # So, mps for the win.
    DEVICE: Literal["cuda", "mps", "cpu"] = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(DEVICE)

    mnist_trainset, mnist_valset = _load_mnist_data()

    # Load in ViT model and freeze all the layers except for a single Linear classification
    # layer
    model = _load_vit_model()
    model = _freeze_entire_model_except_ending_linear_classifier(model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Load training and validation in with batch sizes of 64
    train_loader = DataLoader(mnist_trainset, batch_size=64, shuffle=True, sampler=None)
    val_loader = DataLoader(mnist_valset, batch_size=64, shuffle=False, sampler=None)

    NUM_EPOCHS = 15
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        # Train on all the training data in a single epoch.
        train_loss = train_on_epoch(
            model=model,
            training_data_loader=train_loader,
            val_data_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=DEVICE,
        )
        # Report back validation metrics for every epoch.
        val_loss, val_accuracy = validate_on_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )
        val_accuracies.append(val_accuracy)

        # Save model only if it outperforms all the other models on Validation Accuracy.
        if max(val_accuracies) == val_accuracy:
            torch.save(model, f"ViT_MNIST_Finetune-v{epoch}.pth")


if __name__ == "__main__":
    wandb_init()
    main()
    wandb.finish()
