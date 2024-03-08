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


def _freeze_entire_model_except_ending_linear_classifier(
    model: ViTModel,
    num_classes: int = 10,
) -> nn.Module:
    """
    Adjusts the ViT model for MNIST dataset (10 classes).
    """
    # Freeze everything except the prediction layer
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Replace the classifier head with a new one for 10 classes (MNIST)
    # This makes sure that the bottleneck features (features in the Linear layer)
    # are trainable.
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    return model


def _freeze_all_layers_except_bottleneck_and_linear_classifier(
    model: ViTModel,
    num_classes: int = 10,
) -> nn.Module:
    """
    Adjusts the ViT model for MNIST dataset (10 classes).
    """
    # Freeze all but the bottleneck features.
    for name, param in model.named_parameters():
        is_not_trainable_layer = (
            not name.startswith("encoder.layer.11")
            and not name.startswith("pooler")
            and not name.startswith("layernorm")
        )
        if is_not_trainable_layer:
            param.requires_grad = False

    # Replace the classifier head with a new one for 10 classes (MNIST)
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    return model


def train_on_epoch(
    model,
    training_data_loader,
    val_data_loader,
    criterion,
    optimizer,
    epoch,
    device,
) -> float:
    total_training_loss = 0
    for batch_idx, (data, target) in enumerate(training_data_loader):
        model.train()
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # ViT encoder has 2 outputs, final embedding vectors for all image tokens and a stack of attention weights.
        # Here we are not using attention weights during training/validation.
        # embeddings: [batch_size, n_tokens, embedding dim]  e.g.[16, 197, 768]
        outputs = model(data)
        pooler_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state

        ## Extract [CLS] token (at index 0) 's embeddings used for classification ##
        embedding_cls_token = last_hidden_state[:,0,:] # [batch_size, embedding dim]

        training_loss = criterion(embedding_cls_token, target)
        training_loss.backward()

        optimizer.step()
        total_training_loss += training_loss.item()

        wandb.log({
            "training_loss": training_loss.item(),
        })

        if batch_idx % 100 == 0:
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


def validate_on_epoch(model, data_loader, criterion, device, in_train_loop=False) -> Tuple[float, float]:
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
            pooler_output = outputs.pooler_output
            last_hidden_state = outputs.last_hidden_state

            ## Extract [CLS] token (at index 0) 's embeddings used for classification ##
            embedding_cls_token = outputs.last_hidden_state[:,0,:] # [batch_size, embedding dim]

            validation_loss += criterion(embedding_cls_token, target).item()  # Sum up batch loss
            pred = embedding_cls_token.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 10 == 0 and in_train_loop:
                print(f"Validation Loss: {validation_loss / len(data_loader.dataset):.6f} ")

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
    DEVICE: Literal["cuda", "mps", "cpu"] = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    mnist_trainset, mnist_valset = _load_mnist_data()

    # WE RAN THIS CODE TWICE. THE ONLY THING WE SWITCHED UP IS THAT IN ONE OF THEM,
    # WE FROZE ALL THE LAYERS EXCEPT THE FINAL LINEAR CLASSIFIER, AND IN THE OTHER
    # ONE, WE FROZE ALL THE LAYERS EXCEPT THE FINAL LAYER IN THE MODEL AND THE LINEAR
    # CLASSIFIER.
    model = _load_vit_model()
    model = _freeze_all_layers_except_bottleneck_and_linear_classifier(model).to(DEVICE)
    # model = _freeze_entire_model_except_ending_linear_classifier(model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)

    train_loader = DataLoader(mnist_trainset, batch_size=64, shuffle=True, sampler=None)
    val_loader = DataLoader(mnist_valset, batch_size=64, shuffle=False, sampler=None)

    NUM_EPOCHS = 15
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_on_epoch(
            model=model,
            training_data_loader=train_loader,
            val_data_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=DEVICE,
        )
        val_loss, val_accuracy = validate_on_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )
        val_accuracies.append(val_accuracy)
        if max(val_accuracies) == val_accuracy:
            torch.save(model, f"model_checkpoints/ViT_MNIST_Finetune-v{epoch}.pth")


if __name__ == "__main__":
    wandb_init()
    main()
    wandb.finish()
