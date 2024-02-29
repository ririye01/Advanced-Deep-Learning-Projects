import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from vit_model import ViT

def main():
    # TODO: 1- Load Imagenet as PyTorch tensors

    train_dataloader = None # TODO: Create torch dataloader
    test_dataloader = None # TODO: Create torch dataloader
    
    # 2- Define the model and training option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    vit_model = None # TODO: Create the ViT here.
    N_EPOCHS = 5
    LR = 1e-6

    # 3- Training loop
    optimizer = Adam(vit_model.parameters(), lr=LR)
    loss_fn = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch: {epoch + 1} in training", leave=False):
            x, y = batch
            x, y =  x.to(device), y.to(device)
            y_hat = vit_model(x)
            loss = loss_fn(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tqdm.write(f"Epoch {epoch + 1} / {N_EPOCHS} loss: {train_loss:.2f}")

    # 4- Testing loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0
        for batch in tqdm(test_dataloader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = vit_model(x)
            loss = loss_fn(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_dataloader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        
        #
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}")


if __name__ == '__main__':
    main()