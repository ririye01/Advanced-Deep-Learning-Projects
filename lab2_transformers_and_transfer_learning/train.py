import torch
from torch.cuda import is_available

from VGG_Net import VGG_Net


if __name__ == "__main__":
    # Set the device to CUDA if available, metal if Mac, or CPU if we want to train for 10,000 years
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Set model
    model = VGG_Net().to(device)
