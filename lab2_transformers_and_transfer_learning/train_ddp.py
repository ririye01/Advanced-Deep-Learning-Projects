import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple, List
from torch import Tensor

import argparse

from ImageNetDataset import ImageNetDataset
from VGG_Net import VGG_Net


# Setup for distributed training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# Cleanup for distributed training
def cleanup():
    dist.destroy_process_group()


def train(rank: int, world_size: int, epochs: int, root_dir: str, batch_size: int = 32):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageNetDataset(root_dir=root_dir, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler
    )

    model = models.vgg19(pretrained=False, num_classes=1000).cuda(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.cuda(rank), labels.cuda(rank)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Checkpointing and logging
        if rank == 0:  # Save checkpoint and log metrics only from the master process
            torch.save(model.module.state_dict(), f"vgg19_imagenet_epoch_{epoch}.pth")
            with open("training_log.txt", "a") as log_file:
                log_file.write(
                    f"Epoch: {epoch}, Loss: {running_loss / len(dataloader)}\n"
                )

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    root_dir = "~/../tdohm/Data/train"
    epochs = 10
    batch_size = 32

    train(args.local_rank, world_size, epochs, root_dir, batch_size)
