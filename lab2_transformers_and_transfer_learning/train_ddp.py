import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.datasets import ImageNet


import os

from VGG_Net import VGG_Net


def init_distributed() -> None:
    """
    Sets up distributed trining ENV variables and backend.
    """
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


def _fetch_train_batch():
    pass

def train() -> VGG_Net:
    for batch in batches:
        _fetch_train_batch()
    return VGG_Net()



def main() -> None:



if __name__ == "__main__":
    init_distributed()
    


    main()
