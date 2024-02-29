import torch.distributed as dist
import torch


def is_dist_available_and_initialized() -> bool:
    if not dist.is_available() and not dist.is_initialized():
        return False

    return True


def save_on_master(*args, **kwargs) -> None:
    if is_main_process():
        torch.save(*args, **kwargs)


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0

    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0
