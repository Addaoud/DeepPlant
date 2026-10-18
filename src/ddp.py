import torch.distributed as dist
import os
from torch.cuda import set_device


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = (
        "localhost"  # Change this to the master node's IP address if using multiple machines
    )
    os.environ["MASTER_PORT"] = "12355"  # Pick a free port on the master node
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # A workaround if dist.barrier not working properly
    set_device(rank)


def is_dist_avail_and_initialized():
    """Code from https://github.com/pytorch/vision/blob/main/references/classification/utils.py"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """Code from https://github.com/pytorch/vision/blob/main/references/classification/utils.py"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Code from https://github.com/pytorch/vision/blob/main/references/classification/utils.py"""
    return get_rank() == 0


def cleanup():
    dist.destroy_process_group()
