from datetime import timedelta
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


@contextmanager
def dist_context(backend: str = "nccl", timeout: int = 10):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout),
        device_id=local_rank,
    )

    torch.cuda.set_device(local_rank)

    try:
        yield
    finally:
        dist.destroy_process_group()


def is_main():
    return get_rank() == 0


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
