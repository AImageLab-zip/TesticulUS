"""
Helpers for distributed training.
"""

import io
import os
import socket
from venv import logger

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist
import torch
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # comm = MPI.COMM_WORLD
    port = str(find_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    print(f"using port:{port}")
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
        hostname = "127.0.0.1"
    dist.init_process_group(backend=backend, init_method="env://")
    # comm = dist.group.WORLD

    # os.environ["MASTER_ADDR"] = dist.broadcast(hostname, src=0)

    # port = dist.broadcast(_find_free_port(), src=0)
    # os.environ["MASTER_PORT"] = str(port)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    # if MPI.COMM_WORLD.Get_rank() == 0:
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        dist.broadcast(num_chunks, src=dist.get_group_rank())
        # MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            # MPI.COMM_WORLD.bcast(data[i: i + chunk_size])
            dist.broadcast(data[i: i + chunk_size], group_src=dist.get_group_rank())

    else:
        # num_chunks = MPI.COMM_WORLD.bcast(None)
        num_chunks = dist.broadcast(None, group_src=dist.get_group_rank())
        data = bytes()
        for _ in range(num_chunks):
            data += dist.broadcast(None, group_src=dist.get_group_rank())

    return th.load(io.BytesIO(data), **kwargs)

# Optional: Add a wrapper for more convenient loading


def distributed_load(path, device=None, **kwargs):
    """
    Convenience wrapper for distributed checkpoint loading.
    Initializes process group if not already initialized.

    Args:
        path (str): Path to the checkpoint file
        device (torch.device, optional): Target device for loading
        **kwargs: Additional arguments to pass to torch.load()

    Returns:
        Loaded state dictionary
    """
    # Initialize process group if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    return load_state_dict(path, device, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
