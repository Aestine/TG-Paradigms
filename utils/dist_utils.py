"""
Distributed training utilities for SmolVLM-DisTime.
Based on InternVL/DisTime dist_utils.py
"""

import os
import socket
import subprocess
from datetime import timedelta
from typing import Optional

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

# Default timeout for distributed operations
DIST_TIMEOUT = timedelta(minutes=60)


def _find_free_port() -> int:
    """Find a free port on the local machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _is_free_port(port: int) -> bool:
    """Check if a port is free."""
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def init_dist(launcher: str = 'pytorch', backend: str = 'nccl', **kwargs):
    """
    Initialize distributed training environment.
    
    Args:
        launcher: One of 'pytorch', 'slurm', 'mpi'
        backend: Distributed backend ('nccl' or 'gloo')
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend: str, **kwargs):
    """Initialize distributed training with PyTorch launcher (torchrun)."""
    rank = int(os.environ.get('RANK', 0))
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    
    # Try deepspeed init first, fall back to torch.distributed
    try:
        import deepspeed
        deepspeed.init_distributed(dist_backend=backend, timeout=DIST_TIMEOUT)
    except ImportError:
        dist.init_process_group(backend=backend, timeout=DIST_TIMEOUT, **kwargs)


def _init_dist_slurm(backend: str, port: Optional[int] = None):
    """
    Initialize distributed training with SLURM launcher.
    
    Args:
        backend: Distributed backend
        port: Master port (will use MASTER_PORT env var or 29500 if not specified)
    """
    proc_id = int(os.environ.get('SLURM_PROCID', 0))
    ntasks = int(os.environ.get('SLURM_NTASKS', 1))
    node_list = os.environ.get('SLURM_NODELIST', '')
    
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    
    # Get master address
    if '[' in node_list:
        # Parse SLURM node list format like "node[1-4]"
        result = subprocess.check_output(
            f'scontrol show hostname {node_list} | head -n1',
            shell=True
        )
        addr = result.decode('utf-8').strip()
    else:
        addr = node_list
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = addr
    
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    
    # Initialize distributed
    try:
        import deepspeed
        deepspeed.init_distributed(dist_backend=backend, timeout=DIST_TIMEOUT)
    except ImportError:
        dist.init_process_group(backend=backend, timeout=DIST_TIMEOUT)


def _init_dist_mpi(backend: str, **kwargs):
    """Initialize distributed training with MPI launcher."""
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    
    os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK', '0')
    
    dist.init_process_group(backend=backend, timeout=DIST_TIMEOUT, **kwargs)


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank (GPU index on current node)."""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce tensor across all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source to all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


def gather_tensor(tensor: torch.Tensor, dst: int = 0):
    """Gather tensors from all processes to destination."""
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.gather(tensor, gathered if get_rank() == dst else None, dst=dst)
    return gathered


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
