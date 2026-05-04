import os
import torch


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size

    gpu_id = int(os.environ.get("SLURM_LOCALID", 0)) if "SLURM_LOCALID" in os.environ else int(os.environ.get("LOCAL_RANK", 0))
    dist_rank = int(os.environ.get("SLURM_PROCID", 0)) if "SLURM_PROCID" in os.environ else int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1)) if "SLURM_NTASKS" in os.environ else int(os.environ.get("WORLD_SIZE", 1))

    distributed = world_size > 1
    use_gpu = mode
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    if use_gpu:
        torch.cuda.set_device(gpu_id)
    torch.backends.cudnn.benchmark = True
