import os
import hostlist
from pathlib import Path
import torch
import torch.distributed as dist

import segm.utils.torch as ptu
import subprocess

def _first_hostname():
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        # Dernier recours : adresse locale
        return "127.0.0.1"
    out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
    return out.split()[0].decode("utf-8")

def init_process(backend: str = "nccl"):
    # 1) MASTER_ADDR / MASTER_PORT
    master_addr = os.environ.get("MASTER_ADDR") or _first_hostname()
    os.environ["MASTER_ADDR"] = master_addr

    if "MASTER_PORT" not in os.environ:
        # Port stable et (quasi) unique par job
        job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        os.environ["MASTER_PORT"] = str(12000 + (job_id % 20000))  # not 29500

    # 2) RANK / WORLD_SIZE / LOCAL_RANK
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]


    # 4) Device par local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Debug clair
    print(
        f"[init] MASTER_ADDR={os.environ['MASTER_ADDR']} "
        f"MASTER_PORT={os.environ['MASTER_PORT']} "
        f"RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={local_rank} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}",
        flush=True,
    )

    # 5) Init
    dist.init_process_group(init_method="env://", backend=backend, world_size=int(os.environ.get('WORLD_SIZE', '0')), rank=int(os.environ.get('RANK','0')))

    # 6) Barrière + logs sobres
    dist.barrier()
    rank = dist.get_rank()
    if rank == 0:
        print("All processes are connected.", flush=True)

def silence_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def sync_model(sync_dir, model):
    # https://github.com/ylabbe/cosypose/blob/master/cosypose/utils/distributed.py
    sync_path = Path(sync_dir).resolve() / "sync_model.pkl"
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        torch.save(model.state_dict(), sync_path)
    if ptu.distributed:
        dist.barrier()
        if ptu.dist_rank > 0:
            model.load_state_dict(torch.load(sync_path))
        dist.barrier()
        if ptu.dist_rank == 0 and ptu.world_size > 1:
            sync_path.unlink()
    return model


def barrier():
    dist.barrier()


def destroy_process():
    dist.destroy_process_group()
