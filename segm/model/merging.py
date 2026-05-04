"""MPM compatibility functions for the Segmenter fork."""

from __future__ import annotations

import torch

from mpm.merge import (
    compose_merge_maps,
    fast_global_merge_mnn,
    mutual_pair_merge,
    reconstruct_tokens,
)

__all__ = [
    "compose_merge_maps",
    "fast_global_merge_mnn",
    "fast_merge",
    "mutual_pair_merge",
    "reconstruct_tokens",
]


def fast_merge(
    inputs: torch.Tensor,
    group_ids: torch.Tensor,
    group_sizes: torch.Tensor,
) -> torch.Tensor:
    """Group-wise mean reduction used by the original Segmenter data path."""
    batch, length, dim = inputs.shape
    groups = group_sizes.shape[1]
    idx = group_ids.to(device=inputs.device, dtype=torch.long, non_blocking=True)
    sizes = group_sizes.to(device=inputs.device, dtype=inputs.dtype, non_blocking=True)

    out = inputs.new_zeros((batch, groups, dim), dtype=torch.float32)
    out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, dim), inputs.float())
    out = out / sizes.clamp(min=1).float()
    return out.to(dtype=inputs.dtype)

