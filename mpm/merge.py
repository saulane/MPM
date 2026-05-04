from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MergeResult:
    """Outputs of one MPM call."""

    tokens: torch.Tensor
    ids: torch.Tensor
    counts: torch.Tensor
    valid_mask: torch.Tensor


def mutual_pair_merge(
    tokens: torch.Tensor,
    region_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> MergeResult:
    """Merge mutual nearest-neighbor token pairs in cosine space.

    Parameters
    ----------
    tokens:
        Tensor of shape ``(B, N, D)``.
    region_ids:
        Optional integer tensor of shape ``(B, N)``. When provided, tokens only
        pair with tokens from the same region.
    valid_mask:
        Optional boolean tensor of shape ``(B, N)``. Invalid padded tokens are
        ignored and mapped to representative id 0.

    Returns
    -------
    MergeResult
        Merged tokens padded to the largest number of representatives in the
        batch, the original-token to merged-token map, representative counts,
        and the merged-token validity mask.
    """
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B, N, D), got {tuple(tokens.shape)}")

    batch, length, dim = tokens.shape
    device = tokens.device
    dtype = tokens.dtype

    if length == 0:
        raise ValueError("MPM requires at least one token")

    if valid_mask is None:
        valid_mask = torch.ones(batch, length, device=device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        if valid_mask.shape != (batch, length):
            raise ValueError(
                f"valid_mask must have shape {(batch, length)}, got {tuple(valid_mask.shape)}"
            )

    if region_ids is not None:
        region_ids = region_ids.to(device=device)
        if region_ids.shape != (batch, length):
            raise ValueError(
                f"region_ids must have shape {(batch, length)}, got {tuple(region_ids.shape)}"
            )

    index = torch.arange(length, device=device)[None, :]
    normalized = F.normalize(tokens, dim=-1)
    sim = normalized @ normalized.transpose(1, 2)
    sim = sim.masked_fill(
        torch.eye(length, device=device, dtype=torch.bool)[None],
        float("-inf"),
    )

    if region_ids is not None:
        same_region = region_ids[:, :, None].eq(region_ids[:, None, :])
        sim = sim.masked_fill(~same_region, float("-inf"))

    valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
    sim = sim.masked_fill(~valid_pairs, float("-inf"))

    has_candidate = torch.isfinite(sim).any(dim=-1)
    best = sim.argmax(dim=-1)
    best = torch.where(has_candidate, best, index.expand(batch, -1))

    best_of_best = torch.gather(best, 1, best)
    mutual = (best_of_best == index) & (best != index) & valid_mask
    representatives = (mutual & (index < best)) | (valid_mask & ~mutual)

    rep_anchor = torch.where(mutual, torch.minimum(index, best), index)
    rep_anchor = torch.where(valid_mask, rep_anchor, torch.zeros_like(rep_anchor))

    rep_ids_full = representatives.to(torch.long).cumsum(dim=1) - 1
    rep_ids_full = torch.where(
        representatives,
        rep_ids_full,
        torch.zeros_like(rep_ids_full),
    )
    ids = torch.gather(rep_ids_full, 1, rep_anchor)
    ids = torch.where(valid_mask, ids, torch.zeros_like(ids))

    n_rep = max(int(representatives.sum(dim=1).max().item()), 1)
    merged = tokens.new_zeros((batch, n_rep, dim))
    counts = tokens.new_zeros((batch, n_rep, 1))

    valid_float = valid_mask.unsqueeze(-1).to(dtype)
    merged.scatter_add_(1, ids.unsqueeze(-1).expand(-1, -1, dim), tokens * valid_float)
    counts.scatter_add_(1, ids.unsqueeze(-1), valid_float)
    merged = merged / counts.clamp(min=1)

    merged_valid = counts.squeeze(-1) > 0
    return MergeResult(tokens=merged, ids=ids, counts=counts, valid_mask=merged_valid)


def reconstruct_tokens(merged_tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Copy merged representatives back to the original token grid."""
    if merged_tokens.ndim != 3 or ids.ndim != 2:
        raise ValueError("merged_tokens must be (B, M, D) and ids must be (B, N)")
    if merged_tokens.shape[0] != ids.shape[0]:
        raise ValueError("merged_tokens and ids batch sizes differ")

    dim = merged_tokens.shape[-1]
    gather_ids = ids.to(device=merged_tokens.device, dtype=torch.long)
    return torch.gather(
        merged_tokens,
        1,
        gather_ids.unsqueeze(-1).expand(-1, -1, dim),
    )


def compose_merge_maps(previous_ids: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
    """Compose two merge maps.

    ``previous_ids`` maps the original token grid to an intermediate grid, while
    ``next_ids`` maps that intermediate grid to the next merged grid.
    """
    if previous_ids.ndim != 2 or next_ids.ndim != 2:
        raise ValueError("merge maps must both be rank-2 tensors")
    if previous_ids.shape[0] != next_ids.shape[0]:
        raise ValueError("merge maps must have the same batch size")
    return torch.gather(next_ids, 1, previous_ids.to(next_ids.device))


def fast_global_merge_mnn(
    tokens: torch.Tensor,
    region_ids: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
):
    """Compatibility wrapper for the original Segmenter integration."""
    result = mutual_pair_merge(tokens, region_ids=region_ids, valid_mask=valid_mask)
    return result.tokens, result.ids, result.counts

