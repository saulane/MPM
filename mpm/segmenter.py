from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def apply_mpm_to_segmenter(model: torch.nn.Module, mpm_layers: Sequence[int] = (2, 5)):
    """Enable MPM on a Segmenter checkpoint loaded from this repository."""
    if not hasattr(model, "encoder"):
        raise TypeError("expected a Segmenter model with an encoder")
    model.encoder.global_merge = True
    model.encoder.mpm_layers = list(mpm_layers)
    return model


def load_segmenter(
    checkpoint: str | Path,
    mpm_layers: Sequence[int] | None = (2, 5),
    map_location: str | torch.device = "cpu",
):
    """Load a Segmenter checkpoint and optionally enable MPM."""
    from segm.model.factory import load_model

    model, variant = load_model(str(checkpoint), map_location=map_location)
    if mpm_layers is not None:
        apply_mpm_to_segmenter(model, mpm_layers=mpm_layers)
    return model, variant

