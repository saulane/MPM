from __future__ import annotations

import types
from typing import Iterable, Sequence

import torch

from mpm.merge import compose_merge_maps, mutual_pair_merge, reconstruct_tokens


def _parse_layers(layers: Sequence[int] | str | None) -> tuple[int, ...]:
    if layers is None:
        return (2, 5)
    if isinstance(layers, str):
        return tuple(sorted({int(x) for x in layers.replace(",", " ").split()}))
    return tuple(sorted({int(x) for x in layers}))


def _infer_prefix_tokens(model: torch.nn.Module) -> int:
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)
    count = 0
    if getattr(model, "cls_token", None) is not None:
        count += 1
    if getattr(model, "dist_token", None) is not None:
        count += 1
    if hasattr(model, "reg_token") and getattr(model, "reg_token") is not None:
        count += int(model.reg_token.shape[1])
    return count


def _cat_prefix_tokens(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    if getattr(model, "cls_token", None) is None:
        return x
    cls = model.cls_token.expand(batch, -1, -1)
    dist = getattr(model, "dist_token", None)
    if dist is not None:
        dist = dist.expand(batch, -1, -1)
        return torch.cat((cls, dist, x), dim=1)
    return torch.cat((cls, x), dim=1)


def _pos_embed(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_pos_embed"):
        return model._pos_embed(x)
    x = _cat_prefix_tokens(model, x)
    pos_embed = getattr(model, "pos_embed", None)
    if pos_embed is not None:
        x = x + pos_embed
    return x


def _apply_pre_blocks(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "patch_drop"):
        x = model.patch_drop(x)
    if hasattr(model, "pos_drop"):
        x = model.pos_drop(x)
    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)
    return x


def _forward_features_mpm(self: torch.nn.Module, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    layers = self._mpm_info["layers"]
    prefix_tokens = self._mpm_info["num_prefix_tokens"]
    trace = self._mpm_info["trace"]

    x = self.patch_embed(x)
    if x.ndim == 4:
        x = x.flatten(1, 2)
    x = _pos_embed(self, x)
    x = _apply_pre_blocks(self, x)

    batch = x.shape[0]
    original_token_count = x.shape[1] - prefix_tokens
    samples = [x[i : i + 1] for i in range(batch)]
    merge_maps: list[torch.Tensor | None] = [None for _ in range(batch)]
    layer_counts: list[list[torch.Tensor]] = [[] for _ in range(batch)]

    for layer_idx, block in enumerate(self.blocks):
        next_samples = []
        for sample_idx, sample in enumerate(samples):
            if layer_idx in layers:
                prefix = sample[:, :prefix_tokens]
                image_tokens = sample[:, prefix_tokens:]
                merged = mutual_pair_merge(image_tokens)
                if merge_maps[sample_idx] is None:
                    merge_maps[sample_idx] = merged.ids
                else:
                    merge_maps[sample_idx] = compose_merge_maps(
                        merge_maps[sample_idx],
                        merged.ids,
                    )
                layer_counts[sample_idx].append(merged.counts.detach().cpu())
                sample = torch.cat((prefix, merged.tokens), dim=1)
            next_samples.append(block(sample))
        samples = next_samples

    restored = []
    for sample_idx, sample in enumerate(samples):
        ids = merge_maps[sample_idx]
        if ids is not None:
            prefix = sample[:, :prefix_tokens]
            image_tokens = reconstruct_tokens(sample[:, prefix_tokens:], ids)
            if image_tokens.shape[1] != original_token_count:
                raise RuntimeError("MPM reconstruction did not restore the original token count")
            sample = torch.cat((prefix, image_tokens), dim=1)
        restored.append(sample)

    x = torch.cat(restored, dim=0)
    if hasattr(self, "norm"):
        x = self.norm(x)
    if hasattr(self, "forward_head"):
        return x
    if hasattr(self, "fc_norm") and getattr(self, "global_pool", "") == "avg":
        return self.fc_norm(x[:, prefix_tokens:].mean(dim=1))
    if hasattr(self, "pre_logits"):
        x = self.pre_logits(x[:, 0])

    if trace:
        self._mpm_info["merge_maps"] = merge_maps
        self._mpm_info["counts"] = layer_counts

    return x


def apply_patch(
    model: torch.nn.Module,
    mpm_layers: Sequence[int] | str | None = (2, 5),
    trace: bool = False,
    num_prefix_tokens: int | str = "auto",
) -> torch.nn.Module:
    """Add MPM to a timm ViT or a Segmenter model in place.

    For timm ViTs, MPM is inserted before the requested encoder blocks and the
    image-token grid is reconstructed before the original head. For Segmenter
    checkpoints from this repository, the encoder already contains the MPM hook,
    so this function only sets ``global_merge`` and ``mpm_layers``.
    """
    layers = _parse_layers(mpm_layers)

    if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
        model.encoder.global_merge = True
        model.encoder.mpm_layers = list(layers)
        return model

    if not hasattr(model, "blocks") or not hasattr(model, "patch_embed"):
        raise TypeError("apply_patch expects a timm VisionTransformer-like model")

    if num_prefix_tokens == "auto":
        prefix_count = _infer_prefix_tokens(model)
    else:
        prefix_count = int(num_prefix_tokens)

    model._mpm_info = {
        "layers": layers,
        "num_prefix_tokens": prefix_count,
        "trace": bool(trace),
        "merge_maps": None,
        "counts": None,
    }
    model.forward_features = types.MethodType(_forward_features_mpm, model)
    return model
