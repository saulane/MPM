"""
Vision Transformer (ViT) building blocks and utilities adapted for
semantic segmentation and token merging.

Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import init_weights, resize_pos_embed
from segm.model.blocks import Block
from segm.model.merging import compose_merge_maps, fast_global_merge_mnn

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights

from typing import Optional, Tuple
import time

import math

class PatchEmbedding(nn.Module):
    """Image to patch embeddings via a strided convolution.

    Parameters
    ----------
    image_size : tuple[int, int]
        Input image size as ``(H, W)``.
    patch_size : int
        Side length of a square patch. Must divide both ``H`` and ``W``.
    embed_dim : int
        Output channel dimension of the patch embeddings.
    channels : int
        Number of input image channels (e.g. 3 for RGB).
    """
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        """Convert an image batch to a sequence of patch embeddings.

        Parameters
        ----------
        im : torch.Tensor
            Tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``(B, N, D)`` where
            ``N = (H/ps) * (W/ps)`` and ``D = embed_dim``.
        """
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone with optional token merging.

    Parameters
    ----------
    image_size : tuple[int, int]
        Input image size used to build the patch embedding grid.
    patch_size : int
        Patch size for the patch embedding layer.
    n_layers : int
        Number of transformer encoder blocks.
    d_model : int
        Token/channel dimension (hidden size).
    d_ff : int
        MLP feedforward dimension.
    n_heads : int
        Number of attention heads per block.
    n_cls : int
        Number of classes for the classification head.
    dropout : float
        Dropout rate applied after positional embedding.
    drop_path_rate : float
        Stochastic depth rate across blocks.
    distilled : bool
        Whether to add a distillation token and auxiliary head.
    channels : int
        Number of input channels.
    flood_fill : bool
        If True, indicates that region-aware operations (e.g., flood-fill
        supervision) may be used; the core model is agnostic but some
        utilities honor this flag.
    """
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
        flood_fill=False,
        global_merge=False
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        self.flood_fill= flood_fill
        self.global_merge = global_merge
        # Optional list of layer indices to apply MPM. The default follows the
        # paper: insert before blocks 2 and 5 with 0-based indexing.
        self.mpm_layers = [2, 5]
        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()
        self.ids_mid = None

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, return_features=False, ids=None, sizes=None):
        """Forward pass through ViT with optional token merging.

        Parameters
        ----------
        im : torch.Tensor
            Input images of shape ``(B, C, H, W)``.
        return_features : bool
            If True, returns final token features instead of logits.
        ids : torch.Tensor | None
            Optional ``(B, N)`` mapping from original tokens to locally merged
            groups. If given, a local merge is applied at input and undone at
            the end of the network.
        sizes : torch.Tensor | None
            Optional ``(B, G, 1)`` sizes for each local group used by
            ``fast_merge``.

        Returns
        -------
        torch.Tensor
            If ``return_features``: final token features of shape ``(B, N+extra, D)``.
            Else: classification logits of shape ``(B, n_cls)``.
        """

        B, _, H, W = im.shape
        PS = self.patch_size

        # ---------- embed + extra tokens ----------
        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        num_extra = 1 + self.distilled
        pos = self.pos_embed
        if x.shape[1] != pos.shape[1]:
            pos = resize_pos_embed(pos,
                                   self.patch_embed.grid_size,
                                   (H // PS, W // PS),
                                   num_extra)
        x = x + pos
        x = self.dropout(x)

        D = x.size(-1)
        num_tokens = x.size(1) - num_extra      # before any merge

        # ---------- first (local) merge at input ----------
        if ids is not None:
            extra = x[:, :num_extra, :]
            x_loc = fast_merge(x[:, num_extra:, :], ids, sizes)    # already provided
            x     = torch.cat([extra, x_loc], dim=1)               # (B, N₁+D, D)
            N1    = x_loc.size(1)                                  # will need later
            token_valid_mask = (sizes.squeeze(-1) > 0).to(device=x.device)
        else:
            N1    = num_tokens
            token_valid_mask = torch.ones(B, N1, device=x.device, dtype=torch.bool)

        # -------------- forward through ViT ----------------
        self.ids_mid = None
        backup = None

        for i, blk in enumerate(self.blocks):
            counts = None
            # ---- global merge at selected layers ----
            should_merge = False
            if self.global_merge:
                should_merge = i in self.mpm_layers

            if should_merge:
                extra = x[:, :num_extra, :]                        # CLS (+dist)
                tokens_mid = x[:, num_extra:, :]                   # (B, N₁, D)

                region_ids = None
                tokens_glb, ids_mid_temp, counts = fast_global_merge_mnn(
                    tokens_mid,
                    region_ids=region_ids,
                    valid_mask=token_valid_mask,
                )
                # ids_mids.append(ids_mid_temp)
                if self.ids_mid is None:
                    self.ids_mid = ids_mid_temp
                else:
                    self.ids_mid = compose_merge_maps(self.ids_mid, ids_mid_temp)
                x = torch.cat([extra, tokens_glb], dim=1)
                token_valid_mask = counts.squeeze(-1) > 0
                # for unmerge later


            # ---- unmerge global merge right before last blk ----
            extra_valid = torch.ones(B, num_extra, device=x.device, dtype=torch.bool)
            attn_valid_mask = torch.cat([extra_valid, token_valid_mask], dim=1)
            attn_mask = attn_valid_mask[:, None, None, :]
            x = blk(x, mask=attn_mask)
            if self.ids_mid is not None and i == len(self.blocks)-1:#len(ids_mids):
                extra = x[:, :num_extra, :]
                expanded = torch.gather(
                    x[:, num_extra:, :],
                    1,
                    self.ids_mid.unsqueeze(-1).expand(-1, -1, D)
                )
                x = torch.cat([extra, expanded], dim=1)

            # if i == 1:
            #     import matplotlib.pyplot as plt
            #     sim = x[:,:1,:] @ x[:,1:,:].permute(0,2,1)

            #     plt.imshow(sim[0].reshape(32,32,-1).detach().cpu().numpy())
            #     plt.show()


        # ---------------- head -----------------
        x = self.norm(x)

        if return_features:
            return x

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = (self.head(x) + self.head_dist(x_dist)) / 2
        else:
            x = self.head(x[:, 0])
        return x


    def get_attention_map(self, im, layer_id):
        """Return attention weights of a specific encoder block.

        Parameters
        ----------
        im : torch.Tensor
            Input images of shape ``(B, C, H, W)``.
        layer_id : int
            Index of the transformer block to probe (0-based).

        Returns
        -------
        torch.Tensor
            Attention matrix of shape ``(B, heads, N+extra, N+extra)`` as
            produced by the selected block.
        """
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
