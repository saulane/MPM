from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from segm.model.vit import VisionTransformer
from segm.model.utils import checkpoint_filter_fn
from segm.model.decoder import DecoderLinear
from segm.model.decoder import MaskTransformer
from segm.model.segmenter import Segmenter
import segm.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg, load_pretrained_backbone=True):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    default_cfg = dict(
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )
    if backbone in default_cfgs and isinstance(default_cfgs[backbone], dict):
        default_cfg.update(default_cfgs[backbone])

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)
    if load_pretrained_backbone:
        if backbone == "vit_base_patch8_384":
            path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
            state_dict = torch.load(path, map_location="cpu")
            filtered_dict = checkpoint_filter_fn(state_dict, model)
            model.load_state_dict(filtered_dict, strict=True)
        elif "deit" in backbone:
            load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
        else:
            load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg, load_pretrained_backbone=True):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    # Route Mask2Former models via a separate constructor
    if decoder_cfg.get("name") == "mask2former":
        swin_cfg = model_cfg.get("swin", {})
        px_cfg = decoder_cfg.get("pixel_decoder", {})
        tf_cfg = decoder_cfg.get("transformer", {})
        model = Mask2FormerSegmenter(
            n_cls=model_cfg["n_cls"],
            swin_cfg=swin_cfg,
            pixel_decoder_cfg=px_cfg,
            transformer_cfg=tf_cfg,
        )
        return model

    print("creating vit")
    encoder = create_vit(model_cfg, load_pretrained_backbone=load_pretrained_backbone)
    print("creating decoder")
    decoder = create_decoder(encoder, decoder_cfg)
    print("creating segmenter")
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def load_model(model_path, map_location=None):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    # Build architecture without fetching external backbone weights.
    # The local checkpoint loaded below provides all parameters.
    model = create_segmenter(net_kwargs, load_pretrained_backbone=False)
    if map_location is None:
        map_location = ptu.device
    data = torch.load(model_path, map_location=map_location)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)
    return model, variant
