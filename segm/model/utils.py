import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import trunc_normal_

import segm.utils.torch as ptu
from torch.profiler import profile, ProfilerActivity
import numpy as np

import time

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    _, _, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    if H <= ws:
        h_anchors = [0]
    else:
        h_anchors = list(range(0, H - ws, window_stride)) + [H - ws]
    if W <= ws:
        w_anchors = [0]
    else:
        w_anchors = list(range(0, W - ws, window_stride)) + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws, wa : wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    result = F.softmax(logit, 0)
    return result


@torch.inference_mode()
def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
    return_flops=False
):
    C = model.n_cls
    seg_map = torch.zeros((C, ori_shape[0], ori_shape[1]), device=ptu.device)
    all_flops = []
    all_fps = []
    for im, im_metas in zip(ims, ims_metas):
        im = im.to(ptu.device, non_blocking=True)
        im = resize(im, window_size)
        flip = im_metas["flip"]
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.cat(windows.pop("crop"), dim=0)
        B = int(crops.shape[0])
        WB = batch_size
        seg_maps = torch.zeros((B, C, window_size, window_size), device=im.device)
        for i in range(0, B, WB):
            if return_flops:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_flops=True,
                ) as prof:
                    seg_maps[i : i + WB] = model(crops[i : i + WB])
                s = time.time()
                model(crops[i : i + WB])
                e = time.time()
                all_fps.append(crops[i : i + WB].size(0) / (e - s))
                events = prof.key_averages(group_by_input_shape=True)
                torch_flops = sum(event.flops for event in events) / crops.size(0)
                all_flops.append(torch_flops)
            else:
                seg_maps[i : i + WB] = model(crops[i : i + WB])
        windows["seg_maps"] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_map += im_seg_map
    seg_map /= len(ims)
    if return_flops:
        return seg_map, np.mean(all_flops), np.mean(all_fps)
    return seg_map


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()
