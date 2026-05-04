import torch
import numpy as np
import torch.distributed as dist
import segm.utils.torch as ptu

import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil
from mmseg.core import mean_iou
from typing import Optional, List, Tuple, Dict

"""
ImageNet classifcation accuracy
"""


def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


"""
Segmentation mean IoU
based on collect_results_cpu
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200
"""


def gather_data(seg_pred, tmp_dir=None):
    """
    distributed data gathering
    prediction and ground truth are stored in a common tmp directory
    and loaded on the master node to compute metrics
    """
    if tmp_dir is None:
        tmpprefix = os.path.expandvars("$DATASET/temp")
    else:
        tmpprefix = os.path.expandvars(tmp_dir)
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device=ptu.device)
    if ptu.dist_rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=tmpprefix)
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=ptu.device
        )
        dir_tensor[: len(tmpdir)] = tmpdir
    # broadcast tmpdir from 0 to to the other nodes
    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    tmpdir = Path(tmpdir)
    """
    Save results in temp file and load them on main process
    """
    tmp_file = tmpdir / f"part_{ptu.dist_rank}.pkl"
    pkl.dump(seg_pred, open(tmp_file, "wb"))
    dist.barrier()
    seg_pred = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            part_seg_pred = pkl.load(open(tmpdir / f"part_{i}.pkl", "rb"))
            seg_pred.update(part_seg_pred)
        shutil.rmtree(tmpdir)
    return seg_pred



def boundary_miou(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    num_classes: int,
    ignore_index: int = -1,
    dilation: int = 1,
    connectivity: int = 8,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute boundary mean IoU (mIoU) between predicted and ground-truth segmentation maps.

    Definition (matching-based):
      1) Extract thin 1-pixel boundaries for each class in pred/gt *restricted to the valid region*.
         A valid boundary pixel must have at least one neighbor WITHIN the valid region whose label differs.
      2) With Chebyshev tolerance `dilation` (0 = exact), compute symmetric matches:
           match_pred = boundary_pred & dilate(boundary_gt, dilation)
           match_gt   = boundary_gt   & dilate(boundary_pred, dilation)
         intersection = 0.5 * (sum(match_pred) + sum(match_gt))
         union        = sum(boundary_pred) + sum(boundary_gt) - intersection
      3) Per-class IoU is intersection/union; mIoU is the mean over classes with union>0.

    Args:
        preds: list of HxW integer arrays with predicted class IDs.
        gts:   list of HxW integer arrays with ground-truth class IDs.
        num_classes: number of evaluated classes (IDs 0..num_classes-1).
        ignore_index: label in GT to ignore (excluded from evaluation).
        dilation: non-negative integer; Chebyshev tolerance for matching (default 1).
        connectivity: 4 or 8 neighborhood for boundary extraction/dilation (default 8).

    Returns:
        (miou, per_class_iou)
    """
    assert len(preds) == len(gts), "preds and gts must have the same length"
    assert connectivity in (4, 8), "connectivity must be 4 or 8"
    assert dilation >= 0, "dilation must be >= 0"

    # Neighbor offsets
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:  # 8-connected
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),            (0, 1),
                   (1, -1),  (1, 0),   (1, 1)]

    def _dilate_bool(mask: np.ndarray, iters: int) -> np.ndarray:
        """Chebyshev-radius dilation using neighborhood OR, `iters` times."""
        if iters == 0:
            return mask
        h, w = mask.shape
        cur = mask
        for _ in range(iters):
            pad = np.pad(cur, 1, mode="constant", constant_values=False)
            acc = cur.copy()
            for dy, dx in offsets:
                acc |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            cur = acc
        return cur

    def _boundary_from_binary_in_valid(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Extract a thin boundary of a binary mask (True=foreground) **within the valid region**.
        A pixel (inside `valid`) is boundary if at least one neighbor (also inside `valid`)
        has a different mask value. Neighbors outside `valid` are ignored.
        """
        h, w = mask.shape
        pad_m = np.pad(mask, 1, mode="constant", constant_values=False)
        pad_v = np.pad(valid, 1, mode="constant", constant_values=False)

        # Start from "assume eroded"; a pixel stays eroded only if all *valid* neighbors are True.
        eroded = mask & valid  # erosion seed is the foreground inside valid
        for dy, dx in offsets:
            neigh_m = pad_m[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            neigh_v = pad_v[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            # If neighbor is valid, require it to be True; if not valid, it imposes no constraint.
            eroded &= np.where(neigh_v, neigh_m, True)

        # Boundary pixels are the foreground-in-valid that are not (valid-neighborhood) eroded.
        return (mask & valid) & ~eroded

    # Accumulators across the dataset
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for pred, gt in zip(preds, gts):
        if pred.shape != gt.shape:
            raise ValueError("Each pred/gt pair must have the same shape")

        valid = (gt != ignore_index)

        for c in range(num_classes):
            # Build class masks (do NOT bake 'valid' in here; we pass it to boundary extractor)
            gt_c   = (gt   == c)
            pred_c = (pred == c)

            # Skip early if neither side contains the class inside the valid region
            if not ((gt_c & valid).any() or (pred_c & valid).any()):
                continue

            # Thin boundaries restricted to valid region
            b_gt = _boundary_from_binary_in_valid(gt_c, valid)
            b_pr = _boundary_from_binary_in_valid(pred_c, valid)

            # If both boundaries are empty, skip this image for this class
            if not (b_gt.any() or b_pr.any()):
                continue

            # Use dilation ONLY to determine matches; do not thicken sets for IoU
            b_gt_d = _dilate_bool(b_gt, dilation) if dilation > 0 else b_gt
            b_pr_d = _dilate_bool(b_pr, dilation) if dilation > 0 else b_pr

            match_pr = np.logical_and(b_pr, b_gt_d).sum(dtype=np.float64)
            match_gt = np.logical_and(b_gt, b_pr_d).sum(dtype=np.float64)

            inter_c = 0.5 * (match_pr + match_gt)
            union_c = b_pr.sum(dtype=np.float64) + b_gt.sum(dtype=np.float64) - inter_c

            if union_c > 0:
                inter[c] += inter_c
                union[c] += union_c
            # else: if union is 0 here, the class had only degenerate "no-boundary" regions; ignore

    # Per-class IoU and mIoU over classes with non-zero union
    per_class_iou: Dict[int, float] = {c: float(inter[c] / union[c])
                                       for c in range(num_classes) if union[c] > 0}
    miou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else float("nan")
    return miou, per_class_iou


def compute_metrics(
    seg_pred,
    seg_gt,
    n_cls,
    ignore_index=None,
    ret_cat_iou=False,
    tmp_dir=None,
    distributed=False,
):
    ret_metrics_mean = torch.zeros(3, dtype=torch.float32, device=ptu.device)
    if ptu.dist_rank == 0:
        list_seg_pred = []
        list_seg_gt = []
        keys = sorted(seg_pred.keys())
        for k in keys:
            list_seg_pred.append(np.asarray(seg_pred[k]))
            list_seg_gt.append(np.asarray(seg_gt[k]))
        ret_metrics = mean_iou(
            results=list_seg_pred,
            gt_seg_maps=list_seg_gt,
            num_classes=n_cls,
            ignore_index=ignore_index,
        )
        ret_metrics = [ret_metrics["aAcc"], ret_metrics["Acc"], ret_metrics["IoU"]]
        ret_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(ret_metric.astype(np.float32)) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=torch.float32,
            device=ptu.device,
        )
        cat_iou = ret_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)

    if ret_cat_iou and ptu.dist_rank == 0:
        ret["cat_iou"] = cat_iou
    return ret
