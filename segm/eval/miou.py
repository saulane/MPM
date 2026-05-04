import sys
import click
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import shutil

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from segm.utils import distributed
from segm.utils.logger import MetricLogger
import segm.utils.torch as ptu

from segm.model.factory import load_model
from segm.data.factory import create_dataset
from segm.metrics import gather_data, compute_metrics

from segm.model.utils import inference
from segm.data.utils import seg_to_rgb, rgb_denormalize, IGNORE_LABEL
from segm import config


def blend_im(im, seg, alpha=0.5):
    pil_im = Image.fromarray(im)
    pil_seg = Image.fromarray(seg)
    im_blend = Image.blend(pil_im, pil_seg, alpha).convert("RGB")
    return np.asarray(im_blend)


def save_im(save_dir, save_name, im, seg_pred, seg_gt, colors, blend, normalization):
    seg_rgb = seg_to_rgb(seg_gt[None], colors)
    pred_rgb = seg_to_rgb(seg_pred[None], colors)
    im_unnorm = rgb_denormalize(im, normalization)
    save_dir = Path(save_dir)

    # save images
    im_uint = (im_unnorm.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
    seg_rgb_uint = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
    seg_pred_uint = (255 * pred_rgb.cpu().numpy()).astype(np.uint8)
    for i in range(pred_rgb.shape[0]):
        if blend:
            blend_pred = blend_im(im_uint[i], seg_pred_uint[i])
            blend_gt = blend_im(im_uint[i], seg_rgb_uint[i])
            ims = (im_uint[i], blend_pred, blend_gt)
        else:
            ims = (im_uint[i], seg_pred_uint[i], seg_rgb_uint[i])
        for im, im_dir in zip(
            ims, (save_dir / "input", save_dir / "pred", save_dir / "gt"),
        ):
            pil_out = Image.fromarray(im)
            im_dir.mkdir(exist_ok=True)
            pil_out.save(im_dir / save_name)


def process_batch(
    model,
    batch,
    window_size,
    window_stride,
    window_batch_size,
    return_flops=False,
    return_im=True,
):
    ims = batch["im"]
    ims_metas = batch["im_metas"]
    ori_shape = ims_metas[0]["ori_shape"]
    ori_shape = (ori_shape[0].item(), ori_shape[1].item())
    filename = batch["im_metas"][0]["ori_filename"][0]

    model_without_ddp = model
    if ptu.distributed:
        model_without_ddp = model.module
    if return_flops:
        seg_pred, flops, fps = inference(
            model_without_ddp,
            ims,
            ims_metas,
            ori_shape,
            window_size,
            window_stride,
            window_batch_size,
            return_flops=return_flops
        )
    else:
        seg_pred = inference(
            model_without_ddp,
            ims,
            ims_metas,
            ori_shape,
            window_size,
            window_stride,
            window_batch_size,
            return_flops=return_flops

        )
    seg_pred = seg_pred.argmax(0)
    im = None
    if return_im:
        im = F.interpolate(ims[-1], ori_shape, mode="bilinear").cpu()

    if return_flops:
        return filename, im, seg_pred.cpu(), flops, fps

    return filename, im, seg_pred.cpu()


def eval_dataset(
    model,
    multiscale,
    model_dir,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
    dataset_kwargs,
    profile_flops=False,
):
    db = create_dataset(dataset_kwargs)
    normalization = db.dataset.normalization
    dataset_name = dataset_kwargs["dataset"]
    im_size = dataset_kwargs["image_size"]
    cat_names = db.base_dataset.names
    n_cls = db.unwrapped.n_cls
    if multiscale:
        db.dataset.set_multiscale_mode()

    logger = MetricLogger(delimiter="  ")
    header = ""
    print_freq = 50

    ims = {}
    seg_pred_maps = {}
    last_colors = None
    idx = 0
    all_flops = []
    all_fps = []
    for batch in logger.log_every(db, print_freq, header):
        if save_images:
            last_colors = batch["colors"]

        do_profile = bool(profile_flops and idx < 20)
        if do_profile:
            filename, im, seg_pred, flops, fps = process_batch(
                model,
                batch,
                window_size,
                window_stride,
                window_batch_size,
                return_flops=True,
                return_im=save_images,
            )
            all_flops.append(flops)
            all_fps.append(fps)
        else:
            filename, im, seg_pred = process_batch(
                model,
                batch,
                window_size,
                window_stride,
                window_batch_size,
                return_flops=False,
                return_im=save_images,
            )
        if save_images:
            ims[filename] = im
        seg_pred_maps[filename] = seg_pred
        idx += 1
        if idx > len(db) * frac_dataset:
            break

    seg_gt_maps = db.dataset.get_gt_seg_maps()
    if save_images:
        if last_colors is None:
            raise RuntimeError("No validation samples were processed, cannot save images.")
        save_dir = model_dir / "images"
        if ptu.dist_rank == 0:
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir()
        if ptu.distributed:
            torch.distributed.barrier()

        for name in sorted(ims):
            instance_dir = save_dir
            filename = name

            if dataset_name == "cityscapes":
                filename_list = name.split("/")
                instance_dir = instance_dir / filename_list[0]
                filename = filename_list[-1]
                if not instance_dir.exists():
                    instance_dir.mkdir()

            save_im(
                instance_dir,
                filename,
                ims[name],
                seg_pred_maps[name],
                torch.tensor(seg_gt_maps[name]),
                last_colors,
                blend,
                normalization,
            )
        if ptu.dist_rank == 0:
            shutil.make_archive(save_dir, "zip", save_dir)
            # shutil.rmtree(save_dir)
            print(f"Saved eval images in {save_dir}.zip")

    if ptu.distributed:
        torch.distributed.barrier()
        seg_pred_maps = gather_data(seg_pred_maps)

    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )

    if ptu.dist_rank == 0:
        scores["inference"] = "single_scale" if not multiscale else "multi_scale"
        scores["mean_flops"] = np.mean(all_flops) if len(all_flops) > 0 else 0
        scores["mean_fps"] = np.mean(all_fps) if len(all_fps) > 0 else 0
        suffix = "ss" if not multiscale else "ms"
        scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
        for k, v in scores.items():
            if k != "cat_iou" and k != "inference":
                if hasattr(v, "item"):
                    scores[k] = v.item()
                else:
                    scores[k] = float(v)
            if k != "cat_iou":
                print(f"{k}: {scores[k]}")
        scores_str = yaml.dump(scores)
        with open(model_dir / f"scores_{suffix}.yml", "w") as f:
            f.write(scores_str)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--im-size", default=None, type=int)
@click.option("--multiscale/--singlescale", default=False, is_flag=True)
@click.option("--blend/--no-blend", default=True, is_flag=True)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--window-batch-size", default=4, type=int)
@click.option("--use-gpu", default=True, type=bool)
@click.option("--use-mps", default=False, type=bool)
@click.option("--save-images/--no-save-images", default=False, is_flag=True)
@click.option("--frac_dataset", "--frac-dataset", default=1.0, type=float)
@click.option("--global-merge/--no-global-merge", default=False, is_flag=True, help="Enable MPM with the default paper schedule.")
@click.option(
    "--profile-flops/--no-profile-flops",
    default=False,
    is_flag=True,
    help="Profile FLOPs/FPS for first 20 validation items (slower).",
)
@click.option(
    "--mpm_layers",
    "--mpm-layers",
    type=str,
    multiple=True,
    default=(),
    help=(
        "Encoder block indices for global merging. Accepts repeated options "
        "(e.g., --mpm-layers 0 --mpm-layers 1) or comma-separated values "
        "(e.g., --mpm-layers 0,1,2)."
    ),
)
def main(
    model_path,
    dataset_name,
    im_size,
    multiscale,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    use_gpu,
    use_mps,
    save_images,
    frac_dataset,
    global_merge,
    profile_flops,
    mpm_layers,
):
    model_dir = Path(model_path).parent

    # start distributed mode
    ptu.set_gpu_mode(use_gpu)
    if use_gpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        distributed.init_process()

    if use_mps:
        ptu.device = "mps"

    model, variant = load_model(model_path)

    # Configure global merging behavior
    if global_merge:
        model.encoder.global_merge = True
    else:
        model.encoder.global_merge = False
    # If explicit layers are provided, enable global merge and set layers
    if mpm_layers is not None and len(mpm_layers) > 0:
        # parse strings that may contain comma-separated indices
        parsed_layers = []
        try:
            for v in mpm_layers:
                tokens = str(v).replace(',', ' ').split()
                for t in tokens:
                    parsed_layers.append(int(t))
        except ValueError:
            raise click.BadParameter("--mpm-layers expects integers, e.g., --mpm-layers 0,1,2")
        if len(parsed_layers) > 0:
            model.encoder.global_merge = True
            model.encoder.mpm_layers = sorted(set(parsed_layers))

    patch_size = model.patch_size
    model.eval()
    model.to(ptu.device)
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    cfg = config.load_config()
    dataset_cfg = cfg["dataset"][dataset_name]
    normalization = variant["dataset_kwargs"]["normalization"]
    if im_size is None:
        im_size = dataset_cfg.get("im_size", variant["dataset_kwargs"]["image_size"])
    if window_size is None:
        window_size = variant["dataset_kwargs"]["crop_size"]
    if window_stride is None:
        window_stride = variant["dataset_kwargs"]["crop_size"] - 32

    dataset_kwargs = dict(
        dataset=dataset_name,
        image_size=im_size,
        crop_size=im_size,
        patch_size=patch_size,
        batch_size=1,
        num_workers=10,
        split="val",
        normalization=normalization,
        crop=False,
        rep_aug=False,
    )

    eval_dataset(
        model,
        multiscale,
        model_dir,
        blend,
        window_size,
        window_stride,
        window_batch_size,
        save_images,
        frac_dataset,
        dataset_kwargs,
        profile_flops=profile_flops,
    )

    if ptu.distributed:
        distributed.barrier()
        distributed.destroy_process()
    sys.exit(0)


if __name__ == "__main__":
    main()
