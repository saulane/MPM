from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints"

MODELS = {
    "ade20k-seg-t-mask": {
        "directory": "ade20k/seg_tiny_mask",
        "files": ("checkpoint.pth", "variant.yml"),
    },
    "ade20k-seg-s-mask": {
        "directory": "ade20k/seg_small_mask",
        "files": ("checkpoint.pth", "variant.yml"),
    },
    "ade20k-seg-b-mask": {
        "directory": "ade20k/seg_base_mask",
        "files": ("checkpoint.pth", "variant.yml"),
    },
    "ade20k-seg-l-mask": {
        "directory": "ade20k/seg_large_mask_640",
        "files": ("checkpoint.pth", "variant.yml"),
    },
    "cityscapes-seg-l-mask": {
        "directory": "cityscapes/seg_large_mask",
        "files": ("checkpoint.pth", "variant.yml"),
    },
    "pcontext-seg-l-mask": {
        "directory": "pascal_context/seg_large_mask",
        "files": ("checkpoint.pth", "variant.yml"),
    },
}


def download_model(name: str, output_dir: Path, overwrite: bool = False) -> Path:
    if name not in MODELS:
        valid = ", ".join(sorted(MODELS))
        raise SystemExit(f"Unknown model {name!r}. Valid models: {valid}")

    spec = MODELS[name]
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)

    for filename in spec["files"]:
        dst = target / filename
        if dst.exists() and not overwrite:
            print(f"exists: {dst}")
            continue
        url = f"{BASE_URL}/{spec['directory']}/{filename}"
        print(f"download: {url}")
        urlretrieve(url, dst)

    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Segmenter checkpoints.")
    parser.add_argument("name", choices=sorted(MODELS))
    parser.add_argument("--output-dir", default="checkpoints", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = download_model(args.name, args.output_dir, overwrite=args.overwrite)
    print(path)


if __name__ == "__main__":
    main()

