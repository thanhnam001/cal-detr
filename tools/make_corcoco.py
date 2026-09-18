# ------------------------------------------------------------------------
# Build CorCOCO, the corrupted-COCO out-domain eval set from Cal-DETR's
# paper (arXiv:2311.03570), Table 1 ("D-ECE (CorCOCO)", "APbox (CorCOCO)").
#
# Per the paper: "CorCOCO contains a similar set of images as present in
# val2017 of MS-COCO but with the corrupted version. Random corruptions with
# random severity levels are introduced for evaluation in an out-domain
# scenario." -- citing Hendrycks & Dietterich 2019 ("Benchmarking Neural
# Network Robustness to Common Corruptions and Perturbations", the source
# of the standard ImageNet-C / COCO-C corruption suite) for the methodology.
# The paper doesn't specify a random seed or give a corruption/severity
# manifest, so this can only be reproduced UP TO the random draw -- expect
# eval numbers to vary run-to-run of the *generation* step (not of eval
# itself, which is deterministic given a fixed CorCOCO copy). --seed fixes
# that draw so re-running this script is at least reproducible for you.
#
# Ground truth is unaffected by pixel-level corruption -- object locations
# and classes don't move -- so this only regenerates val2017/ images and
# reuses the original instances_val2017.json unchanged.
#
# Uses the `imagecorruptions` package (Michaelis et al. 2019, the standard
# wrapper around Hendrycks & Dietterich's corruption functions used
# throughout the detection-robustness literature for exactly this kind of
# "COCO-C" construction) -- pip install imagecorruptions. Note: version
# 1.1.2 predates NumPy 2.0 and current scikit-image, and breaks on both:
#   - `fog` uses the removed np.float_ alias in its plasma_fractal helper.
#   - `gaussian_blur`/`glass_blur`/`zoom_blur` call skimage.filters.gaussian
#     with the removed multichannel=True kwarg (renamed to channel_axis).
# Both patched below rather than touching the installed package or pinning
# older numpy/scikit-image for the whole env.
#
# Usage:
#   python tools/make_corcoco.py --coco_path /path/to/coco2017 \
#       --out /path/to/corcoco --split val --seed 0
# ------------------------------------------------------------------------
import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64  # see module docstring: imagecorruptions 1.1.2 / NumPy 2.0

from imagecorruptions import corrupt, get_corruption_names
import imagecorruptions.corruptions as _ic_corruptions
from PIL import Image

_orig_gaussian = _ic_corruptions.gaussian


def _gaussian_compat(image, sigma=1.0, **kwargs):
    """Shim for skimage.filters.gaussian's removed multichannel= kwarg (see
    module docstring). multichannel=True meant "channels are the last axis",
    which current skimage expresses as channel_axis=-1."""
    if "multichannel" in kwargs:
        mc = kwargs.pop("multichannel")
        kwargs["channel_axis"] = -1 if mc else None
    return _orig_gaussian(image, sigma=sigma, **kwargs)


_ic_corruptions.gaussian = _gaussian_compat

# the 15 standard ImageNet-C corruptions (Hendrycks & Dietterich 2019), matching
# get_corruption_names()'s default -- NOT get_corruption_names('all'), which adds 4
# extra "validation" corruptions from that paper not part of the main benchmark suite.
CORRUPTIONS = get_corruption_names()


def corrupt_one(img_path, out_path, corruption_name, severity):
    img = np.array(Image.open(img_path).convert("RGB"))
    out = corrupt(img, severity=severity, corruption_name=corruption_name)
    Image.fromarray(out.astype(np.uint8)).save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_path", required=True, help="source COCO root (has val2017/, annotations/)")
    ap.add_argument("--out", required=True, help="output CorCOCO root")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; each image draws deterministically from "
                         "seed + image_id, so reruns with the same seed are reproducible")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    a = ap.parse_args()

    src_root = Path(a.coco_path)
    out_root = Path(a.out)
    ann_file = src_root / "annotations" / f"instances_{a.split}2017.json"
    assert ann_file.exists(), f"{ann_file} not found"

    with open(ann_file) as f:
        coco = json.load(f)
    images = coco["images"]
    if a.limit:
        images = images[: a.limit]

    out_img_dir = out_root / f"{a.split}2017"
    out_ann_dir = out_root / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    # GT is unaffected by pixel corruption -- copy annotations unchanged (all splits'
    # files, not just the one being corrupted, so --coco_path can point straight at
    # this output root for eval without missing files).
    for f in (src_root / "annotations").glob("*.json"):
        shutil.copy(f, out_ann_dir / f.name)

    manifest_path = out_root / f"{a.split}_corruption_manifest.csv"
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["image_id", "file_name", "corruption_name", "severity"])

        for i, img_info in enumerate(images):
            iid = img_info["id"]
            fname = img_info["file_name"]
            rng = random.Random(a.seed + iid)  # deterministic per image, order-independent
            corruption_name = rng.choice(CORRUPTIONS)
            severity = rng.randint(1, 5)

            src_path = src_root / f"{a.split}2017" / fname
            out_path = out_img_dir / fname
            corrupt_one(src_path, out_path, corruption_name, severity)
            writer.writerow([iid, fname, corruption_name, severity])

            if (i + 1) % 200 == 0:
                print(f"[{i + 1}/{len(images)}] corrupted", flush=True)

    print(f"\nwrote {len(images)} corrupted images to {out_img_dir}")
    print(f"manifest: {manifest_path}")
    print(f"annotations copied unchanged to {out_ann_dir}")
    print(f"\neval against this set with:\n"
          f"  python tools/eval_coco_ap.py --coco_path {out_root} ...\n"
          f"  python tools/eval_dece.py --coco_path {out_root} ...")


if __name__ == "__main__":
    main()
