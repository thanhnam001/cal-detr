# ------------------------------------------------------------------------
# Convert Cityscapes (gtFine instance annotations) to a COCO-format json,
# for use with this repo's datasets/coco.py loader (--dataset_file coco).
#
# Reproduces the 8-class in-domain detection setup used by Cal-DETR's
# Cityscapes experiment (arXiv:2311.03570, Table 4): person, rider, car,
# truck, bus, train, motorcycle, bicycle. This is also the standard class
# set from the Cityscapes->Foggy-Cityscapes domain-adaptive-detection
# benchmark (Chen et al. 2018), which Cal-DETR's setup follows.
#
# Unlike mmdetection's tools/dataset_converters/cityscapes.py, this has no
# dependency on the `cityscapesscripts` package (not installed in any env
# on this machine) -- the label table below is hardcoded from Cityscapes'
# own cityscapesscripts/helpers/labels.py and will not change (the label
# definitions have been stable since the dataset's release).
#
# Output category_ids are remapped to a CONTIGUOUS 1..8 range (not the raw
# Cityscapes labelIds 24-33), so --num_classes 9 (8 classes + 1, matching
# this repo's own "+1" convention for num_classes=91=90coco+1) gives a
# reasonably sized classification head instead of one padded out to 34
# classes for 26 of which there is never a positive label.
#
# Usage:
#   python tools/cityscapes2coco.py --root <cityscapes_root> --split train
#   python tools/cityscapes2coco.py --root <cityscapes_root> --split val
#   # out-domain eval (Foggy Cityscapes, "severe fog" = beta 0.02):
#   python tools/cityscapes2coco.py --root <cityscapes_root> --split val \
#       --img-dir leftImg8bit_foggy --foggy-beta 0.02 --out foggy_val.json
# ------------------------------------------------------------------------
import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
from PIL import Image

# (labelId, name) for the 8 "thing" classes with hasInstances=True and
# ignoreInEval=False in Cityscapes' official label definitions. caravan(29)
# and trailer(30) also have hasInstances=True but are ignoreInEval=True, so
# they're correctly excluded here -- this list matches the paper's "8
# classes person, rider, car, truck, bus, train, motorbike, and bicycle".
CITYSCAPES_THING_LABELS = [
    (24, "person"),
    (25, "rider"),
    (26, "car"),
    (27, "truck"),
    (28, "bus"),
    (31, "train"),
    (32, "motorcycle"),
    (33, "bicycle"),
]
# raw Cityscapes labelId -> contiguous 1..8 COCO-style category_id
LABELID_TO_CATID = {lid: i + 1 for i, (lid, _) in enumerate(CITYSCAPES_THING_LABELS)}


def collect_files(img_dir, gt_dir, foggy_beta=None):
    """Pair each RGB image with its instanceIds annotation PNG.

    Foggy Cityscapes images are named
    <city>_<seq>_<frame>_leftImg8bit_foggy_beta_<beta>.png and share the
    SAME gtFine annotation as the corresponding clear image (fog is a
    synthetic post-process on the RGB only, not a re-annotation).
    """
    suffix = "_leftImg8bit_foggy_beta_{}.png".format(foggy_beta) if foggy_beta else "_leftImg8bit.png"
    files = []
    for img_file in sorted(glob.glob(osp.join(img_dir, "**", "*.png"), recursive=True)):
        if not img_file.endswith(suffix):
            continue
        base = osp.basename(img_file)[: -len(suffix)]
        city = osp.basename(osp.dirname(img_file))
        inst_file = osp.join(gt_dir, city, base + "_gtFine_instanceIds.png")
        if not osp.exists(inst_file):
            raise FileNotFoundError(
                f"no gtFine annotation for {img_file} (expected {inst_file}); "
                f"Foggy Cityscapes reuses clear-image annotations, make sure --gt-dir "
                f"points at the CLEAR gtFine tree, not a foggy one")
        files.append((img_file, inst_file, city, base))
    if not files:
        raise RuntimeError(f"no images found under {img_dir} matching suffix '{suffix}'")
    print(f"found {len(files)} images")
    return files


def image_annotations(inst_file):
    inst_img = np.array(Image.open(inst_file))
    # instance ids >= 1000 encode label_id*1000 + instance_index (non-crowd);
    # ids in [24,33] with no *1000 offset are crowd regions for that label.
    anno_info = []
    for inst_id in np.unique(inst_img):
        label_id = int(inst_id // 1000) if inst_id >= 1000 else int(inst_id)
        if label_id not in LABELID_TO_CATID:
            continue
        mask = inst_img == inst_id
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        anno_info.append({
            "category_id": LABELID_TO_CATID[label_id],
            "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
            "area": float(mask.sum()),
            "iscrowd": 1 if inst_id < 1000 else 0,
        })
    return anno_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Cityscapes root (parent of leftImg8bit/, gtFine/)")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--img-dir", default="leftImg8bit", help="image subdir name (e.g. leftImg8bit_foggy)")
    ap.add_argument("--gt-dir", default="gtFine", help="annotation subdir name (always the CLEAR gtFine)")
    ap.add_argument("--foggy-beta", default=None, choices=[None, "0.005", "0.01", "0.02"],
                    help="set for Foggy Cityscapes; 0.02 = 'severe fog' used in the paper's out-domain eval")
    ap.add_argument("--out", default=None, help="output json path (default: <root>/<split>[_foggy].json)")
    a = ap.parse_args()

    img_dir = osp.join(a.root, a.img_dir, a.split)
    gt_dir = osp.join(a.root, a.gt_dir, a.split)
    out_path = a.out or osp.join(a.root, f"{a.split}{'_foggy' + a.foggy_beta if a.foggy_beta else ''}.json")

    files = collect_files(img_dir, gt_dir, a.foggy_beta)

    images, annotations = [], []
    img_id = ann_id = 0
    for img_file, inst_file, city, base in files:
        with Image.open(img_file) as im:
            w, h = im.size
        images.append({
            "id": img_id,
            "file_name": osp.relpath(img_file, a.root).replace(os.sep, "/"),
            "width": w,
            "height": h,
        })
        for anno in image_annotations(inst_file):
            anno["id"] = ann_id
            anno["image_id"] = img_id
            annotations.append(anno)
            ann_id += 1
        img_id += 1
        if img_id % 200 == 0:
            print(f"  {img_id}/{len(files)} images processed")

    categories = [{"id": cid, "name": name} for cid, (_, name) in
                 zip(range(1, len(CITYSCAPES_THING_LABELS) + 1), CITYSCAPES_THING_LABELS)]

    with open(out_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"wrote {out_path}  ({len(images)} images, {len(annotations)} annotations, "
          f"{len(categories)} categories)")


if __name__ == "__main__":
    main()
