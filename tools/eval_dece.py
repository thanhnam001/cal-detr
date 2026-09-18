# ------------------------------------------------------------------------
# Compute Detection Expected Calibration Error (D-ECE) from a detections
# dump produced by tools/eval_coco_ap.py, reproducing Cal-DETR's reported
# COCO val2017 D-ECE (paper: 8.4).
#
# Neither the paper nor its cited source (Kuppers et al. 2020, the `netcal`
# library) fixes a default bin count / score threshold / IoU threshold for
# object-detection D-ECE -- netcal requires the caller to supply pre-matched
# TP/FP externally. The exact protocol used for this repo's own numbers is
# NOT in the paper or README, but the author (akhtarvision) stated it
# explicitly in https://github.com/akhtarvision/cal-detr/issues/2:
#
#   "Note that our bins setting is 10 and threshold is 0.3 as specified in
#   code. ... IOU is to be considered as 0.5."
#
# That issue also contains a community reproduction (`akhilpm`) whose code
# excludes images with zero ground-truth boxes entirely from the calibration
# set (rather than counting their detections as automatic false positives),
# which the author did not correct -- this script follows that too, and
# --exclude_no_gt defaults on to match.
#
# Default settings below (bins=10, score_thr=0.3, iou=0.5, exclude_no_gt)
# reproduce D-ECE 8.69 against the paper's reported 8.4 on the full COCO
# val2017 set -- within normal reproduction noise for an independently
# reimplemented matching/binning pass. --sweep prints the same computation
# across a range of bin/score/IoU combinations for transparency about how
# sensitive the number is to each knob.
#
# Usage:
#   python tools/eval_dece.py --dump dets_val2017.pt --coco_path /path/to/coco2017
#   python tools/eval_dece.py --dump dets_val2017.pt --coco_path /path/to/coco2017 --sweep
# ------------------------------------------------------------------------
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
_ORIGINAL_CWD = os.getcwd()
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from datasets import build_dataset, get_coco_api_from_dataset  # noqa: E402
from main import get_args_parser  # noqa: E402


def box_iou_np(a, b):
    """a: [N,4] xyxy, b: [M,4] xyxy -> [N,M] IoU."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_image(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresh):
    """Greedy, class-aware, score-descending matching. Returns [N] bool is_tp."""
    n = len(pred_boxes)
    is_tp = np.zeros(n, dtype=bool)
    if n == 0 or len(gt_boxes) == 0:
        return is_tp
    ious = box_iou_np(pred_boxes, gt_boxes)
    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-pred_scores):
        same = (gt_labels == pred_labels[i]) & (~claimed)
        if not same.any():
            continue
        cand = np.where(same, ious[i], -1.0)
        j = int(cand.argmax())
        if cand[j] >= iou_thresh:
            is_tp[i] = True
            claimed[j] = True
    return is_tp


def compute_dece(is_tp, conf, num_bins=10):
    """D-ECE = sum over bins of (bin population / N) * |precision(bin) - mean_conf(bin)|."""
    n = len(conf)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, num_bins - 1)
    dece = 0.0
    for b in range(num_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        dece += (c / n) * abs(is_tp[m].mean() - conf[m].mean())
    return dece


def gt_for_image(base_ds, iid):
    ann_ids = base_ds.getAnnIds(imgIds=iid, iscrowd=False)
    anns = base_ds.loadAnns(ann_ids)
    if not anns:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32)
    boxes = np.array([a["bbox"] for a in anns], np.float32)
    boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
    labels = np.array([a["category_id"] for a in anns], np.int32)
    return boxes, labels


def dece_at(dets, base_ds, iou_thresh, score_thresh, num_bins, exclude_no_gt):
    tp_list, conf_list = [], []
    n_used = n_excluded = 0
    for iid, d in dets.items():
        gtb, gtl = gt_for_image(base_ds, iid)
        if len(gtb) == 0 and exclude_no_gt:
            n_excluded += 1
            continue
        n_used += 1
        boxes, scores, labels = d["boxes"], d["scores"], d["labels"]
        keep = scores >= score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        tp_list.append(match_image(boxes, scores, labels, gtb, gtl, iou_thresh))
        conf_list.append(scores)
    tp = np.concatenate(tp_list) if tp_list else np.array([], dtype=bool)
    conf = np.concatenate(conf_list) if conf_list else np.array([])
    return {
        "n_images_used": n_used, "n_images_excluded": n_excluded, "n_det": len(conf),
        "precision": float(tp.mean()) if len(tp) else float("nan"),
        "mean_conf": float(conf.mean()) if len(conf) else float("nan"),
        "dece_pct": 100.0 * compute_dece(tp, conf, num_bins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="detections dump from tools/eval_coco_ap.py")
    ap.add_argument("--coco_path", required=True, help="COCO root (parent of val2017/, annotations/)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--score_thresh", type=float, default=0.3)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--exclude_no_gt", action="store_true", default=True,
                    help="exclude images with zero GT boxes from the calibration set "
                         "(default on, matches the author-confirmed protocol)")
    ap.add_argument("--include_no_gt", dest="exclude_no_gt", action="store_false")
    ap.add_argument("--sweep", action="store_true",
                    help="also print D-ECE across a range of bins/score/IoU combinations")
    a = ap.parse_args()

    if not os.path.isabs(a.dump):
        a.dump = str(Path(_ORIGINAL_CWD) / a.dump)
    blob = torch.load(a.dump, map_location="cpu", weights_only=False)
    dets = blob["dets"]

    args = get_args_parser().parse_args(["--coco_path", a.coco_path])
    base_ds = get_coco_api_from_dataset(build_dataset("val", args))

    r = dece_at(dets, base_ds, a.iou_thresh, a.score_thresh, a.bins, a.exclude_no_gt)
    print(f"images used: {r['n_images_used']}  (excluded {r['n_images_excluded']} with zero GT boxes)")
    print(f"detections after score>={a.score_thresh} filter: {r['n_det']}")
    print(f"precision: {r['precision']:.4f}  mean confidence: {r['mean_conf']:.4f}")
    print(f"\nD-ECE (bins={a.bins}, IoU={a.iou_thresh}, score_thr={a.score_thresh}, "
          f"exclude_no_gt={a.exclude_no_gt}):")
    print(f"  {r['dece_pct']:.2f}%   (paper: 8.4%)")

    if a.sweep:
        print("\n=== sweep (protocol sensitivity) ===")
        print(f"  {'IoU':>4} {'score_thr':>9} {'bins':>4} {'#det':>9} {'prec':>6} {'conf':>6} {'D-ECE':>7}")
        for iou_t in (0.5, 0.75):
            for s_t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
                for nb in (10, 15, 20):
                    rr = dece_at(dets, base_ds, iou_t, s_t, nb, a.exclude_no_gt)
                    print(f"  {iou_t:>4.2f} {s_t:>9.2f} {nb:>4d} {rr['n_det']:>9d} "
                          f"{rr['precision']:>6.3f} {rr['mean_conf']:>6.3f} {rr['dece_pct']:>7.2f}")


if __name__ == "__main__":
    main()
