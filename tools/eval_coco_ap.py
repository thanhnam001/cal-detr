# ------------------------------------------------------------------------
# Reproduce Cal-DETR's reported COCO val2017 bbox AP (paper: 44.4) from a
# released checkpoint, on hardware too memory-constrained for the repo's own
# main.py --eval path to finish reliably.
#
# Why not just `python main.py --eval`:
#   engine.py's evaluate() calls datasets/coco_eval.py:CocoEvaluator.update()
#   once per BATCH, which calls pycocotools COCO.loadRes() every time -- that
#   rebuilds a full COCO index (with img<->annotation reference cycles) from
#   scratch on every call. Over ~2500-5000 batches the resulting cyclic
#   garbage falls behind Python's GC faster than it's collected, and host RAM
#   grows unboundedly until the OS kills the process (observed: killed at 88%
#   through COCO val2017 on a 24GB-RAM Windows machine). This script instead
#   accumulates raw detections in memory (numpy, not pycocotools objects) and
#   runs pycocotools' COCOeval exactly ONCE at the end.
#
# Also periodically checkpoints to --dump so a crash (or a Ctrl-C) doesn't
# lose completed work -- rerunning the same command resumes automatically by
# skipping image ids already in the checkpoint.
#
# On a laptop GPU with limited dedicated VRAM (e.g. 6GB), COCO's variable
# image resolutions can also push PyTorch's CUDA caching allocator's reserved
# high-water-mark past dedicated VRAM; Windows' WDDM driver then spills the
# excess into "shared GPU memory" backed 1:1 by system RAM, which looks
# exactly like a host RAM leak. --batch_size 1 (avoids padding waste from
# pairing differently-sized images) plus a periodic torch.cuda.empty_cache()
# call keeps this bounded; see the --debug_mem flag to verify on your own
# hardware if you hit the same symptom.
#
# Usage:
#   python tools/eval_coco_ap.py --ckpt r50_deformable_detr_caldetr_coco \
#       --coco_path /path/to/coco2017 --dump dets_val2017.pt
#
# Run with a torch/torchvision env compatible after util/misc.py's version
# guard fix (see that file); if the compiled MSDeformAttn CUDA extension
# isn't built for your torch/CUDA combo, models/ops falls back automatically
# to the pure-PyTorch reference implementation (slower, numerically
# equivalent -- it's the same function models/ops/test.py validates the CUDA
# kernel against).
# ------------------------------------------------------------------------
import argparse
import contextlib
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

REPO = Path(__file__).resolve().parent.parent
_ORIGINAL_CWD = os.getcwd()  # capture BEFORE chdir, so relative --dump/--coco_path
                             # paths resolve against where the user invoked this from,
                             # not silently against the repo root
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import util.misc as utils  # noqa: E402
from datasets import build_dataset, get_coco_api_from_dataset  # noqa: E402
from models import build_model  # noqa: E402
from main import get_args_parser  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

try:
    import psutil
    _PROC = psutil.Process()
    def _rss_mb():
        return _PROC.memory_info().rss / (1024 * 1024)
except ImportError:
    def _rss_mb():
        return float("nan")


def run_coco_eval_once(base_ds, results_list, img_ids):
    """Single pycocotools pass over ALL detections (not per-batch).

    img_ids MUST be restricted to exactly the images that were evaluated --
    pycocotools defaults params.imgIds to ALL images in the ground-truth
    file, and any GT image with no uploaded detections is scored as pure
    false negatives, silently tanking AP. The repo's own CocoEvaluator sets
    this per batch; here it's set once for the whole run.
    """
    if not results_list:
        return [0.0] * 12
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        coco_dt = COCO.loadRes(base_ds, results_list)
        coco_eval = COCOeval(base_ds, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = sorted(img_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats.tolist()


def run_inference(ckpt, coco_path, batch_size, num_workers, limit, dump_path,
                  ckpt_every=500, debug_mem=False):
    args = get_args_parser().parse_args([
        "--coco_path", coco_path,
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--resume", ckpt,
        "--eval",
    ])
    args.distributed = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[load] epoch={ck.get('epoch')} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])

    dataset_val = build_dataset(image_set="val", args=args)
    base_ds = get_coco_api_from_dataset(dataset_val)
    all_ids = list(dataset_val.ids)  # deterministic sorted order (see torchvision_datasets/coco.py)

    # ---- resume support: skip image ids already checkpointed ----
    dets = {}
    results_list = []
    done_ids = set()
    if os.path.exists(dump_path):
        blob = torch.load(dump_path, map_location="cpu", weights_only=False)
        dets = blob.get("dets", {})
        results_list = blob.get("results_list", [])
        done_ids = set(dets.keys())
        print(f"[resume] found checkpoint with {len(done_ids)} images already done")

    remaining_indices = [i for i, iid in enumerate(all_ids) if iid not in done_ids]
    if limit:
        remaining_indices = remaining_indices[:limit]
    if not remaining_indices:
        print("[resume] nothing left to do, all images already in checkpoint")
        return dets, results_list, base_ds

    subset = Subset(dataset_val, remaining_indices)
    loader = DataLoader(subset, batch_size, sampler=SequentialSampler(subset),
                        drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers)

    n_batches = len(loader)
    n_total_images = len(all_ids) if not limit else len(remaining_indices) + len(done_ids)
    t0 = time.time()

    def checkpoint():
        torch.save({"dets": dets, "results_list": results_list}, dump_path)

    with torch.no_grad():
        for it, (samples, targets) in enumerate(loader):
            samples = samples.to(device)
            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            res = {t["image_id"].item(): o for t, o in zip(targets, results)}
            for iid, o in res.items():
                boxes = o["boxes"].cpu().numpy().astype(np.float32)
                scores = o["scores"].cpu().numpy().astype(np.float32)
                labels = o["labels"].cpu().numpy().astype(np.int32)
                dets[iid] = {"boxes": boxes, "scores": scores, "labels": labels}
                for b, s, l in zip(boxes, scores, labels):
                    results_list.append({
                        "image_id": int(iid),
                        "category_id": int(l),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s),
                    })
            del outputs, results, res, samples, orig_target_sizes, targets

            if debug_mem and it < 100:
                print(f"  [dbg it={it}] rss={_rss_mb():.0f}MB "
                      f"gpu_alloc={torch.cuda.memory_allocated()/1e6:.0f}MB "
                      f"gpu_reserved={torch.cuda.memory_reserved()/1e6:.0f}MB", flush=True)

            if (it + 1) % 50 == 0:
                # see module docstring: bounds GPU-reserved growth on small-VRAM cards
                torch.cuda.empty_cache()

            if it % 200 == 0:
                el = time.time() - t0
                rate = (it + 1) / max(el, 1e-6)
                print(f"[{len(done_ids) + (it + 1) * batch_size}/{n_total_images}] "
                      f"{el / 60:.1f} min elapsed, eta {(n_batches - it - 1) / rate / 60:.1f} min "
                      f"rss={_rss_mb():.0f}MB gpu_reserved={torch.cuda.memory_reserved()/1e6:.0f}MB",
                      flush=True)
                gc.collect()

            if (it + 1) % ckpt_every == 0:
                checkpoint()
                print(f"[checkpoint] saved at {len(dets)} images", flush=True)

    checkpoint()
    print(f"[dump] {dump_path}  ({len(dets)} images total)")
    return dets, results_list, base_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a Cal-DETR checkpoint (--resume-style)")
    ap.add_argument("--coco_path", required=True, help="COCO root (parent of val2017/, annotations/)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="1 recommended on small-VRAM GPUs -- avoids padding waste from "
                         "batching differently-sized images together, see module docstring")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only first N val images (smoke test)")
    ap.add_argument("--dump", default="dets_val2017.pt",
                    help="checkpoint/output path for raw detections; rerunning with the "
                         "same --dump resumes from wherever it left off")
    ap.add_argument("--ckpt_every", type=int, default=500,
                    help="checkpoint the dump every N batches processed")
    ap.add_argument("--from_dump", action="store_true",
                    help="skip inference entirely, just re-score an existing completed --dump")
    ap.add_argument("--debug_mem", action="store_true",
                    help="print per-iteration RSS/GPU-reserved for the first 100 iterations")
    a = ap.parse_args()
    if not os.path.isabs(a.dump):
        a.dump = str(Path(_ORIGINAL_CWD) / a.dump)

    if a.from_dump:
        blob = torch.load(a.dump, map_location="cpu", weights_only=False)
        dets, results_list = blob["dets"], blob["results_list"]
        args = get_args_parser().parse_args(["--coco_path", a.coco_path])
        base_ds = get_coco_api_from_dataset(build_dataset("val", args))
    else:
        dets, results_list, base_ds = run_inference(
            a.ckpt, a.coco_path, a.batch_size, a.num_workers, a.limit, a.dump,
            a.ckpt_every, a.debug_mem)

    print(f"\n[eval] running single pycocotools COCOeval pass over {len(dets)} images...")
    stats = run_coco_eval_once(base_ds, results_list, list(dets.keys()))

    names = ["AP", "AP50", "AP75", "APs", "APm", "APl",
             "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    print("\n=== COCO bbox (paper: AP 44.4) ===")
    for n, v in zip(names, stats):
        print(f"  {n:6s} {100 * v:.2f}")
    print(f"\nraw detections saved to {a.dump} -- feed into tools/eval_dece.py for calibration")


if __name__ == "__main__":
    main()
