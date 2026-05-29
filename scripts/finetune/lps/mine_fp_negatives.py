"""Mine detector false-positive structures as LPS rejection negatives.

Runs the fine-tuned YOLO over the (real) training pages, finds structure
detections that do NOT correspond to any ground-truth structure (IoU < 0.5),
keeps the ones near a label (the informative adjacent-distractor case), and
writes them to ``<train>/fp_negatives/<stem>.json`` as a list of [x1,y1,x2,y2].

LPSDataset(reject_negatives=True) then pairs each with its nearest labels as
target-0 samples, teaching the scorer to reject spurious detections — the exact
failure that drags Hungarian's end-to-end precision down.

NOTE: the detector trained on these pages, so it makes fewer/different FPs here
than on unseen pages; train FPs are a proxy for test-time FPs (visual character
transfers). Caveat acknowledged.

Usage:
    uv run python scripts/finetune/lps/mine_fp_negatives.py \
        --train-dir data/finetune/lps/train \
        --detector runs/labels_detect/finetune_3way/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.detector import detect_tiled

IOU_GT = 0.5          # detection counts as a real structure if IoU >= this
NEAR_LABEL_PX = 1000  # keep FP only if within this px of some label centroid
MAX_FP_PER_PAGE = 15  # cap (keep nearest-to-label) to avoid swamping training


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cent(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _min_label_dist(box, label_cents) -> float:
    if not label_cents:
        return float("inf")
    cx, cy = _cent(box)
    return min(((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 for lx, ly in label_cents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=Path("data/finetune/lps/train"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--real-only", action="store_true", default=True,
                    help="Only mine pages whose name contains '_real' (default).")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    gt_dir = args.train_dir / "ground_truth"
    img_dir = args.train_dir / "images"
    out_dir = args.train_dir / "fp_negatives"
    out_dir.mkdir(exist_ok=True)

    cache: dict[str, list] = {}  # resolved image path -> detections
    files = sorted(gt_dir.glob("*.json"))
    n_pages = n_fp = pages_with_fp = 0

    for k, gt in enumerate(files):
        stem = gt.stem
        if args.real_only and "_real" not in stem:
            continue
        ip = img_dir / f"{stem}.png"
        if not ip.exists():
            ip = img_dir / f"{stem}.jpg"
        if not ip.exists():
            continue

        real_path = os.path.realpath(ip)
        if real_path not in cache:
            img_np = np.array(Image.open(ip).convert("L").convert("RGB"))
            cache[real_path] = detect_tiled(model, img_np, tile_size=1536, conf=args.conf)
        dets = cache[real_path]

        entries = json.loads(gt.read_text())
        gt_structs = [e["struct_bbox"] for e in entries]
        label_cents = [_cent(e["label_bbox"]) for e in entries if e.get("label_bbox") is not None]

        fps = []
        for d in dets:
            if d["class_id"] != 0:
                continue
            box = d["bbox"]
            if any(_iou(box, gs) >= IOU_GT for gs in gt_structs):
                continue  # real structure detection
            if _min_label_dist(box, label_cents) > NEAR_LABEL_PX:
                continue  # far from any label — easy negative, skip
            fps.append((box, _min_label_dist(box, label_cents)))

        fps.sort(key=lambda t: t[1])
        fps = [b for b, _ in fps[:MAX_FP_PER_PAGE]]

        (out_dir / f"{stem}.json").write_text(json.dumps(fps))
        n_pages += 1
        n_fp += len(fps)
        if fps:
            pages_with_fp += 1
        if (k + 1) % 200 == 0:
            print(f"  processed {k + 1}/{len(files)} gt files, mined {n_fp} FPs so far")

    print(f"\nDone. pages written={n_pages}  unique images detected={len(cache)}")
    print(f"FP negatives mined={n_fp}  pages with >=1 FP={pages_with_fp}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
