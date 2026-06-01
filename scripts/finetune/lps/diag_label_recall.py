"""Diagnose the LABEL detection bottleneck on real_test.

Runs detection ONCE per page at a low conf floor (captures all candidates with
scores), for both tiled and full-image modes, then sweeps conf thresholds in
post-processing. Answers:

  H1  conf 0.3 too high?          -> label recall vs conf, vs FP-per-page
  H2  tiling cuts labels?         -> tiled vs full-image label recall
  KEY are misses recoverable?     -> for labels missed at conf 0.3 tiled,
                                     was there ANY candidate box (conf>=floor)
                                     overlapping it? (below-threshold vs truly-undetected)
  characterize misses             -> size distribution of missed vs hit labels;
                                     proximity to tile boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.detector import detect_tiled
from structflo.cser.inference.tiling import generate_tiles


def detect_full_imgsz(model, img, conf, imgsz):
    """Full-image inference at an explicit imgsz."""
    results = model(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        out.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(box.conf[0]), "class_id": int(box.cls[0])})
    return out


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _best_iou(g, dets):
    """Return (best_iou, best_conf) of dets overlapping g."""
    bi, bc = 0.0, 0.0
    for d in dets:
        v = _iou(g, d["bbox"])
        if v > bi:
            bi, bc = v, d["conf"]
    return bi, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/finetune/lps/real_test"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--floor", type=float, default=0.01, help="low conf floor for candidate capture")
    ap.add_argument("--tile-size", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    gt_dir = args.data_dir / "ground_truth"
    img_dir = args.data_dir / "images"

    confs = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    iou_thrs = [0.5, 0.3]
    modes = ["tiled", "full640", "full1280", "full2048"]
    # per (mode, conf, iou, class): hits / total ; class 1=label, 0=struct
    rec = {m: {c: {t: {cl: [0, 0] for cl in (0, 1)} for t in iou_thrs} for c in confs} for m in modes}
    # FP label boxes per page (label-class dets not overlapping any GT label IoU>=0.3) per (mode,conf)
    fp = {m: {c: 0 for c in confs} for m in modes}
    n_pages = 0

    # miss characterization at the operating point (tiled, conf 0.3, IoU 0.5)
    miss_sizes, hit_sizes = [], []
    miss_recoverable = 0  # candidate existed at floor but below 0.3
    miss_truly_undetected = 0  # no candidate box at all (IoU>=0.3)
    miss_near_edge = 0  # missed label whose box straddles/touches a tile boundary
    miss_total = 0

    files = sorted(gt_dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    for k, gt in enumerate(files):
        entries = json.loads(gt.read_text())
        ip = img_dir / f"{gt.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gt.stem}.jpg"
        if not ip.exists():
            continue
        img_np = np.array(Image.open(ip).convert("L").convert("RGB"))
        h, w = img_np.shape[:2]
        n_pages += 1

        gt_l = [e["label_bbox"] for e in entries if e.get("label_bbox") is not None]
        gt_s = [e["struct_bbox"] for e in entries]
        gt_by = {0: gt_s, 1: gt_l}

        # detect once at floor per mode
        det = {
            "tiled": detect_tiled(model, img_np, tile_size=args.tile_size, conf=args.floor),
            "full640": detect_full_imgsz(model, img_np, args.floor, 640),
            "full1280": detect_full_imgsz(model, img_np, args.floor, 1280),
            "full2048": detect_full_imgsz(model, img_np, args.floor, 2048),
        }

        for mode in modes:
            for cl in (0, 1):
                dets_all = [d for d in det[mode] if d["class_id"] == cl]
                for c in confs:
                    dets = [d for d in dets_all if d["conf"] >= c]
                    for t in iou_thrs:
                        hit = sum(any(_iou(g, d["bbox"]) >= t for d in dets) for g in gt_by[cl])
                        rec[mode][c][t][cl][0] += hit
                        rec[mode][c][t][cl][1] += len(gt_by[cl])
                    if cl == 1:
                        for d in dets:
                            if not any(_iou(g, d["bbox"]) >= 0.3 for g in gt_l):
                                fp[mode][c] += 1

        # miss characterization @ tiled conf 0.3 IoU 0.5
        lab_t_floor = [d for d in det["tiled"] if d["class_id"] == 1]
        lab_t_030 = [d for d in lab_t_floor if d["conf"] >= 0.30]
        # tile boundary x/y coords
        tiles = generate_tiles(w, h, args.tile_size, 0.20)
        xbounds = set()
        ybounds = set()
        for x1, y1, x2, y2 in tiles:
            xbounds.update([x1, x2])
            ybounds.update([y1, y2])
        for g in gt_l:
            area = (g[2] - g[0]) * (g[3] - g[1])
            hit030 = any(_iou(g, d["bbox"]) >= 0.5 for d in lab_t_030)
            if hit030:
                hit_sizes.append(area)
                continue
            miss_total += 1
            miss_sizes.append(area)
            # recoverable? candidate at floor overlapping IoU>=0.3
            bi, _ = _best_iou(g, lab_t_floor)
            if bi >= 0.3:
                miss_recoverable += 1
            else:
                miss_truly_undetected += 1
            # near tile edge? any interior boundary line passes through the box
            near = False
            for xb in xbounds:
                if 0 < xb < w and g[0] - 5 <= xb <= g[2] + 5:
                    near = True
            for yb in ybounds:
                if 0 < yb < h and g[1] - 5 <= yb <= g[3] + 5:
                    near = True
            if near:
                miss_near_edge += 1

        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(files)} pages")

    print(f"\n=== {n_pages} pages, conf floor {args.floor}, tile {args.tile_size} ===\n")
    for mode in modes:
        print(f"--- {mode.upper()} ---")
        print(f"  {'conf':>5} | {'LABEL R.5':>10} {'LABEL R.3':>10} | {'STRUCT R.5':>10} {'STRUCT R.3':>10} | {'LblFP/pg':>8}")
        for c in confs:
            def _v(cl, t):
                a = rec[mode][c][t][cl]
                return a[0] / a[1] if a[1] else 0
            print(f"  {c:>5.2f} | {_v(1,0.5):>10.1%} {_v(1,0.3):>10.1%} | "
                  f"{_v(0,0.5):>10.1%} {_v(0,0.3):>10.1%} | {fp[mode][c]/n_pages:>8.2f}")
        print()

    print("=== MISS CHARACTERIZATION (tiled, conf 0.30, IoU>=0.5) ===")
    print(f"  total misses: {miss_total}")
    if miss_total:
        print(f"  recoverable (candidate existed at floor, conf<0.30): {miss_recoverable} ({miss_recoverable/miss_total:.1%})")
        print(f"  truly undetected (no candidate IoU>=0.3 even at floor): {miss_truly_undetected} ({miss_truly_undetected/miss_total:.1%})")
        print(f"  near tile boundary: {miss_near_edge} ({miss_near_edge/miss_total:.1%})")
    if miss_sizes and hit_sizes:
        ms = np.array(miss_sizes) ** 0.5  # sqrt(area) ~ side length px
        hs = np.array(hit_sizes) ** 0.5
        print("\n  label box size (sqrt-area px):")
        print(f"    MISSED: median {np.median(ms):.0f}  p25 {np.percentile(ms,25):.0f}  p75 {np.percentile(ms,75):.0f}  min {ms.min():.0f}")
        print(f"    HIT:    median {np.median(hs):.0f}  p25 {np.percentile(hs,25):.0f}  p75 {np.percentile(hs,75):.0f}  min {hs.min():.0f}")


if __name__ == "__main__":
    main()
