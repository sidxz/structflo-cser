"""Sweep the relational matcher's dustbin margin to trade precision for recall.

The detector pass and the SetMatcher Sinkhorn output Z depend only on the page,
not on the acceptance margin — so Z is computed once per page and cheaply
re-thresholded across margins. Tunes on real_val, reports on real_test
(no selection-on-test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from structflo.cser.relmatch.features import node_features
from structflo.cser.relmatch.model import load_checkpoint


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cent(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def cache_pages(model_det, relmodel, gt_dir, img_dir, conf, imgsz, device):
    """Per page: detect, compute Z once, keep struct/label boxes + GT entries."""
    pages = []
    for gtf in sorted(gt_dir.glob("*.json")):
        ip = img_dir / f"{gtf.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gtf.stem}.jpg"
        if not ip.exists():
            continue
        entries = json.loads(gtf.read_text())
        img_rgb = np.array(Image.open(ip).convert("L").convert("RGB"))
        page_h, page_w = img_rgb.shape[:2]
        res = model_det(img_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
        s_boxes, s_conf, l_boxes, l_conf = [], [], [], []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            if int(box.cls[0]) == 0:
                s_boxes.append([x1, y1, x2, y2])
                s_conf.append(float(box.conf[0]))
            else:
                l_boxes.append([x1, y1, x2, y2])
                l_conf.append(float(box.conf[0]))
        gt_pairs = sum(1 for e in entries if e.get("label_bbox") is not None)
        if not s_boxes or not l_boxes:
            pages.append({"entries": entries, "Z": None, "s_boxes": s_boxes,
                          "l_boxes": l_boxes, "gt_pairs": gt_pairs})
            continue
        boxes = s_boxes + l_boxes
        classes = [0] * len(s_boxes) + [1] * len(l_boxes)
        confs = s_conf + l_conf
        nodes = torch.from_numpy(node_features(boxes, classes, confs, page_w, page_h)).to(device)
        is_struct = torch.tensor([True] * len(s_boxes) + [False] * len(l_boxes), device=device)
        with torch.no_grad():
            Z = relmodel(nodes, is_struct).cpu().numpy()
        pages.append({"entries": entries, "Z": Z, "s_boxes": s_boxes,
                      "l_boxes": l_boxes, "gt_pairs": gt_pairs})
    return pages


def score(pages, margin):
    tp = npred = gt_total = 0
    for pg in pages:
        gt_total += pg["gt_pairs"]
        Z = pg["Z"]
        if Z is None:
            continue
        n_s, n_l = len(pg["s_boxes"]), len(pg["l_boxes"])
        core = Z[:n_s, :n_l]
        dust = Z[:n_s, n_l]
        r_ind, c_ind = linear_sum_assignment(-core)
        entries = pg["entries"]
        for r, c in zip(r_ind, c_ind):
            if core[r, c] < dust[r] - margin:
                continue
            npred += 1
            ps, pl = pg["s_boxes"][r], pg["l_boxes"][c]
            bi, bv = -1, 0.0
            for i, e in enumerate(entries):
                v = _iou(ps, e["struct_bbox"])
                if v > bv:
                    bi, bv = i, v
            if bi < 0:
                continue
            gl = entries[bi].get("label_bbox")
            if gl is not None and bv >= 0.5 and _inside(_cent(gl), pl):
                tp += 1
    P = tp / max(npred, 1)
    R = tp / max(gt_total, 1)
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data/finetune/lps"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--relmatch", type=Path, default=Path("runs/relmatch_det/best.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(str(args.detector))
    relmodel, _ = load_checkpoint(args.relmatch, device=device)

    cache = {}
    for split in ("val", "real_test"):
        cache[split] = cache_pages(
            det, relmodel, args.base / split / "ground_truth",
            args.base / split / "images", args.conf, args.imgsz, device,
        )

    margins = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print(f"{'margin':>7} | {'val P':>6} {'val R':>6} {'val F1':>6} | {'test P':>6} {'test R':>6} {'test F1':>7}")
    print("-" * 60)
    best_m, best_vf = 0.0, -1.0
    rows = {}
    for m in margins:
        vp, vr, vf = score(cache["val"], m)
        tp, tr, tf = score(cache["real_test"], m)
        rows[m] = (tp, tr, tf)
        if vf > best_vf:
            best_vf, best_m = vf, m
        print(f"{m:>7.1f} | {vp:>6.3f} {vr:>6.3f} {vf:>6.3f} | {tp:>6.3f} {tr:>6.3f} {tf:>7.3f}")
    tp, tr, tf = rows[best_m]
    print("-" * 60)
    print(f"best margin by VAL F1 = {best_m}  →  REAL_TEST  P {tp:.3f}  R {tr:.3f}  F1 {tf:.3f}")


if __name__ == "__main__":
    main()
