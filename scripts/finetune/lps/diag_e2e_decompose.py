"""Decompose the end-to-end pair F1: is it detection-limited or matching-limited,
and how much does the strict IoU>=0.5-on-the-label criterion deflate it?

Reports, on real_test with real YOLO detections:
  - structure vs LABEL detection recall (IoU 0.5 and 0.3)
  - pair P/R/F1 under three correctness criteria:
      strict   : struct IoU>=0.5 AND label IoU>=0.5         (the eval_end2end criterion)
      iou0.3   : struct IoU>=0.3 AND label IoU>=0.3
      centroid : struct IoU>=0.5 AND GT label centroid inside predicted label box
                 (pairing-appropriate: right structure paired to right label region)
  - matcher-only ceiling: among GT pairs whose struct AND label were both detected
    (IoU>=0.5), fraction the matcher linked correctly — isolates matching from detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.detector import detect_tiled
from structflo.cser.lps.matcher import LearnedMatcher
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.models import Detection


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


def _recall(gt_boxes, det_boxes, thr):
    if not gt_boxes:
        return float("nan")
    hit = 0
    for g in gt_boxes:
        if any(_iou(g, d) >= thr for d in det_boxes):
            hit += 1
    return hit / len(gt_boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/finetune/lps/real_test"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--lps", type=Path, default=Path("runs/lps_finetune/best.pt"))
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    hung = HungarianMatcher()
    lps = LearnedMatcher(weights=str(args.lps), min_score=args.min_score)

    gt_dir = args.data_dir / "ground_truth"
    img_dir = args.data_dir / "images"

    # accumulators
    s_gt = s_det5 = s_det3 = 0
    l_gt = l_det5 = l_det3 = 0
    crit = {"strict": {}, "iou0.3": {}, "centroid": {}}
    for mname in ("Hungarian", "LPS"):
        for c in crit:
            crit[c][mname] = {"tp": 0, "np": 0}
    gt_pairs_total = 0
    ceil = {"Hungarian": {"tp": 0, "n": 0}, "LPS": {"tp": 0, "n": 0}}

    files = sorted(gt_dir.glob("*.json"))
    for k, gt in enumerate(files):
        entries = json.loads(gt.read_text())
        ip = img_dir / f"{gt.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gt.stem}.jpg"
        if not ip.exists():
            continue
        img_np = np.array(Image.open(ip).convert("L").convert("RGB"))
        raw = detect_tiled(model, img_np, tile_size=1536, conf=args.conf)
        dets = [Detection.from_dict(d) for d in raw]
        det_s = [d.bbox.as_list() for d in dets if d.class_id == 0]
        det_l = [d.bbox.as_list() for d in dets if d.class_id == 1]

        gt_s = [e["struct_bbox"] for e in entries]
        gt_l = [e["label_bbox"] for e in entries if e.get("label_bbox") is not None]
        labelled = [e for e in entries if e.get("label_bbox") is not None]
        gt_pairs_total += len(labelled)

        s_gt += len(gt_s); l_gt += len(gt_l)
        s_det5 += sum(any(_iou(g, d) >= 0.5 for d in det_s) for g in gt_s)
        s_det3 += sum(any(_iou(g, d) >= 0.3 for d in det_s) for g in gt_s)
        l_det5 += sum(any(_iou(g, d) >= 0.5 for d in det_l) for g in gt_l)
        l_det3 += sum(any(_iou(g, d) >= 0.3 for d in det_l) for g in gt_l)

        for mname, m in (("Hungarian", hung), ("LPS", lps)):
            pairs = m.match(dets, image=np.array(Image.open(ip).convert("L"))) if mname == "LPS" else m.match(dets)
            for c in crit:
                crit[c][mname]["np"] += len(pairs)
            for p in pairs:
                ps, pl = p.structure.bbox.as_list(), p.label.bbox.as_list()
                # best GT struct
                bi, bv = -1, 0.0
                for i, e in enumerate(entries):
                    v = _iou(ps, e["struct_bbox"])
                    if v > bv:
                        bi, bv = i, v
                if bi < 0:
                    continue
                e = entries[bi]
                gl = e.get("label_bbox")
                if gl is None:
                    continue  # matched a distractor -> wrong under all criteria
                il = _iou(pl, gl)
                if bv >= 0.5 and il >= 0.5:
                    crit["strict"][mname]["tp"] += 1
                if _iou(ps, e["struct_bbox"]) >= 0.3 and il >= 0.3:
                    crit["iou0.3"][mname]["tp"] += 1
                if bv >= 0.5 and _inside(_cent(gl), pl):
                    crit["centroid"][mname]["tp"] += 1

            # matcher-only ceiling: GT pairs whose struct AND label both detected
            for e in labelled:
                sd = any(_iou(e["struct_bbox"], d) >= 0.5 for d in det_s)
                ld = any(_iou(e["label_bbox"], d) >= 0.5 for d in det_l)
                if not (sd and ld):
                    continue
                ceil[mname]["n"] += 1
                ok = any(
                    _iou(p.structure.bbox.as_list(), e["struct_bbox"]) >= 0.5
                    and _inside(_cent(e["label_bbox"]), p.label.bbox.as_list())
                    for p in pairs
                )
                ceil[mname]["tp"] += int(ok)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(files)} pages")

    print(f"\nGT pairs={gt_pairs_total}")
    print(f"DETECTION recall:  structures  IoU>=.5 {s_det5/s_gt:.1%}  IoU>=.3 {s_det3/s_gt:.1%}")
    print(f"                   LABELS      IoU>=.5 {l_det5/l_gt:.1%}  IoU>=.3 {l_det3/l_gt:.1%}")
    print()
    for mname in ("Hungarian", "LPS"):
        print(f"=== {mname} ===")
        for c in ("strict", "iou0.3", "centroid"):
            tp = crit[c][mname]["tp"]; npred = crit[c][mname]["np"]
            P = tp / npred if npred else 0.0
            R = tp / gt_pairs_total if gt_pairs_total else 0.0
            F = 2 * P * R / (P + R) if (P + R) else 0.0
            print(f"  {c:>9}: P {P:.3f}  R {R:.3f}  F1 {F:.3f}")
        cc = ceil[mname]
        print(f"  matcher-only ceiling (both detected): {cc['tp']}/{cc['n']} = {cc['tp']/max(cc['n'],1):.1%}\n")


if __name__ == "__main__":
    main()
