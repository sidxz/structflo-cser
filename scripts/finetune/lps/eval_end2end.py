"""End-to-end matcher evaluation on REAL YOLO detections (no ground-truth boxes).

This is the deployment-realistic test. Detection runs with the fine-tuned YOLO
(tiled), so the detections include localisation error, missed boxes, and — the
case of interest — false-positive structure detections that may land next to a
label. Hungarian must force-match those; a visual matcher (LPS) could reject them.

Both matchers run on the SAME detections, so the difference isolates
matching + rejection on noisy real detections. Pairs are scored against GT by
IoU >= 0.5 on both the structure and the label box.

Usage:
    uv run python scripts/finetune/lps/eval_end2end.py \
        --data-dir data/finetune/lps/real_test \
        --detector runs/labels_detect/finetune_3way/weights/best.pt \
        --lps runs/lps_finetune/best.pt
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


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _best_gt(struct_bbox: list[float], entries: list[dict]) -> tuple[int, float]:
    best_i, best_v = -1, 0.0
    for i, e in enumerate(entries):
        v = _iou(struct_bbox, e["struct_bbox"])
        if v > best_v:
            best_i, best_v = i, v
    return best_i, best_v


def _score(pairs, entries) -> int:
    """Count true-positive pairs: struct IoU>=0.5 to a labelled GT struct AND label IoU>=0.5."""
    tp = 0
    for p in pairs:
        i, v = _best_gt(p.structure.bbox.as_list(), entries)
        if v < 0.5:
            continue  # spurious structure detection
        e = entries[i]
        if e.get("label_bbox") is None:
            continue  # matched a distractor / unlabelled structure
        if _iou(p.label.bbox.as_list(), e["label_bbox"]) >= 0.5:
            tp += 1
    return tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/finetune/lps/real_test"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--lps", type=Path, default=Path("runs/lps_finetune/best.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    hung = HungarianMatcher()
    lps = LearnedMatcher(weights=str(args.lps), min_score=0.0, device=args.device)

    gt_dir = args.data_dir / "ground_truth"
    img_dir = args.data_dir / "images"

    h_assign, l_assign, meta = [], [], []
    det_struct_tp = det_struct_fp = gt_struct_total = 0

    files = sorted(gt_dir.glob("*.json"))
    for k, gt in enumerate(files):
        entries = json.loads(gt.read_text())
        if not isinstance(entries, list) or not entries:
            continue
        ip = img_dir / f"{gt.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gt.stem}.jpg"
        if not ip.exists():
            continue

        gray_rgb = Image.open(ip).convert("L").convert("RGB")
        img_np = np.array(gray_rgb)
        raw = detect_tiled(model, img_np, tile_size=1536, conf=args.conf)
        dets = [Detection.from_dict(d) for d in raw]

        gray = np.array(Image.open(ip).convert("L"))
        h_assign.append(hung.match(dets))
        l_assign.append(lps.match(dets, image=gray))

        gt_pos = sum(1 for e in entries if e.get("label_bbox") is not None)
        meta.append({"entries": entries, "gt_pos": gt_pos})

        # detection context: struct detection recall / false positives
        det_structs = [d.bbox.as_list() for d in dets if d.class_id == 0]
        gt_struct_total += len(entries)
        used = set()
        for ds in det_structs:
            i, v = _best_gt(ds, entries)
            if v >= 0.5 and i not in used:
                det_struct_tp += 1
                used.add(i)
            else:
                det_struct_fp += 1
        if (k + 1) % 25 == 0:
            print(f"  detected {k + 1}/{len(files)} pages")

    print(f"\ndata: {args.data_dir}   pages={len(meta)}")
    print(f"detector: {args.detector.name}")
    gt_pairs_total = sum(m["gt_pos"] for m in meta)
    print(f"GT structures={gt_struct_total}  GT pairs(labelled)={gt_pairs_total}")
    print(f"structure detection: TP(IoU>=.5)={det_struct_tp} (recall {det_struct_tp / max(gt_struct_total,1):.1%})  "
          f"false-positive struct dets={det_struct_fp}\n")

    def evaluate(assigns, thresholds, higher_keeps):
        rows = []
        for t in thresholds:
            TP = NP = 0
            for a, m in zip(assigns, meta):
                if higher_keeps:
                    pairs = [p for p in a if p.match_confidence >= t]
                else:
                    pairs = [p for p in a if p.match_distance <= t]
                TP += _score(pairs, m["entries"])
                NP += len(pairs)
            prec = TP / NP if NP else 1.0
            rec = TP / gt_pairs_total if gt_pairs_total else 1.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rows.append((t, prec, rec, f1, NP))
        return rows

    inf = float("inf")
    h_rows = evaluate(h_assign, [inf, 1500, 1000, 750, 500, 350, 250], higher_keeps=False)
    l_rows = evaluate(l_assign, [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], higher_keeps=True)

    def show(name, rows, lbl):
        print(f"=== {name} ===")
        print(f"{lbl:>10} {'Prec':>7} {'Recall':>7} {'F1':>7} {'#pairs':>7}")
        best = max(rows, key=lambda r: r[3])
        for t, p, r, f, n in rows:
            ts = "inf" if t == inf else (f"{t:.2f}" if t < 5 else f"{t:.0f}")
            star = "  <-- best F1" if (t, p, r, f, n) == best else ""
            print(f"{ts:>10} {p:>7.3f} {r:>7.3f} {f:>7.3f} {n:>7}{star}")
        print()

    show("HUNGARIAN (sweep max distance, px)", h_rows, "max_dist")
    show("LEARNED PAIR SCORER (sweep min_score)", l_rows, "min_score")


if __name__ == "__main__":
    main()
