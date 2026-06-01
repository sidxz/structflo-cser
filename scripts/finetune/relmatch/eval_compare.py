"""Compare all three matchers on real_test: Hungarian vs LPS vs Relational.

Part A — GT-box matching (isolates matcher quality, no detection noise):
  feed ground-truth struct+label boxes to each matcher; report
    - assignment accuracy over LABELLED structures (matched to true label)
    - rejection rate over UNLABELLED structures (correctly left unmatched)
    - pair precision (of predicted pairs, fraction that are true pairs)

Part B — end-to-end with full@1280 detections (deployment view):
  detect → each matcher → pairing F1 under the label-centroid criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.lps.matcher import LearnedMatcher
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.models import Detection
from structflo.cser.relmatch.matcher import RelationalMatcher


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


def _key(b):
    return tuple(round(float(v), 2) for v in b)


def _match(matcher, dets, img, name):
    return matcher.match(dets, image=img) if name != "Hungarian" else matcher.match(dets)


def part_a(matchers, gt_dir, img_dir):
    """GT-box matching: per-structure assignment accuracy + rejection."""
    acc = {n: dict(lab=0, lab_ok=0, unlab=0, unlab_ok=0, npred=0, pred_ok=0) for n in matchers}
    files = sorted(gt_dir.glob("*.json"))
    for gtf in files:
        entries = json.loads(gtf.read_text())
        ip = img_dir / f"{gtf.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gtf.stem}.jpg"
        if not ip.exists():
            continue
        img = np.array(Image.open(ip).convert("L"))

        # true label box per struct (by key), or None
        true_label = {}
        dets = []
        for e in entries:
            dets.append(Detection.from_dict({"bbox": e["struct_bbox"], "conf": 1.0, "class_id": 0}))
            true_label[_key(e["struct_bbox"])] = (
                _key(e["label_bbox"]) if e.get("label_bbox") is not None else None
            )
            if e.get("label_bbox") is not None:
                dets.append(Detection.from_dict({"bbox": e["label_bbox"], "conf": 1.0, "class_id": 1}))
        if not any(d.class_id == 1 for d in dets):
            continue

        for name, m in matchers.items():
            pairs = _match(m, dets, img, name)
            matched = {_key(p.structure.bbox.as_list()): _key(p.label.bbox.as_list()) for p in pairs}
            a = acc[name]
            a["npred"] += len(pairs)
            for sk, tl in true_label.items():
                if tl is not None:
                    a["lab"] += 1
                    if matched.get(sk) == tl:
                        a["lab_ok"] += 1
                else:
                    a["unlab"] += 1
                    if sk not in matched:
                        a["unlab_ok"] += 1
            for sk, lk in matched.items():
                if true_label.get(sk) == lk:
                    a["pred_ok"] += 1
    return acc


def part_b(matchers, model, gt_dir, img_dir, conf, imgsz):
    """End-to-end full@imgsz detection → pairing F1 (label-centroid criterion)."""
    stat = {n: dict(tp=0, npred=0) for n in matchers}
    gt_pairs = 0
    files = sorted(gt_dir.glob("*.json"))
    for gtf in files:
        entries = json.loads(gtf.read_text())
        ip = img_dir / f"{gtf.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gtf.stem}.jpg"
        if not ip.exists():
            continue
        img_rgb = np.array(Image.open(ip).convert("L").convert("RGB"))
        res = model(img_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append(Detection.from_dict({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                             "conf": float(box.conf[0]), "class_id": int(box.cls[0])}))
        labelled = [e for e in entries if e.get("label_bbox") is not None]
        gt_pairs += len(labelled)
        img_l = np.array(Image.open(ip).convert("L"))
        for name, m in matchers.items():
            pairs = _match(m, dets, img_l, name)
            stat[name]["npred"] += len(pairs)
            for p in pairs:
                ps, pl = p.structure.bbox.as_list(), p.label.bbox.as_list()
                bi, bv = -1, 0.0
                for i, e in enumerate(entries):
                    v = _iou(ps, e["struct_bbox"])
                    if v > bv:
                        bi, bv = i, v
                if bi < 0:
                    continue
                gl = entries[bi].get("label_bbox")
                if gl is None:
                    continue
                if bv >= 0.5 and _inside(_cent(gl), pl):
                    stat[name]["tp"] += 1
    return stat, gt_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/finetune/lps/real_test"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--lps", type=Path, default=Path("runs/lps_finetune/best.pt"))
    ap.add_argument("--relmatch", type=Path, default=Path("runs/relmatch/best.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    matchers = {
        "Hungarian": HungarianMatcher(),
        "LPS": LearnedMatcher(weights=str(args.lps), min_score=0.5),
        "Relational": RelationalMatcher(weights=str(args.relmatch)),
    }
    gt_dir = args.data_dir / "ground_truth"
    img_dir = args.data_dir / "images"

    print("=" * 64)
    print("PART A — GT-box matching (clean; isolates matcher quality)")
    print("=" * 64)
    a = part_a(matchers, gt_dir, img_dir)
    print(f"  {'matcher':>11} | {'assign acc':>10} {'reject':>8} {'pair prec':>10}")
    for n, s in a.items():
        assign = s["lab_ok"] / max(s["lab"], 1)
        reject = s["unlab_ok"] / max(s["unlab"], 1)
        prec = s["pred_ok"] / max(s["npred"], 1)
        print(f"  {n:>11} | {assign:>10.1%} {reject:>8.1%} {prec:>10.1%}")
    print(f"  (labelled structs={a['Hungarian']['lab']}, unlabelled={a['Hungarian']['unlab']})")

    if args.skip_e2e:
        return
    print()
    print("=" * 64)
    print(f"PART B — end-to-end full@{args.imgsz} detection → pairing F1 (centroid)")
    print("=" * 64)
    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    stat, gt_pairs = part_b(matchers, model, gt_dir, img_dir, args.conf, args.imgsz)
    print(f"  GT pairs = {gt_pairs}")
    print(f"  {'matcher':>11} | {'P':>7} {'R':>7} {'F1':>7}")
    for n, s in stat.items():
        P = s["tp"] / max(s["npred"], 1)
        R = s["tp"] / max(gt_pairs, 1)
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        print(f"  {n:>11} | {P:>7.3f} {R:>7.3f} {F:>7.3f}")


if __name__ == "__main__":
    main()
