"""3-way matcher comparison across ALL real pages, split by test/val/train.

CAVEAT: LPS and the relational matcher were TRAINED on the train split (and the
relational selected on val); only the TEST rows are clean held-out numbers.
Train/val rows show fit/generalisation, not unbiased performance.

Part A — GT-box matching (clean; isolates matcher quality).
Part B — end-to-end full@1280 detection → pairing F1 (label-centroid criterion).
Relational uses the det-trained model; margin 0 for Part A, the val-tuned
margin (default 2.0) for Part B.
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

SPLITS = ("test", "val", "train", "all")


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


def _do_match(m, dets, img, name):
    return m.match(dets, image=img) if name != "Hungarian" else m.match(dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data"))
    ap.add_argument("--manifest", type=Path, default=Path("data/finetune/real_split.json"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--lps", type=Path, default=Path("runs/lps_finetune/best.pt"))
    ap.add_argument("--relmatch", type=Path, default=Path("runs/relmatch_det/best.pt"))
    ap.add_argument("--margin", type=float, default=2.0, help="relational dustbin margin for Part B")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    stem2split = {}
    for sp in ("test", "val", "train"):
        for stem in manifest.get(sp, []):
            stem2split[stem] = sp

    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    matchers = {
        "Hungarian": HungarianMatcher(),
        "LPS": LearnedMatcher(weights=str(args.lps), min_score=0.5),
        "Relational": RelationalMatcher(weights=str(args.relmatch)),
    }

    # accumulators
    A = {n: {s: dict(lab=0, lab_ok=0, unlab=0, unlab_ok=0, npred=0, pred_ok=0) for s in SPLITS} for n in matchers}
    B = {n: {s: dict(tp=0, npred=0) for s in SPLITS} for n in matchers}
    gt_pairs = {s: 0 for s in SPLITS}

    gt_dir = args.src / "ground_truth"
    img_dir = args.src / "images"
    files = sorted(gt_dir.glob("*.json"))
    for k, gtf in enumerate(files):
        split = stem2split.get(gtf.stem)
        if split is None:
            continue
        ip = img_dir / f"{gtf.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gtf.stem}.jpg"
        if not ip.exists():
            continue
        entries = json.loads(gtf.read_text())

        # ---- Part A inputs: GT boxes ----
        true_label = {}
        gt_dets = []
        for e in entries:
            gt_dets.append(Detection.from_dict({"bbox": e["struct_bbox"], "conf": 1.0, "class_id": 0}))
            true_label[_key(e["struct_bbox"])] = (
                _key(e["label_bbox"]) if e.get("label_bbox") is not None else None
            )
            if e.get("label_bbox") is not None:
                gt_dets.append(Detection.from_dict({"bbox": e["label_bbox"], "conf": 1.0, "class_id": 1}))
        has_label = any(d.class_id == 1 for d in gt_dets)

        img_l = np.array(Image.open(ip).convert("L"))

        # ---- Part B inputs: detections (one YOLO pass) ----
        img_rgb = np.array(Image.open(ip).convert("L").convert("RGB"))
        res = model(img_rgb, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        det_dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det_dets.append(Detection.from_dict({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                                 "conf": float(box.conf[0]), "class_id": int(box.cls[0])}))
        labelled = [e for e in entries if e.get("label_bbox") is not None]

        for tgt in (split, "all"):
            gt_pairs[tgt] += len(labelled)

        for name, m in matchers.items():
            # Part A (GT boxes); relational strict (margin 0)
            if has_label:
                if name == "Relational":
                    m.dustbin_margin = 0.0
                pairs = _do_match(m, gt_dets, img_l, name)
                matched = {_key(p.structure.bbox.as_list()): _key(p.label.bbox.as_list()) for p in pairs}
                for tgt in (split, "all"):
                    a = A[name][tgt]
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

            # Part B (detections); relational tuned margin
            if name == "Relational":
                m.dustbin_margin = args.margin
            pairs = _do_match(m, det_dets, img_l, name)
            for tgt in (split, "all"):
                B[name][tgt]["npred"] += len(pairs)
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
                    for tgt in (split, "all"):
                        B[name][tgt]["tp"] += 1
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(files)} pages")

    print("\n" + "=" * 70)
    print("PART A — GT-box matching (clean matcher quality)   [assign / reject / prec]")
    print("=" * 70)
    for s in SPLITS:
        lab = A["Hungarian"][s]["lab"]
        unlab = A["Hungarian"][s]["unlab"]
        print(f"\n  [{s.upper()}]  labelled={lab}  unlabelled={unlab}")
        print(f"    {'matcher':>11} | {'assign':>7} {'reject':>7} {'prec':>7}")
        for n in matchers:
            a = A[n][s]
            assign = a["lab_ok"] / max(a["lab"], 1)
            reject = a["unlab_ok"] / max(a["unlab"], 1)
            prec = a["pred_ok"] / max(a["npred"], 1)
            print(f"    {n:>11} | {assign:>7.1%} {reject:>7.1%} {prec:>7.1%}")

    print("\n" + "=" * 70)
    print(f"PART B — end-to-end full@{args.imgsz} → pairing F1 (centroid; Relational margin={args.margin})")
    print("=" * 70)
    for s in SPLITS:
        print(f"\n  [{s.upper()}]  GT pairs={gt_pairs[s]}")
        print(f"    {'matcher':>11} | {'P':>7} {'R':>7} {'F1':>7}")
        for n in matchers:
            b = B[n][s]
            P = b["tp"] / max(b["npred"], 1)
            R = b["tp"] / max(gt_pairs[s], 1)
            F = 2 * P * R / (P + R) if (P + R) else 0.0
            print(f"    {n:>11} | {P:>7.3f} {R:>7.3f} {F:>7.3f}")


if __name__ == "__main__":
    main()
