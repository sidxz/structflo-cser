"""Build detection-based training data for the relational matcher.

Runs the YOLO detector (full-image @ imgsz) over the GT train/val pages, then
builds per-page samples whose nodes are the *detected* boxes (real localisation
noise, false-positive structures, missed labels, real confidences) and whose
targets are derived from the ground truth:

  detected struct → best GT struct (IoU>=0.5):
      - no GT match           → dustbin   (false-positive structure)
      - GT struct unlabelled  → dustbin
      - GT label not detected → dustbin   (missed label)
      - else                  → the detected label best matching the GT label

This makes the training distribution match inference, fixing the over-rejection
seen when a GT-trained matcher meets noisy detection boxes.

Output: data/finetune/relmatch_det/{train,val}/<stem>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _best(box, cands, thr):
    bi, bv = -1, thr
    for i, c in enumerate(cands):
        v = _iou(box, c)
        if v >= bv:
            bi, bv = i, v
    return bi


def build_split(model, gt_dir, img_dir, out_dir, conf, imgsz):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(gt_dir.glob("*.json"))
    n_pages = n_fp_struct = n_det_struct = n_missed_label = n_dustbin = n_real = 0
    written = 0
    for gtf in files:
        stem = gtf.stem
        ip = img_dir / f"{stem}.png"
        if not ip.exists():
            ip = img_dir / f"{stem}.jpg"
        if not ip.exists():
            continue
        entries = json.loads(gtf.read_text())
        gt_structs = [e["struct_bbox"] for e in entries]
        gt_label = [e.get("label_bbox") for e in entries]  # None if unlabelled

        img_rgb = np.array(Image.open(ip).convert("L").convert("RGB"))
        page_h, page_w = img_rgb.shape[:2]
        res = model(img_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]

        d_struct_boxes, d_struct_conf = [], []
        d_label_boxes, d_label_conf = [], []
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0])
            cf = float(box.conf[0])
            if cls == 0:
                d_struct_boxes.append([x1, y1, x2, y2])
                d_struct_conf.append(cf)
            else:
                d_label_boxes.append([x1, y1, x2, y2])
                d_label_conf.append(cf)

        if not d_struct_boxes or not d_label_boxes:
            continue

        n_labels = len(d_label_boxes)
        targets = []
        for ds in d_struct_boxes:
            n_det_struct += 1
            gi = _best(ds, gt_structs, 0.5)
            if gi < 0:
                targets.append(n_labels)  # FP struct → dustbin
                n_fp_struct += 1
                n_dustbin += 1
                continue
            tl = gt_label[gi]
            if tl is None:
                targets.append(n_labels)  # unlabelled GT struct
                n_dustbin += 1
                continue
            lk = _best(tl, d_label_boxes, 0.5)
            if lk < 0:
                targets.append(n_labels)  # true label not detected
                n_missed_label += 1
                n_dustbin += 1
            else:
                targets.append(lk)
                n_real += 1

        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "page_w": float(page_w),
                    "page_h": float(page_h),
                    "struct_boxes": d_struct_boxes,
                    "struct_conf": d_struct_conf,
                    "label_boxes": d_label_boxes,
                    "label_conf": d_label_conf,
                    "targets": targets,
                }
            )
        )
        written += 1
        n_pages += 1

    print(
        f"  {out_dir.name}: {written} pages written | det structs {n_det_struct} "
        f"(FP {n_fp_struct}, missed-label {n_missed_label}, real-pair {n_real}, "
        f"→dustbin {n_dustbin})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/finetune/lps"))
    ap.add_argument("--out", type=Path, default=Path("data/finetune/relmatch_det"))
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_3way/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.detector))
    for split in ("train", "val"):
        print(f"[prep] {split} …")
        build_split(
            model,
            args.src / split / "ground_truth",
            args.src / split / "images",
            args.out / split,
            args.conf,
            args.imgsz,
        )
    print(f"[prep] done → {args.out}")


if __name__ == "__main__":
    main()
