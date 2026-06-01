"""Run the LPS scorer on GT boxes across ALL real pages and inspect the scores.

GT boxes (not detections) isolate the matcher from detection noise: this is the
purest view of "what does the scorer think of real (struct,label) pairs".

For every labelled structure we record:
  - true_score      : scorer prob for the (struct, its true label) pair
  - runner_up       : best scorer prob for (struct, a WRONG label) on the page
  - margin          : true_score - runner_up   (>0 => correct independent ranking)
For every UNLABELLED structure (30% of real!) we record:
  - max score it gets to ANY label (false-pairing risk; should be low)

Then we decompose Hungarian-on-scores assignment outcomes for labelled structs:
  - correct            : matched to true label AND score >= min_score
  - threshold_miss     : matched to true label BUT score < min_score (dropped)
  - ranking_error      : matched to a DIFFERENT label (cost matrix wrong)
and compare to distance-Hungarian on the identical boxes.

Split by test/val/train (train = seen during fine-tuning).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from structflo.cser.lps.matcher import LearnedMatcher
from structflo.cser.pipeline.models import Detection


def _centroid(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _hist(vals, edges):
    vals = np.asarray(vals)
    if vals.size == 0:
        return [0] * (len(edges) - 1)
    h, _ = np.histogram(vals, bins=edges)
    return h.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data"))
    ap.add_argument("--split-manifest", type=Path, default=Path("data/finetune/real_split.json"))
    ap.add_argument("--lps", type=Path, default=Path("runs/lps_finetune/best.pt"))
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(args.split_manifest.read_text())
    stem2split = {}
    for sp in ("test", "val", "train"):
        for stem in manifest.get(sp, []):
            stem2split[stem] = sp

    matcher = LearnedMatcher(weights=str(args.lps), min_score=args.min_score)

    gt_dir = args.src / "ground_truth"
    img_dir = args.src / "images"
    files = sorted(gt_dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    # accumulators per split (+ "all")
    splits = ("test", "val", "train", "all")
    acc = {
        s: {
            "true_scores": [],
            "margins": [],
            "unlabelled_max": [],
            "n_labelled": 0,
            "argmax_correct": 0,  # independent: argmax over labels == true
            "lps_correct": 0,  # Hungarian-on-scores, matched true & >=min_score
            "thresh_miss": 0,  # Hungarian matched true label but score<min_score
            "rank_err": 0,  # Hungarian matched a different label
            "dist_correct": 0,  # distance-Hungarian matched true
            "n_unlabelled": 0,
            "n_pages": 0,
        }
        for s in splits
    }

    skipped = 0
    for k, gtf in enumerate(files):
        stem = gtf.stem
        split = stem2split.get(stem)
        if split is None:
            skipped += 1
            continue
        entries = json.loads(gtf.read_text())
        ip = img_dir / f"{stem}.png"
        if not ip.exists():
            ip = img_dir / f"{stem}.jpg"
        if not ip.exists():
            skipped += 1
            continue

        struct_boxes = [e["struct_bbox"] for e in entries]
        # labels with their owning struct index (entry index)
        label_owner, label_boxes = [], []
        for i, e in enumerate(entries):
            if e.get("label_bbox") is not None:
                label_owner.append(i)
                label_boxes.append(e["label_bbox"])
        if not struct_boxes or not label_boxes:
            continue

        img = np.array(Image.open(ip).convert("L"))
        h, w = img.shape[:2]

        structs = [Detection.from_dict({"bbox": b, "conf": 1.0, "class_id": 0}) for b in struct_boxes]
        labels = [Detection.from_dict({"bbox": b, "conf": 1.0, "class_id": 1}) for b in label_boxes]

        # full score matrix [n_struct, n_label]
        S = matcher._score_matrix(structs, labels, img, float(w), float(h))

        owner_to_col = {label_owner[j]: j for j in range(len(label_boxes))}

        # Hungarian on scores (LPS) and on distance (baseline)
        r_lps, c_lps = linear_sum_assignment(1.0 - S)
        lps_assign = {int(r): int(c) for r, c in zip(r_lps, c_lps)}

        dist = np.zeros((len(structs), len(labels)))
        for si, sb in enumerate(struct_boxes):
            scx, scy = _centroid(sb)
            for lj, lb in enumerate(label_boxes):
                lcx, lcy = _centroid(lb)
                dist[si, lj] = ((scx - lcx) ** 2 + (scy - lcy) ** 2) ** 0.5
        r_d, c_d = linear_sum_assignment(dist)
        dist_assign = {int(r): int(c) for r, c in zip(r_d, c_d)}

        for tgt in (split, "all"):
            a = acc[tgt]
            a["n_pages"] += 1
            for si in range(len(structs)):
                if si in owner_to_col:  # labelled structure
                    a["n_labelled"] += 1
                    jt = owner_to_col[si]
                    ts = float(S[si, jt])
                    a["true_scores"].append(ts)
                    row = S[si].copy()
                    if len(labels) > 1:
                        runner = float(np.max(np.delete(row, jt)))
                    else:
                        runner = 0.0
                    a["margins"].append(ts - runner)
                    if int(np.argmax(row)) == jt:
                        a["argmax_correct"] += 1
                    # LPS Hungarian outcome
                    pred = lps_assign.get(si, None)
                    if pred == jt:
                        if ts >= args.min_score:
                            a["lps_correct"] += 1
                        else:
                            a["thresh_miss"] += 1
                    else:
                        a["rank_err"] += 1
                    # distance baseline
                    if dist_assign.get(si, None) == jt:
                        a["dist_correct"] += 1
                else:  # unlabelled structure
                    a["n_unlabelled"] += 1
                    a["unlabelled_max"].append(float(np.max(S[si])))

        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(files)} pages")

    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\nLPS weights: {args.lps}   min_score={args.min_score}   skipped={skipped}\n")
    for s in splits:
        a = acc[s]
        n = a["n_labelled"]
        if n == 0:
            continue
        ts = np.array(a["true_scores"])
        mg = np.array(a["margins"])
        print(f"===== {s.upper()}  ({a['n_pages']} pages, {n} labelled structs, {a['n_unlabelled']} unlabelled) =====")
        print(f"  TRUE-PAIR score:  mean {ts.mean():.3f}  median {np.median(ts):.3f}  "
              f"p10 {np.percentile(ts,10):.3f}  p25 {np.percentile(ts,25):.3f}  <0.5: {(ts<0.5).mean():.1%}")
        print(f"  MARGIN (true-runner): mean {mg.mean():.3f}  median {np.median(mg):.3f}  <=0 (mis-ranked): {(mg<=0).mean():.1%}")
        print(f"  true-score hist [0..1 /0.1]: {_hist(ts, edges)}")
        print(f"  ASSIGNMENT (labelled structs):")
        print(f"    independent argmax correct : {a['argmax_correct']/n:.1%}")
        print(f"    LPS Hungarian correct      : {a['lps_correct']/n:.1%}  ({a['lps_correct']}/{n})")
        print(f"      threshold-miss (true but <{args.min_score}): {a['thresh_miss']/n:.1%}  ({a['thresh_miss']})")
        print(f"      ranking-error  (wrong label)        : {a['rank_err']/n:.1%}  ({a['rank_err']})")
        print(f"    distance-Hungarian correct : {a['dist_correct']/n:.1%}  ({a['dist_correct']}/{n})")
        if a["unlabelled_max"]:
            um = np.array(a["unlabelled_max"])
            print(f"  UNLABELLED-struct max score: mean {um.mean():.3f}  median {np.median(um):.3f}  "
                  f">=0.5 (would false-pair): {(um>=0.5).mean():.1%}")
            print(f"    unlabelled max-score hist: {_hist(um, edges)}")
        print()


if __name__ == "__main__":
    main()
