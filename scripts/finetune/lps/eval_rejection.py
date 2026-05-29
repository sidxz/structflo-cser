"""Rejection-aware matcher evaluation.

Unlike sf-eval-lps (which filters out unlabelled structures and scores only the
matching among labelled ones), this feeds EVERY structure — including the
unlabelled / distractor structures — plus all labels to each matcher, then scores:

  * pair precision / recall / F1   (a fabricated pair for a distractor, or a
                                     wrong label, costs precision; an orphaned
                                     real structure costs recall)
  * distractor rejection recall    (fraction of unlabelled structures correctly
                                     left unmatched)

Detections are built from ground-truth boxes (perfect localisation) so this
isolates the *matching + rejection* decision from detection error.

Hungarian rejects only via a centroid-distance gate (and the count-mismatch
leftover); the LPS rejects via min_score. We sweep each threshold so the
comparison is fair and report the full precision/recall/reject curve.

Usage:
    uv run python scripts/finetune/lps/eval_rejection.py \
        --data-dir data/finetune/lps/real_test \
        --weights runs/lps_finetune/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.lps.matcher import LearnedMatcher
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.models import BBox, Detection

_TOL = 2.0


def _cent(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _close(a: list[float], b: list[float], tol: float = _TOL) -> bool:
    ax, ay = _cent(a)
    bx, by = _cent(b)
    return abs(ax - bx) < tol and abs(ay - by) < tol


def _load_pages(data_dir: Path) -> list[dict]:
    """Return per-page records: structs (all), labels (labelled only), gt map, image."""
    gt_dir = data_dir / "ground_truth"
    img_dir = data_dir / "images"
    pages = []
    for gt in sorted(gt_dir.glob("*.json")):
        entries = json.loads(gt.read_text())
        if not isinstance(entries, list) or not entries:
            continue
        ip = img_dir / f"{gt.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{gt.stem}.jpg"
        if not ip.exists():
            continue
        pages.append({"stem": gt.stem, "entries": entries, "img": ip})
    return pages


def _score_pairs(pairs, entries) -> tuple[int, int, set]:
    """Return (true_positives, fabricated/wrong false-positives, matched struct-centroids)."""
    tp = 0
    matched = set()
    for p in pairs:
        s_bbox = p.structure.bbox.as_list()
        l_bbox = p.label.bbox.as_list()
        e = next((e for e in entries if _close(e["struct_bbox"], s_bbox)), None)
        matched.add(_cent(s_bbox))
        if e is None:
            continue
        if e.get("label_bbox") is not None and _close(e["label_bbox"], l_bbox):
            tp += 1
    return tp, len(pairs) - tp, matched


def _metrics(tp, n_pred, gt_pos, n_distract, rejected) -> dict:
    prec = tp / n_pred if n_pred else 1.0
    rec = tp / gt_pos if gt_pos else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rej = rejected / n_distract if n_distract else float("nan")
    return {"P": prec, "R": rec, "F1": f1, "reject": rej}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/finetune/lps/real_test"))
    ap.add_argument("--weights", type=Path, default=Path("runs/lps_finetune/best.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pages = _load_pages(args.data_dir)
    n_struct = sum(len(p["entries"]) for p in pages)
    n_lab = sum(sum(1 for e in p["entries"] if e.get("label_bbox") is not None) for p in pages)
    n_dis = n_struct - n_lab
    print(f"data: {args.data_dir}")
    print(f"pages={len(pages)}  structures={n_struct}  labelled={n_lab}  "
          f"distractors(unlabelled)={n_dis} ({100 * n_dis / max(n_struct, 1):.1f}%)\n")

    hung = HungarianMatcher()  # no gate; we threshold on returned distance
    lps = LearnedMatcher(weights=str(args.weights), min_score=0.0, device=args.device)

    # Precompute per-page assignments once (both matchers), with raw scores/distances.
    h_assign, l_assign, meta = [], [], []
    for p in pages:
        entries = p["entries"]
        det = []
        for e in entries:
            det.append(Detection(bbox=BBox(*e["struct_bbox"]), conf=1.0, class_id=0))
        for e in entries:
            if e.get("label_bbox") is not None:
                det.append(Detection(bbox=BBox(*e["label_bbox"]), conf=1.0, class_id=1))
        img = np.array(Image.open(p["img"]).convert("L"))
        h_assign.append(hung.match(det))                 # match_distance set
        l_assign.append(lps.match(det, image=img))       # match_confidence set (min_score=0)
        gt_pos = sum(1 for e in entries if e.get("label_bbox") is not None)
        n_d = len(entries) - gt_pos
        distract_cents = {_cent(e["struct_bbox"]) for e in entries if e.get("label_bbox") is None}
        meta.append({"entries": entries, "gt_pos": gt_pos, "n_d": n_d, "dcents": distract_cents})

    def evaluate(assigns, thresholds, key, higher_keeps):
        rows = []
        for t in thresholds:
            TP = NP = GT = ND = REJ = 0
            for a, m in zip(assigns, meta):
                if higher_keeps:  # LPS: keep pairs with confidence >= t
                    pairs = [p for p in a if p.match_confidence >= t]
                else:             # Hungarian: keep pairs with distance <= t
                    pairs = [p for p in a if p.match_distance <= t]
                tp, fp, matched = _score_pairs(pairs, m["entries"])
                TP += tp
                NP += len(pairs)
                GT += m["gt_pos"]
                ND += m["n_d"]
                REJ += sum(1 for c in m["dcents"] if c not in matched)
            rows.append((t, _metrics(TP, NP, GT, ND, REJ)))
        return rows

    inf = float("inf")
    h_rows = evaluate(h_assign, [inf, 2000, 1500, 1000, 750, 500, 350, 250, 150], "dist", higher_keeps=False)
    l_rows = evaluate(l_assign, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "score", higher_keeps=True)

    def show(name, rows, thr_label):
        print(f"=== {name} ===")
        print(f"{thr_label:>10} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Reject%':>8}")
        best = max(rows, key=lambda r: r[1]["F1"])
        for t, mx in rows:
            ts = "inf" if t == inf else (f"{t:.1f}" if isinstance(t, float) and t < 5 else f"{t:.0f}")
            star = "  <-- best F1" if (t, mx) == best else ""
            rej = "n/a" if mx["reject"] != mx["reject"] else f"{100 * mx['reject']:.0f}%"
            print(f"{ts:>10} {mx['P']:>7.3f} {mx['R']:>7.3f} {mx['F1']:>7.3f} {rej:>8}{star}")
        print()

    show("HUNGARIAN (sweep max centroid distance, px)", h_rows, "max_dist")
    show("LEARNED PAIR SCORER (sweep min_score)", l_rows, "min_score")


if __name__ == "__main__":
    main()
