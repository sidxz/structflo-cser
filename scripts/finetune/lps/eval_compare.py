"""Compare baseline vs fine-tuned LPS.

Evaluates both checkpoints on three val sets:
  1. Real test (held out) — the paper number; shares pages with the detector test
  2. Real val (selection set) — drove best.pt selection (reference only)
  3. Original synthetic val — regression check

Prints a summary table with deltas at the end.

Usage:
    uv run python scripts/finetune/lps/eval_compare.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from structflo.cser.lps.dataset import LPSDataset, PageGroupSampler
from structflo.cser.lps.scorer import PairScorer

PROJECT_ROOT = Path(__file__).parents[3]

FINETUNE_DATA = PROJECT_ROOT / "data" / "finetune" / "lps"
SYNTH_DATA = PROJECT_ROOT / "data" / "generated"

BASELINE = PROJECT_ROOT / "runs" / "lps" / "best.pt"
FINETUNED = PROJECT_ROOT / "runs" / "lps_finetune" / "best.pt"


@torch.no_grad()
def _evaluate(weights: Path, val_ds: LPSDataset) -> dict[str, float] | None:
    if not weights.exists():
        print(f"  weights not found: {weights}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PairScorer()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    sampler = PageGroupSampler(val_ds._path_idx, shuffle=False, seed=42)
    loader = DataLoader(
        val_ds,
        batch_size=512,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )

    pw = val_ds.pos_weight()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    total_loss = 0.0
    correct = 0
    n = 0

    for batch in loader:
        geom = batch["geom"].to(device)
        sc = batch["struct_crop"].to(device)
        lc = batch["label_crop"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        logits = model(sc, lc, geom)
        loss = criterion(logits, target)

        bs = target.size(0)
        total_loss += loss.item() * bs
        preds = (logits.sigmoid() >= 0.5).float()
        correct += (preds == target).sum().item()
        n += bs

    return {
        "Accuracy": correct / max(n, 1),
        "Loss": total_loss / max(n, 1),
        "Samples": n,
    }


def _delta_str(baseline: float, finetuned: float, lower_is_better: bool = False) -> str:
    d = finetuned - baseline
    sign = "+" if d >= 0 else ""
    if lower_is_better:
        arrow = "v" if d < 0 else "^"
    else:
        arrow = "^" if d > 0 else "v"
    return f"{sign}{d:.4f} {arrow}"


def _print_comparison(
    label: str, baseline: dict[str, float], finetuned: dict[str, float]
) -> None:
    print(f"\n  {label}")
    print(f"  Samples: {int(baseline['Samples'])}")
    print(f"  {'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>14}")
    print(f"  {'-' * 50}")

    b, f = baseline["Accuracy"], finetuned["Accuracy"]
    print(f"  {'Accuracy':<12} {b:>9.2%} {f:>11.2%} {_delta_str(b, f):>14}")

    b, f = baseline["Loss"], finetuned["Loss"]
    print(
        f"  {'Loss':<12} {b:>10.4f} {f:>12.4f} {_delta_str(b, f, lower_is_better=True):>14}"
    )


def _build_val_ds(data_dir: Path) -> LPSDataset | None:
    gt_dir = data_dir / "ground_truth"
    if not gt_dir.exists() or not any(gt_dir.glob("*.json")):
        print(f"  No ground truth found in {gt_dir}")
        return None
    return LPSDataset(
        data_dir, neg_per_pos=3, bbox_jitter=0.0, augment=False, seed=42
    )


def _run_pair(
    label: str,
    val_ds: LPSDataset | None,
    results: dict[str, dict[str, dict[str, float]]],
    key: str,
) -> None:
    if val_ds is None:
        return

    print(f"Running baseline on {label} ...")
    r = _evaluate(BASELINE, val_ds)
    if r:
        results.setdefault(key, {})["baseline"] = r

    print(f"Running fine-tuned on {label} ...")
    r = _evaluate(FINETUNED, val_ds)
    if r:
        results.setdefault(key, {})["finetuned"] = r


def main() -> None:
    results: dict[str, dict[str, dict[str, float]]] = {}

    # --- Build datasets ---
    print("Building real-test dataset (held out) ...")
    real_test_ds = _build_val_ds(FINETUNE_DATA / "real_test")
    if real_test_ds:
        print(f"  Pairs: {len(real_test_ds)}")

    print("Building real-val dataset (selection set) ...")
    real_val_ds = _build_val_ds(FINETUNE_DATA / "val")
    if real_val_ds:
        print(f"  Pairs: {len(real_val_ds)}")

    print("Building synthetic val dataset ...")
    synth_val_ds = _build_val_ds(SYNTH_DATA / "val")
    if synth_val_ds:
        print(f"  Pairs: {len(synth_val_ds)}")

    # --- Evaluate ---
    _run_pair("real test (held out)", real_test_ds, results, "real_test")
    _run_pair("real val (selection)", real_val_ds, results, "real_val")
    _run_pair("synthetic val", synth_val_ds, results, "synth_val")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("LPS EVAL SUMMARY")
    print("=" * 60)

    rt = results.get("real_test", {})
    if "baseline" in rt and "finetuned" in rt:
        _print_comparison("Real test — held out (paper number)", rt["baseline"], rt["finetuned"])

    rv = results.get("real_val", {})
    if "baseline" in rv and "finetuned" in rv:
        _print_comparison("Real val — selection set (reference)", rv["baseline"], rv["finetuned"])

    sv = results.get("synth_val", {})
    if "baseline" in sv and "finetuned" in sv:
        _print_comparison("Synthetic val — regression check", sv["baseline"], sv["finetuned"])

    # --- Verdict ---
    if "finetuned" in rt and "finetuned" in sv:
        real_delta = rt["finetuned"]["Accuracy"] - rt["baseline"]["Accuracy"]
        synth_delta = sv["finetuned"]["Accuracy"] - sv["baseline"]["Accuracy"]
        print(f"\n  Verdict:")
        print(f"    Real-data accuracy delta    : {real_delta:+.2%}")
        print(f"    Synthetic accuracy delta    : {synth_delta:+.2%}")
        if real_delta > 0.005 and synth_delta > -0.01:
            print(f"    --> Fine-tuning helped on real data with no synthetic regression")
        elif real_delta > 0.005 and synth_delta <= -0.01:
            print(f"    --> Fine-tuning helped on real data BUT regressed on synthetic")
            print(f"        Consider reducing REAL_OVERSAMPLE or increasing N_SYNTH_TRAIN")
        elif real_delta <= 0.005:
            print(f"    --> Minimal effect on real data — may need more annotations or epochs")

    print()


if __name__ == "__main__":
    main()
