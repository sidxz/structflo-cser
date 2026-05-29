"""Compare baseline (synthetic-only) vs fine-tuned YOLO.

Evaluates both models on three val sets:
  1. Real test (held out) — the paper number; never trained on or selected against
  2. Real val (selection set) — drove early-stopping/checkpointing (reference only)
  3. Original synthetic val (1000 pages) — regression check

Prints a summary table with deltas at the end.

Usage:
    uv run python scripts/finetune/yolo/eval_compare.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parents[3]

REAL_TEST_YAML = PROJECT_ROOT / "data" / "finetune" / "yolo" / "data_real_test.yaml"
REAL_VAL_YAML = PROJECT_ROOT / "data" / "finetune" / "yolo" / "data.yaml"
SYNTH_YAML = PROJECT_ROOT / "config" / "data.yaml"

RUN_NAME = "finetune_3way"
BASELINE = PROJECT_ROOT / "runs" / "labels_detect" / "yolo11l_panels" / "weights" / "best.pt"
FINETUNED = PROJECT_ROOT / "runs" / "labels_detect" / RUN_NAME / "weights" / "best.pt"

METRICS = ["mAP50", "mAP50-95", "Precision", "Recall"]


def _val(weights: Path, data_yaml: Path) -> dict[str, float] | None:
    if not weights.exists():
        print(f"  weights not found: {weights}")
        return None
    if not data_yaml.exists():
        print(f"  data config not found: {data_yaml}")
        return None

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), verbose=False)

    return {
        "mAP50": m.box.map50,
        "mAP50-95": m.box.map,
        "Precision": m.box.mp,
        "Recall": m.box.mr,
    }


def _delta_str(baseline: float, finetuned: float) -> str:
    d = finetuned - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _print_comparison(label: str, baseline: dict, finetuned: dict) -> None:
    print(f"\n  {label}")
    print(f"  {'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-' * 46}")
    for m in METRICS:
        b, f = baseline[m], finetuned[m]
        print(f"  {m:<12} {b:>10.4f} {f:>12.4f} {_delta_str(b, f):>10}")


def _run_pair(
    label: str,
    data_yaml: Path,
    results: dict[str, dict[str, dict[str, float]]],
    key: str,
) -> None:
    print(f"Running baseline on {label} ...")
    r = _val(BASELINE, data_yaml)
    if r:
        results.setdefault(key, {})["baseline"] = r

    print(f"Running fine-tuned on {label} ...")
    r = _val(FINETUNED, data_yaml)
    if r:
        results.setdefault(key, {})["finetuned"] = r


def main() -> None:
    results: dict[str, dict[str, dict[str, float]]] = {}

    _run_pair("real test (held out)", REAL_TEST_YAML, results, "real_test")
    _run_pair("real val (selection)", REAL_VAL_YAML, results, "real_val")
    _run_pair("synthetic val", SYNTH_YAML, results, "synth_val")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("YOLO EVAL SUMMARY")
    print("=" * 60)

    rt = results.get("real_test", {})
    if "baseline" in rt and "finetuned" in rt:
        _print_comparison(
            "Real test — held out (paper number)", rt["baseline"], rt["finetuned"]
        )

    rv = results.get("real_val", {})
    if "baseline" in rv and "finetuned" in rv:
        _print_comparison(
            "Real val — selection set (reference)", rv["baseline"], rv["finetuned"]
        )

    sv = results.get("synth_val", {})
    if "baseline" in sv and "finetuned" in sv:
        _print_comparison(
            "Synthetic val — regression check", sv["baseline"], sv["finetuned"]
        )

    # --- Verdict ---
    if "finetuned" in rt and "finetuned" in sv:
        real_delta = rt["finetuned"]["mAP50"] - rt["baseline"]["mAP50"]
        synth_delta = sv["finetuned"]["mAP50"] - sv["baseline"]["mAP50"]
        print(f"\n  Verdict:")
        print(f"    Real-data mAP50 delta      : {real_delta:+.4f}")
        print(f"    Synthetic mAP50 delta      : {synth_delta:+.4f}")
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
