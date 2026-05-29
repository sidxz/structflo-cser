"""Prepare combined data directory for LPS fine-tuning (3-way real split).

Reuses the SAME real train/val/test page assignment as the detector by reading
the shared split manifest written by scripts/finetune/yolo/prepare_data.py, so
the detector and the LPS report on the identical held-out real pages.

Creates data/finetune/lps/ with symlinks:
  train/{images,ground_truth}      — synthetic subsample + oversampled real-train
  val/{images,ground_truth}        — real-only selection set (drives best.pt)
  real_test/{images,ground_truth}  — real-only, held out for reporting

The LPS dataset reads ground_truth/*.json + images/* (no YOLO labels needed).

Prerequisite:
    uv run python scripts/finetune/yolo/prepare_data.py   # writes real_split.json

Usage:
    uv run python scripts/finetune/lps/prepare_data.py
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[3]

REAL_DATA_DIR = Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data")
SYNTH_DATA_DIR = PROJECT_ROOT / "data" / "generated"

OUT_DIR = PROJECT_ROOT / "data" / "finetune" / "lps"
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "finetune" / "real_split.json"

N_SYNTH_TRAIN = 2000

# LPS is small (~557K params); ~2x oversampling reaches a ~50/50 real:synth mix.
REAL_OVERSAMPLE = 2

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symlink(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    dst.symlink_to(src.resolve())


def _find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _real_pairs_for(stems: list[str]) -> list[tuple[Path, Path]]:
    """Map manifest stems -> (image, ground_truth_json) for real pages."""
    img_dir = REAL_DATA_DIR / "images"
    gt_dir = REAL_DATA_DIR / "ground_truth"
    pairs = []
    missing = 0
    for stem in stems:
        img = _find_image(img_dir, stem)
        gt = gt_dir / f"{stem}.json"
        if img is None or not gt.exists():
            missing += 1
            continue
        pairs.append((img, gt))
    if missing:
        print(f"  WARNING: {missing} manifest stems had no image/ground_truth")
    return pairs


def _collect_synth(img_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for gt in sorted(gt_dir.glob("*.json")):
        img = _find_image(img_dir, gt.stem)
        if img is not None:
            pairs.append((img, gt))
    return pairs


def _populate_clean(pairs: list[tuple[Path, Path]], out: Path) -> None:
    (out / "images").mkdir(parents=True)
    (out / "ground_truth").mkdir(parents=True)
    for img, gt in pairs:
        _symlink(img, out / "images" / img.name)
        _symlink(gt, out / "ground_truth" / gt.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    if not SPLIT_MANIFEST.exists():
        raise SystemExit(
            f"Split manifest not found: {SPLIT_MANIFEST}\n"
            "Run scripts/finetune/yolo/prepare_data.py first."
        )
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    print(f"Using split manifest : {SPLIT_MANIFEST}")
    print(
        f"Manifest test/val/train: "
        f"{len(manifest['test'])} / {len(manifest['val'])} / {len(manifest['train'])}"
    )

    real_test = _real_pairs_for(manifest["test"])
    real_val = _real_pairs_for(manifest["val"])
    real_train = _real_pairs_for(manifest["train"])
    print(f"Real test/val/train  : {len(real_test)} / {len(real_val)} / {len(real_train)}")

    synth_train_pairs = _collect_synth(
        SYNTH_DATA_DIR / "train" / "images",
        SYNTH_DATA_DIR / "train" / "ground_truth",
    )
    synth_train_sample = random.sample(
        synth_train_pairs, min(N_SYNTH_TRAIN, len(synth_train_pairs))
    )
    print(f"Synthetic train      : {len(synth_train_pairs)} (sampling {len(synth_train_sample)})")

    # --- Reset output dirs ---
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "train" / "images").mkdir(parents=True)
    (OUT_DIR / "train" / "ground_truth").mkdir(parents=True)

    # --- Populate train: synthetic (1 copy) + real-train (oversampled) ---
    count = 0
    for img, gt in synth_train_sample:
        _symlink(img, OUT_DIR / "train" / "images" / img.name)
        _symlink(gt, OUT_DIR / "train" / "ground_truth" / gt.name)
        count += 1

    for copy_i in range(REAL_OVERSAMPLE):
        for img, gt in real_train:
            name = f"{img.stem}_real{copy_i:02d}{img.suffix}"
            gt_name = f"{img.stem}_real{copy_i:02d}.json"
            _symlink(img, OUT_DIR / "train" / "images" / name)
            _symlink(gt, OUT_DIR / "train" / "ground_truth" / gt_name)
            count += 1

    print(
        f"Train pages total    : {count}  "
        f"({len(synth_train_sample)} synth + {len(real_train)}x{REAL_OVERSAMPLE} real)"
    )

    # --- Clean real-only val (selection) and test (report) ---
    _populate_clean(real_val, OUT_DIR / "val")
    _populate_clean(real_test, OUT_DIR / "real_test")
    print(f"Val (selection)      : {len(real_val)} pages")
    print(f"Real test (report)   : {len(real_test)} pages")

    print(f"\nOutput: {OUT_DIR}")
    print("Done. Next: run scripts/finetune/lps/train.sh")


if __name__ == "__main__":
    main()
