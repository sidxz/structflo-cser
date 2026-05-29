"""Prepare combined (synthetic + real) data directory for YOLO fine-tuning.

Three-way real split for a defensible paper number:
  train/       — synthetic subsample + oversampled real-train pages
  real_val/    — real-only, drives early-stopping & checkpoint selection
  real_test/   — real-only, held out for final reporting (never trained on
                 and never used for model selection)

Usage:
    uv run python scripts/finetune/yolo/prepare_data.py

Adjust the knobs below for your run.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — adjust these for your run
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parents[3]

REAL_DATA_DIR = Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data")
SYNTH_DATA_DIR = PROJECT_ROOT / "data" / "generated"

OUT_DIR = PROJECT_ROOT / "data" / "finetune" / "yolo"

# How many synthetic images to mix into training.
N_SYNTH_TRAIN = 2000

# Real-page split. real_test is the number reported in the paper and must never
# be used for training or model selection; real_val drives early-stopping and
# checkpoint selection during training.
N_REAL_TEST = 100
N_REAL_VAL = 75

# Oversample factor for real-train images (~2x reaches a ~50/50 real:synth mix).
REAL_OVERSAMPLE = 2

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symlink(src: Path, dst: Path) -> None:
    """Create a symlink, removing any existing one."""
    dst.unlink(missing_ok=True)
    dst.symlink_to(src.resolve())


def _collect_pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) pairs where both files exist."""
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def _populate_clean(pairs: list[tuple[Path, Path]], out: Path) -> None:
    """Symlink a real-only split (1 copy each, original names)."""
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    for img, lbl in pairs:
        _symlink(img, out / "images" / img.name)
        _symlink(lbl, out / "labels" / lbl.name)


def _write_yaml(path: Path, val_subdir: str) -> None:
    path.write_text(
        f"path: {OUT_DIR.resolve()}\n"
        f"train: train/images\n"
        f"val: {val_subdir}\n"
        f"\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: chemical_structure\n"
        f"  1: compound_label\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    # --- Discover data ---
    real_pairs = _collect_pairs(REAL_DATA_DIR / "images", REAL_DATA_DIR / "labels")
    synth_train_pairs = _collect_pairs(
        SYNTH_DATA_DIR / "train" / "images", SYNTH_DATA_DIR / "train" / "labels"
    )

    print(f"Real annotated pages : {len(real_pairs)}")
    print(f"Synthetic train      : {len(synth_train_pairs)}")

    need = N_REAL_TEST + N_REAL_VAL + 1
    if len(real_pairs) < need:
        raise ValueError(f"Need at least {need} real pages, got {len(real_pairs)}")

    # --- Three-way real split ---
    random.shuffle(real_pairs)
    real_test = real_pairs[:N_REAL_TEST]
    real_val = real_pairs[N_REAL_TEST : N_REAL_TEST + N_REAL_VAL]
    real_train = real_pairs[N_REAL_TEST + N_REAL_VAL :]

    print(
        f"Real test/val/train  : {len(real_test)} / {len(real_val)} / {len(real_train)}"
    )

    # --- Write shared split manifest (consumed by lps/prepare_data.py so the
    #     detector and LPS report on the identical held-out real pages) ---
    manifest_path = OUT_DIR.parent / "real_split.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "source": "scripts/finetune/yolo/prepare_data.py",
                "test": [img.stem for img, _ in real_test],
                "val": [img.stem for img, _ in real_val],
                "train": [img.stem for img, _ in real_train],
            },
            indent=2,
        )
    )
    print(f"Split manifest       : {manifest_path}")

    # --- Subsample synthetic ---
    synth_train_sample = random.sample(
        synth_train_pairs, min(N_SYNTH_TRAIN, len(synth_train_pairs))
    )

    # --- Reset output dirs ---
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "train" / "images").mkdir(parents=True)
    (OUT_DIR / "train" / "labels").mkdir(parents=True)

    # --- Populate train: synthetic (1 copy) + real-train (oversampled) ---
    count = 0
    for img, lbl in synth_train_sample:
        _symlink(img, OUT_DIR / "train" / "images" / img.name)
        _symlink(lbl, OUT_DIR / "train" / "labels" / lbl.name)
        count += 1

    for copy_i in range(REAL_OVERSAMPLE):
        for img, lbl in real_train:
            name = f"{img.stem}_real{copy_i:02d}{img.suffix}"
            lbl_name = f"{img.stem}_real{copy_i:02d}.txt"
            _symlink(img, OUT_DIR / "train" / "images" / name)
            _symlink(lbl, OUT_DIR / "train" / "labels" / lbl_name)
            count += 1

    print(
        f"Train images total   : {count}  "
        f"({len(synth_train_sample)} synth + {len(real_train)}x{REAL_OVERSAMPLE} real)"
    )

    # --- Populate clean real-only val (selection) and test (report) ---
    _populate_clean(real_val, OUT_DIR / "real_val")
    _populate_clean(real_test, OUT_DIR / "real_test")
    print(f"Real val (selection) : {len(real_val)} pages")
    print(f"Real test (report)   : {len(real_test)} pages")

    # --- Write YAML configs ---
    # Training validates on the real selection set (data.yaml is the train.sh default).
    _write_yaml(OUT_DIR / "data.yaml", "real_val/images")
    # Held-out real test set, used only for final reporting in eval_compare.py.
    _write_yaml(OUT_DIR / "data_real_test.yaml", "real_test/images")

    print("\nData configs written to:")
    print(f"  {OUT_DIR / 'data.yaml'}            (val = real selection set)")
    print(f"  {OUT_DIR / 'data_real_test.yaml'}  (val = held-out real test)")
    print("Done. Next: run scripts/finetune/yolo/train.sh")


if __name__ == "__main__":
    main()
