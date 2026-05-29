#!/usr/bin/env bash
# Fine-tune YOLO detector on combined synthetic + real data.
#
# Prerequisites:
#   uv run python scripts/finetune/yolo/prepare_data.py
#
# Usage:
#   bash scripts/finetune/yolo/train.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CHECKPOINT="$PROJECT_ROOT/runs/labels_detect/yolo11l_panels/weights/best.pt"
OUT_DIR="$PROJECT_ROOT/runs/labels_detect"

# Overridable via env vars (defaults reproduce the original trial run).
DATA_YAML="${DATA_YAML:-$PROJECT_ROOT/data/finetune/yolo/data.yaml}"
RUN_NAME="${RUN_NAME:-finetune_trial}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
MOSAIC="${MOSAIC:-0.5}"

if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: $DATA_YAML not found. Run prepare_data.py first."
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint $CHECKPOINT not found."
    exit 1
fi

echo "=== YOLO Fine-tune ==="
echo "  Data:       $DATA_YAML"
echo "  Checkpoint: $CHECKPOINT"
echo "  Run name:   $RUN_NAME"
echo "  Epochs:     $EPOCHS  (patience $PATIENCE, mosaic $MOSAIC)"
echo ""

uv run python -c "
from ultralytics import YOLO

model = YOLO('$CHECKPOINT')

model.train(
    data='$DATA_YAML',
    epochs=$EPOCHS,
    patience=$PATIENCE,
    batch=8,
    imgsz=1280,
    optimizer='AdamW',
    lr0=1e-4,          # 10x lower than from-scratch
    lrf=0.01,
    weight_decay=5e-4,
    warmup_epochs=1,    # short warmup (already trained)
    cos_lr=True,
    # Same augmentation as original training
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.1,
    degrees=3.0,
    translate=0.1,
    scale=0.3,
    shear=1.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=$MOSAIC,
    mixup=0.0,
    copy_paste=0.0,
    workers=8,
    project='$OUT_DIR',
    name='$RUN_NAME',
    exist_ok=True,
    seed=42,
    plots=True,
    save_period=5,
)

# Validate
from pathlib import Path
best = Path('$OUT_DIR') / '$RUN_NAME' / 'weights' / 'best.pt'
if best.exists():
    m = YOLO(str(best)).val(data='$DATA_YAML')
    print()
    print('--- Fine-tuned model metrics ---')
    print(f'mAP50:     {m.box.map50:.4f}')
    print(f'mAP50-95:  {m.box.map:.4f}')
    print(f'Precision: {m.box.mp:.4f}')
    print(f'Recall:    {m.box.mr:.4f}')
    print(f'Weights:   {best}')
"
