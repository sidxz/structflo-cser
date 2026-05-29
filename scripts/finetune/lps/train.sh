#!/usr/bin/env bash
# Fine-tune LPS (Learned Pair Scorer) on combined synthetic + real data.
#
# Prerequisites:
#   uv run python scripts/finetune/lps/prepare_data.py
#
# Usage:
#   bash scripts/finetune/lps/train.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/finetune/lps"
CHECKPOINT="$PROJECT_ROOT/runs/lps/best.pt"
OUT_DIR="$PROJECT_ROOT/runs/lps_finetune"

if [ ! -d "$DATA_DIR/train" ]; then
    echo "ERROR: $DATA_DIR/train not found. Run prepare_data.py first."
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint $CHECKPOINT not found."
    exit 1
fi

echo "=== LPS Fine-tune (trial) ==="
echo "  Data:       $DATA_DIR"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output:     $OUT_DIR"
echo ""

uv run sf-train-lps \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --finetune "$CHECKPOINT" \
    --epochs 10 \
    --lr 3e-4 \
    --batch 512 \
    --workers 8
