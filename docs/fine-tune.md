
  # YOLO
  uv run python scripts/finetune/yolo/prepare_data.py
  bash scripts/finetune/yolo/train.sh
  uv run python scripts/finetune/yolo/eval_compare.py

  # LPS
  uv run python scripts/finetune/lps/prepare_data.py
  bash scripts/finetune/lps/train.sh
  uv run python scripts/finetune/lps/eval_compare.py

  Eval now checks two things per model:
  1. Finetune val (50 synth + 2 real) — did real data help?
  2. Original synthetic val (2000 pages) — did fine-tuning regress?

  To publish if results look good:
  python scripts/publish_weights.py --model cser-detector --version v0.3 \
      --weights-file runs/labels_detect/finetune_trial/weights/best.pt

  python scripts/publish_weights.py --model cser-lps --version v0.2 \
      --weights-file runs/lps_finetune/best.pt




 Based on the current defaults in scripts/finetune/*/prepare_data.py, here's what to scale up:

  ┌─────────────────┬─────────────┬────────────────────────────┬────────────────────────────────────────────────────┐
  │      Param      │ Trial (now) │ Full dataset (~100+ pages) │                        Why                         │
  ├─────────────────┼─────────────┼────────────────────────────┼────────────────────────────────────────────────────┤
  │ N_SYNTH_TRAIN   │ 200         │ 2000–5000                  │ More synthetic to prevent forgetting               │
  ├─────────────────┼─────────────┼────────────────────────────┼────────────────────────────────────────────────────┤
  │ N_SYNTH_VAL     │ 50          │ 200–500                    │ More reliable val metrics                          │
  ├─────────────────┼─────────────┼────────────────────────────┼────────────────────────────────────────────────────┤
  │ REAL_OVERSAMPLE │ 10          │ 3–5                        │ Less oversampling needed since you have more pages │
  ├─────────────────┼─────────────┼────────────────────────────┼────────────────────────────────────────────────────┤
  │ N_REAL_VAL      │ 2           │ 10–20% of total            │ Meaningful real val set                            │
  └─────────────────┴─────────────┴────────────────────────────┴────────────────────────────────────────────────────┘

  The goal is to keep real data at roughly 30–50% of training. With 100 real pages at 3x oversample + 2000 synthetic, that's 300/(300+2000) = ~13% — bump oversample to 5 for ~20%, or lower N_SYNTH_TRAIN
  to 1000 for ~33%.

  For training hyperparams in train.sh:
  - Epochs: trial uses 10, full run can stay at 10–15 (more data per epoch means less risk of overfitting)
  - LR: keep as-is (1e-4 YOLO, 3e-4 LPS) — these are already conservative


With 500 annotated pages, here are the recommended values and what each param does:

  prepare_data.py params

  N_SYNTH_TRAIN (trial: 200 → recommended: 3000–5000)

  Number of synthetic images randomly sampled into the training set. Synthetic data acts as a regularizer — it prevents the model from overfitting to quirks of your specific papers (particular fonts, DPI,
   layout style). Too few and the model forgets synthetic-learned features; too many and real data gets drowned out. With 500 real pages, 3000–5000 synthetic keeps the ratio healthy.

  N_SYNTH_VAL (trial: 50 → recommended: 300–500)

  Synthetic images in the validation set. More gives you stable metrics — with only 50, a single noisy page can swing mAP by several points. 300+ makes the regression check trustworthy.

  REAL_OVERSAMPLE (trial: 10 → recommended: 2–3)

  Each real image gets this many symlink copies with unique names, so YOLO/LPS treats them as separate training examples. With only 14 pages you needed 10x to make real data visible. With 500 pages, 2–3x
  is enough. At 3x: 1500 real / (1500 + 4000 synth) = 27% of training. Too high and the model memorizes your annotation set instead of generalizing.

  N_REAL_VAL (trial: 2 → recommended: 50–75)

  Real pages held out for validation (never seen during training). This is your ground truth for "did fine-tuning actually help on real documents?". At 50 pages you get reliable per-class metrics. The
  remaining 425–450 go to training.

  train.sh params (YOLO)

  epochs=10 → recommended: 10–15

  One full pass over all training data. More epochs = more chances to learn, but with 500 real pages you have enough data per epoch that 10–15 is sufficient. Early stopping (patience=5) will halt if val
  metrics plateau anyway.

  lr0=1e-4 → keep as-is

  Starting learning rate, 10x lower than from-scratch training (1e-3). Low LR is critical for fine-tuning — too high and you destroy the features learned from 20K synthetic pages. Too low and you never
  adapt. 1e-4 is a standard fine-tune rate.

  warmup_epochs=1 → keep as-is

  Epochs where LR ramps from ~0 to lr0. Prevents large early gradient updates that could destabilize the pretrained weights. 1 is enough since we're already starting with a well-trained model.

  patience=5 → keep as-is

  Stop training if val mAP doesn't improve for this many epochs. Prevents overfitting if the model converges early.

  train.sh params (LPS)

  --lr 3e-4 → keep as-is

  Same logic — lower than from-scratch (1e-3) but the LPS model is tiny (557K params) so it can tolerate a slightly higher fine-tune LR than YOLO.

  --epochs 10 → recommended: 10–15

  --batch 512 → keep as-is

  Batch size. LPS samples are small (crops + 14-dim features), so 512 fits easily in memory and gives stable gradient estimates.

  Concrete config for 500 pages

  # prepare_data.py (both yolo and lps)
  N_SYNTH_TRAIN = 4000
  N_SYNTH_VAL = 400
  N_REAL_VAL = 50
  REAL_OVERSAMPLE = 3

  That gives you:
  - Train: 4000 synth + 450×3 = 5350 images (25% real)
  - Val: 400 synth + 50 real = 450 images

  Training hyperparams stay the same — they're already tuned for fine-tuning.