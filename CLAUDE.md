# CLAUDE.md — structflo-cser

## What this project does

Chemical Structure-Label pair Extraction and Recognition (CSER) from scientific document pages.
Given a PDF or image of a chemistry paper/patent, the pipeline detects chemical structure drawings
and their compound labels (e.g. "CHEMBL12345", "Compound 1a"), pairs them, then extracts SMILES
strings and OCR text.

## Package name & layout

- **PyPI package**: `structflo-cser`
- **Top-level packages** (wheel): `structflo`, `annotate`
- **Source root**: `structflo/cser/` — all library code lives here
- **Annotate tool**: `annotate/` — Flask web app for manual bbox annotation

### Module map

```
structflo/cser/
  config.py           PageConfig dataclass (A4@300DPI defaults, slide layouts)
  _geometry.py         Pure bbox utilities (clamp, intersect, placement)
  weights.py           HF Hub weight registry + auto-download (resolve_weights)
  __init__.py          Package version

  data/
    smiles.py          Fetch/load SMILES from ChEMBL CSV
    distractor_images.py  Download/load distractor images for training data

  rendering/
    chemistry.py       RDKit 2D structure rendering to PIL images
    text.py            Label text rendering (random compound IDs, fonts, rotation)

  distractors/
    charts.py          Synthetic chart/figure distractors
    shapes.py          Geometric shapes for hard negatives
    text_elements.py   Prose blocks, captions, stray text

  generation/
    page.py            Core page compositor (place structures + labels + distractors)
    dataset.py         Dataset generation orchestrator (multiprocessing, YOLO label export)
    specialty.py       Specialty layouts (SAR tables, MMP sheets, data cards)
    tabular.py         Excel-style and grid compound layouts

  training/
    trainer.py         YOLO11l training wrapper (AdamW, cosine LR, grayscale augmentation)

  inference/
    detector.py        YOLO inference (tiled + full-image), visualisation
    tiling.py          Sliding-window tile generation
    nms.py             Greedy NMS for merging tiled detections
    pairing.py         Hungarian matching on centroid distance

  lps/                 Learned Pair Scorer (replaces Euclidean matching)
    features.py        14-dim geometric features + visual crop extraction
    scorer.py          PairScorer CNN (~557K params): struct_crop + label_crop + geom → logit
    matcher.py         LearnedMatcher (BaseMatcher impl using PairScorer + Hungarian)
    dataset.py         LPS training dataset (positive/negative pair sampling from GT)
    train.py           LPS training loop
    evaluate.py        LPS evaluation script

  pipeline/
    models.py          Core dataclasses: BBox, Detection, CompoundPair
    matcher.py         BaseMatcher ABC + HungarianMatcher
    ocr.py             BaseOCR ABC + EasyOCRExtractor + NullOCR
    smiles_extractor.py  BaseSmilesExtractor ABC + DecimerExtractor + NullSmilesExtractor
    pipeline.py        ChemPipeline: detect → match → enrich (main public API)
    cli.py             sf-extract CLI entry point

  viz/
    labels.py          Visualise YOLO label files on synthetic pages
    detections.py      Matplotlib plots for Detection/CompoundPair objects

annotate/
  __main__.py          Flask annotation tool entry point
  server.py            Flask routes
  pdf.py               PDF page rendering for annotation
  storage.py           Annotation JSON storage
  templates/           HTML templates
```

## CLI entry points (registered in pyproject.toml)

| Command                   | Module                                    | Purpose                              |
|---------------------------|-------------------------------------------|--------------------------------------|
| `sf-generate`             | `structflo.cser.generation.dataset:main`  | Generate synthetic training data     |
| `sf-train`                | `structflo.cser.training.trainer:main`    | Train YOLO11l detector               |
| `sf-detect`               | `structflo.cser.inference.detector:main`  | Run detection on images              |
| `sf-extract`              | `structflo.cser.pipeline.cli:main`        | Full pipeline: detect+match+extract  |
| `sf-viz`                  | `structflo.cser.viz.labels:main`          | Visualise YOLO labels on images      |
| `sf-fetch-smiles`         | `structflo.cser.data.smiles:main`         | Download SMILES from ChEMBL          |
| `sf-download-distractors` | `structflo.cser.data.distractor_images:main` | Download distractor images        |
| `sf-annotate`             | `annotate.__main__:main`                  | Manual annotation web tool           |
| `sf-train-lps`            | `structflo.cser.lps.train:main`           | Train Learned Pair Scorer            |
| `sf-eval-lps`             | `structflo.cser.lps.evaluate:main`        | Evaluate LPS model                   |

## Key public API

```python
from structflo.cser.pipeline import ChemPipeline

pipeline = ChemPipeline()                    # auto-downloads weights
pairs = pipeline.process("page.png")         # detect → match → enrich
pairs = pipeline.process_pdf("paper.pdf")    # per-page processing

# Low-level access
detections = pipeline.detect(image)
pairs = pipeline.match(detections, image=image)
pairs = pipeline.enrich(pairs, image)

# Output
ChemPipeline.to_json(pairs)
ChemPipeline.to_dataframe(pairs)
ChemPipeline.to_records(pairs)
```

## Detection model

- **Architecture**: YOLO11l (ultralytics)
- **Classes**: 2 — `chemical_structure` (0), `compound_label` (1)
- **Training image size**: 1280px
- **Inference**: sliding-window tiling (1536px tiles, 20% overlap) + per-class NMS
- **Training config**: AdamW, cosine LR, grayscale images, no colour augmentation
- **Runs directory**: `runs/labels_detect/`
- **YOLO data config**: `config/data.yaml`

## Matching strategies

1. **HungarianMatcher** — centroid Euclidean distance + `scipy.optimize.linear_sum_assignment`
2. **LearnedMatcher** (LPS) — CNN scorer produces association probability per (struct, label) pair,
   then Hungarian on `1 - score`. Default in ChemPipeline.

## Weights system

Weights are versioned independently of the package and stored on HuggingFace Hub.
`structflo.cser.weights.resolve_weights(model, version)` handles auto-download + caching.

| Model          | HF Repo                         | Latest |
|----------------|----------------------------------|--------|
| cser-detector  | sidxz/structflo-cser-detector   | v0.2   |
| cser-lps       | sidxz/structflo-cser-lps        | v0.1   |

Publish script: `scripts/publish_weights.py`

## Fine-tuning on real data

Scripts live in `scripts/finetune/{yolo,lps}/`, each with `prepare_data.py`, `train.sh`, `eval_compare.py`.

### Data layout
- **Real annotations**: produced by `sf-annotate`, stored externally (symlinked in)
- **Combined data**: `data/finetune/{yolo,lps}/` — symlinks mixing subsampled synthetic + oversampled real
- Knobs at top of each `prepare_data.py`: `N_SYNTH_TRAIN`, `N_SYNTH_VAL`, `REAL_OVERSAMPLE`, `N_REAL_VAL`

### YOLO fine-tune
- Starts from `runs/labels_detect/yolo11l_panels/weights/best.pt`
- Output: `runs/labels_detect/finetune_trial/weights/best.pt`
- Lower LR (1e-4), short warmup (1 epoch), 10 epochs default

### LPS fine-tune
- Uses `sf-train-lps --finetune <checkpoint>` (loads weights only, fresh optimizer/scheduler)
- Distinct from `--resume` which restores full training state (optimizer, scheduler, epoch)
- Starts from `runs/lps/best.pt`, output: `runs/lps_finetune/best.pt`

### Eval
- `eval_compare.py` runs both baseline and fine-tuned on two val sets (finetune val + original synthetic val)
- Prints summary table with deltas and a verdict (improvement vs regression)

### Publishing fine-tuned weights
```bash
python scripts/publish_weights.py --model cser-detector --version vX.Y \
    --weights-file runs/labels_detect/finetune_trial/weights/best.pt
python scripts/publish_weights.py --model cser-lps --version vX.Y \
    --weights-file runs/lps_finetune/best.pt
```

## Synthetic data generation

- Pages: A4@300DPI (2480x3508) as JPEG, also slide layouts (16:9)
- Layout types: free-form (~30%), Excel tables (~14%), grids (~12%), SAR tables (~8%),
  MMP sheets (~7%), data cards (~8%), slides (~13%), hard negatives (~8%)
- Structures rendered via RDKit from ChEMBL SMILES
- Labels: random compound IDs in various styles (CHEMBL, ZINC, Roman numerals, etc.)
- Noise augmentation: JPEG artifacts, blur, brightness, Gaussian noise
- Output: images + YOLO .txt labels + ground truth JSON (per-compound struct/label bboxes + SMILES)
- Default: 2000 train / 200 val pages, multiprocessing with all CPUs

## Build & dev

```bash
uv sync --dev              # install all deps
uv run ruff check structflo/ tests/   # lint
uv run ruff format structflo/ tests/  # format
uv run pytest -q           # tests
uv build                   # build wheel
```

- **Python**: >=3.11 (project uses 3.12)
- **Build system**: hatchling + hatch-vcs (version from git tags)
- **Linting**: ruff
- **Tests**: pytest (tests/ directory)
- **CI**: GitHub Actions — lint + format check + pytest + coverage on push/PR to main
- **PyPI publish**: on git tag `v*`

## Conventions

- All images converted to grayscale before detection (matches training distribution)
- Adapters pattern: `BaseMatcher`, `BaseOCR`, `BaseSmilesExtractor` ABCs for swappable components
- Lazy model loading throughout (YOLO, EasyOCR, DECIMER loaded on first use)
- Weights never committed to git (*.pt in .gitignore), only on HF Hub
- `runs/`, `data/`, `detections/`, `archive/` are gitignored
