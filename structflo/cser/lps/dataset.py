"""PyTorch Dataset for training the Learned Pair Scorer.

Each sample is one candidate (structure, label) pair drawn from the
synthetic data ground truth.  Positive pairs come directly from the GT JSON
associations; hard negative pairs are the spatially *nearest* wrong labels
on the same page.

Internal storage uses flat numpy arrays so that the dataset can be pickled
quickly (~40 MB for 700K+ samples) when DataLoader spawns worker processes.

Image loading in workers uses cv2 to avoid libjpeg mutex deadlocks that
occur when PIL's JPEG state is forked into child processes.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from io import BytesIO
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from structflo.cser.lps.features import (
    LABEL_CROP_SIZE,
    STRUCT_CROP_SIZE,
    crop_region,
    geom_features,
)

Split = Literal["train", "val"]


# ---------------------------------------------------------------------------
# Visual augmentation (training only)
# ---------------------------------------------------------------------------


def _augment_crop(
    crop: np.ndarray,
    rng: np.random.Generator,
    max_rot: float = 45.0,
    p_flip: float = 0.5,
    brightness_range: float = 0.25,
) -> np.ndarray:
    """Domain-robust augmentation on a (1, H, W) float32 crop.

    Designed to bridge the gap between synthetic RDKit renders and real-world
    document crops (ChemDraw, MarvinSketch, scanned PDFs, screenshots).

    Transforms applied independently with their own probabilities:
    - Rotation ± max_rot degrees  (spatial invariance)
    - Horizontal flip             (spatial invariance)
    - Brightness jitter           (multiplicative; scanner exposure variation)
    - Contrast jitter             (histogram stretch; render engine differences)
    - Gaussian blur               (lower DPI / out-of-focus scans)
    - Morphological erosion/dilation (line-width differences across renderers)
    - Gaussian noise              (scanner grain)
    - JPEG recompression          (PDF-internal lossy compression artefacts)
    - Random inversion            (dark-background slides / highlighted regions)
    """
    img = Image.fromarray((crop[0] * 255).astype(np.uint8), mode="L")

    # --- Spatial transforms ---
    angle = float(rng.uniform(-max_rot, max_rot))
    img = img.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255
    )

    if rng.random() < p_flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)

    # --- Intensity transforms ---
    # Brightness jitter: multiplicative shift of all pixels
    if brightness_range > 0 and rng.random() < 0.70:
        arr = np.clip(
            arr * float(rng.uniform(1 - brightness_range, 1 + brightness_range)),
            0.0,
            1.0,
        )

    # Contrast jitter: stretch / compress histogram around mid-grey.
    # Models the wide variation between high-contrast vector PDFs and flat scans.
    if rng.random() < 0.60:
        c = float(rng.uniform(0.70, 1.45))
        arr = np.clip((arr - 0.5) * c + 0.5, 0.0, 1.0)

    # Gaussian blur: simulates lower DPI rendering or slight scan defocus
    if rng.random() < 0.25:
        arr = ndi.gaussian_filter(arr, sigma=float(rng.uniform(0.3, 1.2)))

    # --- Structural / renderer-difference transforms ---
    # Morphological erosion → thinner lines (ChemDraw thin-export, vector PDF)
    # Morphological dilation → thicker lines (bold print, scanned copies)
    # This is the primary fix for the visual domain gap between renderers.
    if rng.random() < 0.35:
        foreground = arr < 0.5  # True = dark ink pixel
        iterations = int(rng.integers(1, 3))  # 1 or 2 px
        if rng.random() < 0.5:
            processed = ndi.binary_erosion(foreground, iterations=iterations)
        else:
            processed = ndi.binary_dilation(foreground, iterations=iterations)
        arr = np.where(processed, 0.0, 1.0).astype(np.float32)

    # --- Noise / compression artefacts ---
    # Gaussian noise: scanner grain, compressed image quantisation
    if rng.random() < 0.20:
        arr = np.clip(
            arr
            + rng.normal(0, float(rng.uniform(0.01, 0.04)), arr.shape).astype(
                np.float32
            ),
            0.0,
            1.0,
        )

    # JPEG recompression: many PDFs store pages as internal JPEG streams;
    # the resulting block artefacts are a common domain shift for real crops.
    if rng.random() < 0.25:
        quality = int(rng.integers(35, 86))
        tmp = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        buf = BytesIO()
        tmp.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        arr = np.array(Image.open(buf), dtype=np.float32) / 255.0

    # Random inversion: dark-background slides, highlighted compound boxes
    if rng.random() < 0.08:
        arr = 1.0 - arr

    return arr[np.newaxis]  # (1, H, W)


# ---------------------------------------------------------------------------
# Per-worker image cache
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _load_page_image(path: str) -> np.ndarray | None:
    """Decode a JPEG page once and cache the result per worker process.

    With ``persistent_workers=True`` and ``PageGroupSampler``, the same worker
    processes all samples from a given page consecutively, so subsequent calls
    with the same path are O(1) array lookups instead of full JPEG decodes.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Page-grouped sampler
# ---------------------------------------------------------------------------


class PageGroupSampler(Sampler[int]):
    """Shuffles pages, then yields every sample from each page consecutively.

    Combined with ``_load_page_image``'s LRU cache and ``persistent_workers``,
    this cuts JPEG decodes from O(N_samples) to O(N_pages) per epoch — roughly
    a 20× I/O reduction for typical datasets (~20 samples per page).

    Call ``set_epoch(epoch)`` before each epoch to re-shuffle pages.
    """

    def __init__(
        self,
        path_idx: np.ndarray,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._n = len(path_idx)
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        unique = np.unique(path_idx)
        self._groups: list[np.ndarray] = [np.where(path_idx == p)[0] for p in unique]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        order = (
            rng.permutation(len(self._groups))
            if self._shuffle
            else range(len(self._groups))
        )
        for gi in order:
            group = self._groups[gi].copy()
            if self._shuffle:
                rng.shuffle(group)
            yield from group.tolist()

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LPSDataset(Dataset):
    """Learned Pair Scorer training dataset.

    Every sample returns:
        ``geom``         — float32 tensor (GEOM_DIM,)
        ``struct_crop``  — float32 tensor (1, 128, 128)
        ``label_crop``   — float32 tensor (1,  64,  96)
        ``target``       — float32 scalar  (1.0 = true pair, 0.0 = negative)

    Args:
        data_dir:    Root of one data split, e.g. ``data/generated/train``.
                     Must contain ``images/`` and ``ground_truth/`` subdirs.
        neg_per_pos: Number of hard negative pairs generated per positive.
                     Negatives are the *neg_per_pos* spatially nearest wrong
                     labels on the same page.
        bbox_jitter: Fraction of bbox size used as uniform coordinate noise.
                     Simulates YOLO localisation errors (recommended: 0.02).
        augment:     If ``True``, apply random rotation/flip/brightness to
                     image crops.  Enable for training, disable for val.
        seed:        Random seed for reproducible jitter and augmentation.
    """

    def __init__(
        self,
        data_dir: Path | str,
        neg_per_pos: int = 3,
        bbox_jitter: float = 0.02,
        augment: bool = False,
        seed: int = 42,
        reject_negatives: bool = False,
    ) -> None:
        """
        Args (added):
            reject_negatives: If True, also emit "this structure has no valid
                label" negatives — pairing each unlabelled GT structure and each
                box listed in an optional ``fp_negatives/<stem>.json`` sidecar
                (mined detector false-positives) with its nearest labels,
                target 0.  Trains the scorer to reject distractor structures.
        """
        self.data_dir = Path(data_dir)
        self.bbox_jitter = bbox_jitter
        self.augment = augment
        self._seed = seed
        self.reject_negatives = reject_negatives
        self._build(neg_per_pos)

    # ------------------------------------------------------------------
    # Build phase — runs once in the main process
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid(bbox: list) -> tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def _dist(a: list, b: list) -> float:
        ax, ay = LPSDataset._centroid(a)
        bx, by = LPSDataset._centroid(b)
        return math.hypot(ax - bx, ay - by)

    def _build(self, neg_per_pos: int) -> None:
        gt_dir = self.data_dir / "ground_truth"
        img_dir = self.data_dir / "images"
        fp_dir = self.data_dir / "fp_negatives"
        has_fp = fp_dir.exists()

        json_files = sorted(gt_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No GT JSON files found in {gt_dir}")

        path_seen: dict[str, int] = {}
        path_list: list[str] = []

        path_indices: list[int] = []
        struct_bboxes: list[list[float]] = []
        label_bboxes: list[list[float]] = []
        page_sizes: list[list[float]] = []
        targets: list[int] = []

        for json_path in json_files:
            stem = json_path.stem

            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{stem}.png"
            if not img_path.exists():
                continue

            # Read dimensions from header only — no full JPEG decode in main process.
            with Image.open(img_path) as im:
                page_w, page_h = im.size

            path_str = str(img_path)
            if path_str not in path_seen:
                path_seen[path_str] = len(path_list)
                path_list.append(path_str)
            pid = path_seen[path_str]

            entries: list[dict] = json.loads(json_path.read_text())
            valid = [e for e in entries if e.get("label_bbox") is not None]
            if not valid:
                continue

            structs = [e["struct_bbox"] for e in valid]
            labels = [e["label_bbox"] for e in valid]
            n = len(valid)

            for i in range(n):
                path_indices.append(pid)
                struct_bboxes.append(structs[i])
                label_bboxes.append(labels[i])
                page_sizes.append([float(page_w), float(page_h)])
                targets.append(1)

            # Rejection negatives: structures with NO valid label (unlabelled GT
            # structures + mined detector false-positives) paired with their
            # nearest labels — target 0.  Teaches the scorer to reject distractors.
            if self.reject_negatives:
                distractors = [
                    e["struct_bbox"] for e in entries if e.get("label_bbox") is None
                ]
                if has_fp:
                    fp_file = fp_dir / f"{stem}.json"
                    if fp_file.exists():
                        distractors.extend(json.loads(fp_file.read_text()))
                for db in distractors:
                    wrong = sorted(
                        ((self._dist(db, labels[j]), j) for j in range(n)),
                        key=lambda t: t[0],
                    )
                    for _, j in wrong[:neg_per_pos]:
                        path_indices.append(pid)
                        struct_bboxes.append(db)
                        label_bboxes.append(labels[j])
                        page_sizes.append([float(page_w), float(page_h)])
                        targets.append(0)

            if n < 2:
                continue

            for i in range(n):
                wrong = sorted(
                    (
                        (self._dist(structs[i], labels[j]), j)
                        for j in range(n)
                        if j != i
                    ),
                    key=lambda t: t[0],
                )
                for _, j in wrong[:neg_per_pos]:
                    path_indices.append(pid)
                    struct_bboxes.append(structs[i])
                    label_bboxes.append(labels[j])
                    page_sizes.append([float(page_w), float(page_h)])
                    targets.append(0)

        self._img_paths: list[str] = path_list
        self._path_idx = np.array(path_indices, dtype=np.int32)
        self._struct_bboxes = np.array(struct_bboxes, dtype=np.float32)
        self._label_bboxes = np.array(label_bboxes, dtype=np.float32)
        self._page_sizes = np.array(page_sizes, dtype=np.float32)
        self._targets = np.array(targets, dtype=np.int8)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._targets)

    def pos_weight(self) -> float:
        """Ratio of negatives to positives — pass as ``pos_weight`` to BCEWithLogitsLoss."""
        n_pos = int(self._targets.sum())
        n_neg = len(self._targets) - n_pos
        return n_neg / max(n_pos, 1)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(self._seed ^ idx)

        s_bbox = self._struct_bboxes[idx].copy()
        l_bbox = self._label_bboxes[idx].copy()
        page_w, page_h = self._page_sizes[idx]
        target = float(self._targets[idx])

        if self.bbox_jitter > 0:
            bw, bh = s_bbox[2] - s_bbox[0], s_bbox[3] - s_bbox[1]
            s_bbox += rng.uniform(-self.bbox_jitter, self.bbox_jitter, 4).astype(
                np.float32
            ) * np.array([bw, bh, bw, bh], dtype=np.float32)
            lw, lh = l_bbox[2] - l_bbox[0], l_bbox[3] - l_bbox[1]
            l_bbox += rng.uniform(-self.bbox_jitter, self.bbox_jitter, 4).astype(
                np.float32
            ) * np.array([lw, lh, lw, lh], dtype=np.float32)

        geom = torch.from_numpy(
            geom_features(s_bbox, l_bbox, float(page_w), float(page_h))
        )

        img_np = _load_page_image(self._img_paths[self._path_idx[idx]])
        if img_np is None:
            img_np = np.zeros((int(page_h), int(page_w)), dtype=np.uint8)

        s_crop = crop_region(img_np, s_bbox, STRUCT_CROP_SIZE)
        l_crop = crop_region(img_np, l_bbox, LABEL_CROP_SIZE)

        if self.augment:
            # Structures: molecules are rotationally symmetric — full ±180°.
            # Labels: real-world text is semi-upright — limit to ±45°.
            s_crop = _augment_crop(s_crop, rng, max_rot=180.0)
            l_crop = _augment_crop(l_crop, rng, max_rot=45.0)

        return {
            "geom": geom,
            "struct_crop": torch.from_numpy(s_crop),
            "label_crop": torch.from_numpy(l_crop),
            "target": torch.tensor(target, dtype=torch.float32),
        }
