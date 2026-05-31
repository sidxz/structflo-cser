"""Page-level dataset for the Relational Matcher.

Each sample is ONE page: all structure boxes + all label boxes + the ground
truth assignment (which label each structure points to, or the dustbin for
unlabelled structures). Geometry-only — no image decode — so it is very light.

Unlike ``LPSDataset`` (which discards the ~30 % unlabelled structures), every
structure is kept here; unlabelled ones become dustbin targets, which is the
rejection signal the relational matcher trains on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from structflo.cser.relmatch.features import node_features


class RelMatchDataset(Dataset):
    """One page per sample.

    Args:
        data_dir:        split root containing ``ground_truth/`` and ``images/``.
        augment:         enable geometry augmentation (jitter + label dropout).
        bbox_jitter:     uniform coordinate noise as a fraction of box size.
        label_dropout_p: probability of dropping each label (its owning
                         structure then becomes a dustbin target — augments the
                         rejection signal and simulates missed detections).
        seed:            base RNG seed.
    """

    def __init__(
        self,
        data_dir: Path | str,
        augment: bool = False,
        bbox_jitter: float = 0.02,
        label_dropout_p: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.bbox_jitter = bbox_jitter
        self.label_dropout_p = label_dropout_p
        self._seed = seed
        self._build()

    def _build(self) -> None:
        gt_dir = self.data_dir / "ground_truth"
        img_dir = self.data_dir / "images"
        files = sorted(gt_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No GT JSON files in {gt_dir}")

        self.pages: list[dict] = []
        for jp in files:
            stem = jp.stem
            ip = img_dir / f"{stem}.jpg"
            if not ip.exists():
                ip = img_dir / f"{stem}.png"
            if not ip.exists():
                continue
            with Image.open(ip) as im:
                page_w, page_h = im.size

            entries = json.loads(jp.read_text())
            struct_boxes = [e["struct_bbox"] for e in entries]
            # label index per structure (entry order); None if unlabelled
            label_boxes: list[list[float]] = []
            struct_to_label: list[int | None] = []
            for e in entries:
                lb = e.get("label_bbox")
                if lb is not None:
                    struct_to_label.append(len(label_boxes))
                    label_boxes.append(lb)
                else:
                    struct_to_label.append(None)
            if not struct_boxes or not label_boxes:
                continue
            self.pages.append(
                {
                    "struct_boxes": np.asarray(struct_boxes, dtype=np.float32),
                    "label_boxes": np.asarray(label_boxes, dtype=np.float32),
                    "struct_to_label": struct_to_label,
                    "page_w": float(page_w),
                    "page_h": float(page_h),
                }
            )

    def __len__(self) -> int:
        return len(self.pages)

    @staticmethod
    def _jitter(boxes: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
        if frac <= 0 or len(boxes) == 0:
            return boxes
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        scale = np.stack([w, h, w, h], axis=1)
        return boxes + rng.uniform(-frac, frac, boxes.shape).astype(np.float32) * scale

    def __getitem__(self, idx: int) -> dict:
        page = self.pages[idx]
        rng = np.random.default_rng(self._seed ^ (idx * 2654435761 & 0xFFFFFFFF))

        struct_boxes = page["struct_boxes"].copy()
        label_boxes = page["label_boxes"].copy()
        struct_to_label = list(page["struct_to_label"])
        n_s = len(struct_boxes)
        n_l = len(label_boxes)

        # --- augmentation -------------------------------------------------
        if self.augment:
            struct_boxes = self._jitter(struct_boxes, self.bbox_jitter, rng)
            label_boxes = self._jitter(label_boxes, self.bbox_jitter, rng)
            if self.label_dropout_p > 0 and n_l > 1:
                keep = rng.random(n_l) >= self.label_dropout_p
                if keep.sum() == 0:  # never drop every label
                    keep[rng.integers(n_l)] = True
                if not keep.all():
                    remap = {-1: None}
                    new_idx = 0
                    for old in range(n_l):
                        if keep[old]:
                            remap[old] = new_idx
                            new_idx += 1
                        else:
                            remap[old] = None
                    label_boxes = label_boxes[keep]
                    struct_to_label = [
                        remap[k] if k is not None else None for k in struct_to_label
                    ]
                    n_l = len(label_boxes)

        # --- node features (structures first, then labels) ----------------
        boxes = list(struct_boxes) + list(label_boxes)
        classes = [0] * n_s + [1] * n_l
        confs = [1.0] * (n_s + n_l)
        nodes = node_features(boxes, classes, confs, page["page_w"], page["page_h"])

        # --- targets: col per struct, or dustbin (= n_l) ------------------
        target = np.full(n_s, n_l, dtype=np.int64)
        for s, lab in enumerate(struct_to_label):
            if lab is not None:
                target[s] = lab

        return {
            "nodes": torch.from_numpy(nodes),
            "is_struct": torch.tensor([True] * n_s + [False] * n_l),
            "target": torch.from_numpy(target),
            "n_s": n_s,
            "n_l": n_l,
        }


class DetMatchDataset(Dataset):
    """Detection-based page samples (nodes = real YOLO detections).

    Loads the JSON produced by ``scripts/finetune/relmatch/prepare_det_data.py``:
    detected struct/label boxes + confidences, with targets already derived from
    the ground truth (false-positive structs and missed-label structs target the
    dustbin). Matches the inference distribution.

    Args:
        data_dir:    split dir of cached ``<stem>.json`` files.
        augment:     enable light box jitter (detections already carry noise).
        bbox_jitter: uniform coordinate noise as a fraction of box size.
        seed:        base RNG seed.
    """

    def __init__(
        self,
        data_dir: Path | str,
        augment: bool = False,
        bbox_jitter: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.bbox_jitter = bbox_jitter
        self._seed = seed
        self._build()

    def _build(self) -> None:
        files = sorted(self.data_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No cached det JSON in {self.data_dir}")
        self.pages: list[dict] = []
        for jp in files:
            d = json.loads(jp.read_text())
            if not d["struct_boxes"] or not d["label_boxes"]:
                continue
            self.pages.append(
                {
                    "struct_boxes": np.asarray(d["struct_boxes"], dtype=np.float32),
                    "struct_conf": np.asarray(d["struct_conf"], dtype=np.float32),
                    "label_boxes": np.asarray(d["label_boxes"], dtype=np.float32),
                    "label_conf": np.asarray(d["label_conf"], dtype=np.float32),
                    "targets": np.asarray(d["targets"], dtype=np.int64),
                    "page_w": float(d["page_w"]),
                    "page_h": float(d["page_h"]),
                }
            )

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, idx: int) -> dict:
        page = self.pages[idx]
        rng = np.random.default_rng(self._seed ^ (idx * 2654435761 & 0xFFFFFFFF))
        struct_boxes = page["struct_boxes"].copy()
        label_boxes = page["label_boxes"].copy()
        if self.augment and self.bbox_jitter > 0:
            struct_boxes = RelMatchDataset._jitter(struct_boxes, self.bbox_jitter, rng)
            label_boxes = RelMatchDataset._jitter(label_boxes, self.bbox_jitter, rng)

        n_s = len(struct_boxes)
        n_l = len(label_boxes)
        boxes = list(struct_boxes) + list(label_boxes)
        classes = [0] * n_s + [1] * n_l
        confs = list(page["struct_conf"]) + list(page["label_conf"])
        nodes = node_features(boxes, classes, confs, page["page_w"], page["page_h"])
        return {
            "nodes": torch.from_numpy(nodes),
            "is_struct": torch.tensor([True] * n_s + [False] * n_l),
            "target": torch.from_numpy(page["targets"]),
            "n_s": n_s,
            "n_l": n_l,
        }
