"""RelationalMatcher — BaseMatcher backed by the SetMatcher (Rung 2).

A third, geometry-only matching strategy alongside ``HungarianMatcher``
(centroid distance) and ``LearnedMatcher`` (visual LPS). It scores all
structures and labels on a page jointly via attention, then reads a 1-to-1
assignment off the Sinkhorn output. Structures whose best match loses to the
learned dustbin are left unmatched (principled rejection — no min_score knob
required, though one is exposed for parity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from structflo.cser.pipeline.matcher import BaseMatcher
from structflo.cser.pipeline.models import CompoundPair, Detection
from structflo.cser.relmatch.features import node_features
from structflo.cser.relmatch.model import load_checkpoint


class RelationalMatcher(BaseMatcher):
    """Pair structures with labels using the relational SetMatcher.

    Args:
        weights:    path to a SetMatcher ``.pt`` checkpoint, a version tag, or
                    ``None`` to resolve the latest published ``cser-relmatcher``.
        min_score:  extra floor on assignment probability (default 0.0 — rely
                    on the learned dustbin for rejection).
        device:     torch device string.
    """

    def __init__(
        self,
        weights: Path | str | None = None,
        min_score: float = 0.0,
        dustbin_margin: float = 0.0,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            dustbin_margin: relaxes rejection — accept a pair when its log-score
                is within ``dustbin_margin`` of the structure's dustbin score
                (``core >= dust - margin``). Larger ⇒ more pairs (higher recall,
                lower precision). 0.0 = strict dustbin rule.
        """
        self.min_score = min_score
        self.dustbin_margin = dustbin_margin
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = self._load(weights)

    def _load(self, weights: Path | str | None) -> torch.nn.Module:
        from structflo.cser.weights import resolve_weights

        path = resolve_weights("cser-relmatcher", version=weights)
        model, _ = load_checkpoint(path, device=str(self._device))
        model.eval()
        return model

    @staticmethod
    def _page_size(
        detections: list[Detection], image: np.ndarray | None
    ) -> tuple[float, float]:
        if image is not None:
            h, w = image.shape[:2]
            return float(w), float(h)
        if detections:
            return (
                float(max(d.bbox.x2 for d in detections)),
                float(max(d.bbox.y2 for d in detections)),
            )
        return 2480.0, 3508.0

    def match(
        self,
        detections: list[Detection],
        image: np.ndarray | None = None,
    ) -> list[CompoundPair]:
        structures = [d for d in detections if d.class_id == 0]
        labels = [d for d in detections if d.class_id == 1]
        if not structures or not labels:
            return []

        page_w, page_h = self._page_size(detections, image)
        n_s, n_l = len(structures), len(labels)

        boxes = [d.bbox.as_list() for d in structures] + [
            d.bbox.as_list() for d in labels
        ]
        classes = [0] * n_s + [1] * n_l
        confs = [d.conf for d in structures] + [d.conf for d in labels]
        nodes = torch.from_numpy(
            node_features(boxes, classes, confs, page_w, page_h)
        ).to(self._device)
        is_struct = torch.tensor([True] * n_s + [False] * n_l, device=self._device)

        with torch.no_grad():
            Z = self._model(nodes, is_struct)  # (n_s+1, n_l+1) log-assignment
        Zc = Z.cpu().numpy()

        core = Zc[:n_s, :n_l]
        dust_col = Zc[:n_s, n_l]  # each struct's "no-match" log-score
        row_ind, col_ind = linear_sum_assignment(-core)

        pairs: list[CompoundPair] = []
        for r, c in zip(row_ind, col_ind):
            prob = float(np.exp(core[r, c]))
            # accept if the match beats this structure's dustbin (within margin) and floor
            if core[r, c] >= dust_col[r] - self.dustbin_margin and prob >= self.min_score:
                pairs.append(
                    CompoundPair(
                        structure=structures[r],
                        label=labels[c],
                        match_distance=float(1.0 - prob),
                        match_confidence=prob,
                    )
                )
        return pairs
