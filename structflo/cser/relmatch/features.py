"""Per-node geometric features for the Relational Matcher.

Unlike ``lps.features.geom_features`` (which encodes a *pair*), this encodes a
*single detection* as a node in the page graph.  Relational reasoning between
nodes is left to the attention layers, so each node carries only its own
geometry, class and confidence — page-normalised so the representation is
resolution-independent.

Pure numpy; no torch / pipeline deps.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

NODE_DIM = 9  # length of the per-node feature vector produced by node_features()


def node_features(
    boxes: Sequence[Sequence[float]],  # list of [x1,y1,x2,y2] pixels
    classes: Sequence[int],  # 0 = structure, 1 = label
    confs: Sequence[float],
    page_w: float,
    page_h: float,
) -> np.ndarray:
    """Return a float32 array of shape ``(N, NODE_DIM)``.

    Feature layout (9 values):
        0  cx_norm        centroid x / page_w
        1  cy_norm        centroid y / page_h
        2  w_norm         width  / page_w
        3  h_norm         height / page_h
        4  log_aspect     log(width / height)        (orientation, scale-free)
        5  sqrt_area_norm sqrt(area) / sqrt(page area)
        6  is_struct      1.0 if class 0 else 0.0
        7  is_label       1.0 if class 1 else 0.0
        8  conf           detection confidence (1.0 for GT boxes)
    """
    pw = max(float(page_w), 1.0)
    ph = max(float(page_h), 1.0)
    page_diag_area = math.sqrt(pw * ph)

    out = np.zeros((len(boxes), NODE_DIM), dtype=np.float32)
    for i, (b, c, cf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = (float(v) for v in b)
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        out[i, 0] = cx / pw
        out[i, 1] = cy / ph
        out[i, 2] = w / pw
        out[i, 3] = h / ph
        out[i, 4] = math.log(w / h)
        out[i, 5] = math.sqrt(w * h) / page_diag_area
        out[i, 6] = 1.0 if int(c) == 0 else 0.0
        out[i, 7] = 1.0 if int(c) == 1 else 0.0
        out[i, 8] = float(cf)
    return out
