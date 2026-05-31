"""SetMatcher — relational structure↔label matcher (Rung 2).

Treats every detection on a page as a node. A transformer encoder lets each
node attend to every other node (structure↔structure, label↔label,
structure↔label) so embeddings are layout-aware. Struct/label embedding dot
products form an affinity matrix; a learnable dustbin row/column plus a
log-space Sinkhorn layer turn it into a soft assignment that natively handles
unmatched structures (the ~30 % unlabelled) and unmatched labels.

This is the optimal-transport matching formulation (cf. SuperGlue, Sarlin et
al. 2020) retargeted from keypoint matching to document structure-label
association. Geometry-only by design — no visual branch — so it is a clean,
distinct third matcher alongside HungarianMatcher and the visual LPS.

Forward operates on ONE page at a time (variable node counts); training uses
gradient accumulation over pages rather than padded batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from structflo.cser.relmatch.features import NODE_DIM


# ---------------------------------------------------------------------------
# Sinkhorn optimal-transport layer (log-space, numerically stable)
# ---------------------------------------------------------------------------


def _log_sinkhorn(
    scores: Tensor,  # (M, N) affinity (log-space, unnormalised)
    bin_score: Tensor,  # scalar dustbin score
    iters: int,
) -> Tensor:
    """Differentiable optimal transport with dustbins.

    Returns the log-assignment matrix ``Z`` of shape ``(M+1, N+1)`` where the
    extra row/column absorb unmatched mass. Each real row/column has marginal
    mass 1; the dustbins take up the slack.
    """
    m, n = scores.shape
    one = scores.new_tensor(1.0)
    ms, ns = m * one, n * one

    bins0 = bin_score.expand(m, 1)
    bins1 = bin_score.expand(1, n)
    alpha = bin_score.expand(1, 1)

    couplings = torch.cat(
        [
            torch.cat([scores, bins0], dim=1),
            torch.cat([bins1, alpha], dim=1),
        ],
        dim=0,
    )  # (M+1, N+1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])  # (M+1,)
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])  # (N+1,)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(couplings + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(couplings + u[:, None], dim=0)
    Z = couplings + u[:, None] + v[None, :]
    return Z - norm  # undo the (M+N) scaling → log-probabilities


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SetMatcher(nn.Module):
    """Relational set-to-set matcher with a Sinkhorn assignment head.

    Args:
        d_model:      token / embedding dimension.
        n_layers:     transformer encoder layers.
        n_heads:      attention heads.
        dim_ff:       feed-forward width.
        dropout:      dropout in the encoder.
        sinkhorn_iters: Sinkhorn normalisation iterations.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_ff: int = 128,
        dropout: float = 0.1,
        sinkhorn_iters: int = 50,
        node_dim: int = NODE_DIM,
    ) -> None:
        super().__init__()
        self.cfg = dict(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=dropout,
            sinkhorn_iters=sinkhorn_iters,
            node_dim=node_dim,
        )
        self.sinkhorn_iters = sinkhorn_iters
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        self._scale = float(d_model) ** 0.5

    def forward(self, nodes: Tensor, is_struct: Tensor) -> Tensor:
        """Score one page.

        Args:
            nodes:     (Nnodes, node_dim) float — all detections on the page.
            is_struct: (Nnodes,) bool — True for structures, False for labels.
                       Structures keep input order; labels keep input order.

        Returns:
            Log-assignment matrix ``Z`` of shape ``(n_s + 1, n_l + 1)``.
        """
        x = self.input_proj(nodes).unsqueeze(0)  # (1, Nnodes, d)
        x = self.encoder(x).squeeze(0)  # (Nnodes, d)
        x = self.out_proj(x)
        sf = x[is_struct]  # (n_s, d)
        lf = x[~is_struct]  # (n_l, d)
        affinity = (sf @ lf.t()) / self._scale  # (n_s, n_l)
        return _log_sinkhorn(affinity, self.bin_score, self.sinkhorn_iters)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_STATE_DICT_KEY = "state_dict"
_CFG_KEY = "model_cfg"


def save_checkpoint(model: SetMatcher, path: Path, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({_STATE_DICT_KEY: model.state_dict(), _CFG_KEY: model.cfg, **meta}, path)


def load_checkpoint(
    path: Path | str,
    device: str = "cpu",
) -> tuple[SetMatcher, dict[str, Any]]:
    """Load a SetMatcher checkpoint, reconstructing the architecture from cfg."""
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get(_CFG_KEY, {})
    model = SetMatcher(**cfg) if cfg else SetMatcher()
    model.load_state_dict(ckpt[_STATE_DICT_KEY])
    model.to(device)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k not in (_STATE_DICT_KEY, _CFG_KEY)}
    return model, meta
