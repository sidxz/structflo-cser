"""structflo.cser.relmatch — Relational set-to-set structure-label matcher.

A geometry-only matcher that reasons over all detections on a page jointly via
attention and a Sinkhorn optimal-transport assignment with learnable dustbins
(unmatched structures / labels). A third strategy alongside ``HungarianMatcher``
(distance) and ``LearnedMatcher`` (visual LPS).

Usage::

    from structflo.cser.relmatch import RelationalMatcher
    from structflo.cser.pipeline import ChemPipeline

    pipeline = ChemPipeline(matcher=RelationalMatcher("runs/relmatch/best.pt"))

Training::

    sf-train-relmatch --data-dir data/finetune/lps --epochs 60
"""

from structflo.cser.relmatch.matcher import RelationalMatcher

__all__ = ["RelationalMatcher"]
