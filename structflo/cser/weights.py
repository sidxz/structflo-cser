"""Weights registry and auto-download helpers.

Weights are versioned independently of the Python package.  The registry is
organised by *model name* so that multiple models (each with their own HF Hub
repo and version history) can coexist without interfering.

Separation of concerns
----------------------
- Retraining on more data   → new weights tag, no code release needed.
- Bug fix / new CLI flag     → new pip release, existing weights unchanged.
- Architecture change        → new weights major + new pkg major, with a clear
                               compatibility error if they are mismatched.

Versioning convention
---------------------
- Weights tags on HF Hub:  ``weights-v1.0``, ``weights-v1.1``, ``weights-v2.0``
- Package versions on PyPI: ``0.1.0``, ``0.2.0``, ``1.0.0``  (PEP 440 semver)

The ``requires`` field in each entry uses PEP 440 specifiers, e.g.
``">=0.1.0,<1.0.0"``.  It expresses which package versions can load that
particular weights file (architecture compatibility).

Usage
-----
>>> from structflo.cser.weights import resolve_weights
>>> path = resolve_weights("cser-detector")                     # latest, auto-download
>>> path = resolve_weights("cser-detector", version="v1.0")     # pin a version
>>> path = resolve_weights("cser-detector", version="/my.pt")   # local file, no download
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Top-level keys are model names.  Under each model, version tags map to
# metadata dicts with the following fields:
#
#   repo_id  : HuggingFace model repo  (e.g. "structflo/cser-detector")
#   filename : file inside the repo    (e.g. "best.pt")
#   revision : git tag on HF Hub      (e.g. "weights-v1.0")
#   sha256   : (optional) hex digest for integrity verification
#   requires : PEP 440 specifier for compatible structflo-cser pkg versions
#
# To publish new weights:
#   1. Push best.pt to HF Hub and create a git tag (e.g. "weights-v1.1")
#   2. Add an entry under the appropriate model below
#   3. Bump the corresponding LATEST entry
#
REGISTRY: dict[str, dict[str, dict]] = {
    "cser-detector": {
        # "v1.0": {
        #     "repo_id":  "structflo/cser-detector",
        #     "filename": "best.pt",
        #     "revision": "weights-v1.0",
        #     "sha256":   "abc123...",
        #     "requires": ">=0.1.0,<1.0.0",
        # },
        "v0.1": {
            "repo_id": "sidxz/structflo-cser-detector",
            "filename": "best.pt",
            "revision": "weights-v0.1",
            "sha256": "2b139a7e78a6721f16187967bd782acf61e4c7389d2097ea05daeb942cda4bf5",
            "requires": ">=0.1.0,<1.0.0",
        },
        "v0.2": {
            "repo_id": "sidxz/structflo-cser-detector",
            "filename": "best.pt",
            "revision": "weights-v0.2",
            "sha256": "8b6d7373bedef50e25a48ef6ac333962d18861d30f98ee33f393cdf3d38f1c26",
            "requires": ">=0.1.0,<1.0.0",
        },
        "v0.3": {
            "repo_id":  "sidxz/structflo-cser-detector",
            "filename": "best.pt",
            "revision": "weights-v0.3",
            "sha256":   "b45ec5c0f1b2919a6bdda52051800f5610d9008f1ad7b3db5041fa222abb8626",
            "requires": ">=0.1.0,<1.0.0",
        },
    },
    "cser-lps": {
        # Populate after first training run and HF Hub publish:
        # "v1.0": {
        #     "repo_id":  "sidxz/structflo-cser-lps",
        #     "filename": "scorer_best.pt",
        #     "revision": "weights-v1.0",
        #     "sha256":   "...",
        #     "requires": ">=0.1.0,<1.0.0",
        # },
        "v0.1": {
            "repo_id": "sidxz/structflo-cser-lps",
            "filename": "best.pt",
            "revision": "weights-v0.1",
            "sha256": "6b80327fe13b67b183e86558e8bac9141e6d1412e948c745b7ba5cfeb2df7b7d",
            "requires": ">=0.1.0,<1.0.0",
        },
        "v0.2": {
            "repo_id":  "sidxz/structflo-cser-lps",
            "filename": "best.pt",
            "revision": "weights-v0.2",
            "sha256":   "67d8c129645415ecf19aa62ac818ee894cdc0b70519ff67d1c84a73152ca11e6",
            "requires": ">=0.1.0,<1.0.0",
        },
    },
}

# The version resolved when the caller does not specify one.
# Keep in sync with REGISTRY — point to the newest entry per model.
LATEST: dict[str, str | None] = {
    "cser-detector": "v0.3",
    "cser-lps": "v0.2",  # set to "v1.0" after first publish
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WeightsNotFoundError(FileNotFoundError):
    """Raised when a weights file or version tag cannot be resolved."""


class WeightsCompatibilityError(RuntimeError):
    """Raised when weights require a different version of structflo-cser."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_weights(
    model: str,
    version: str | Path | None = None,
) -> Path:
    """Return a local path to a ``.pt`` weights file for *model*.

    Parameters
    ----------
    model:
        Registered model name, e.g. ``"cser-detector"``.
    version:
        One of:

        * ``None``  — download / return cached copy of the *LATEST* version
          for this model.
        * A version tag string (e.g. ``"v1.0"``) — look up in the registry
          and auto-download.
        * An existing file path (``str`` or ``Path``) — used as-is, no
          download performed.

    Returns
    -------
    Path
        Absolute path to the resolved ``.pt`` file.

    Raises
    ------
    WeightsNotFoundError
        The path does not exist, the model is unknown, or the version tag is
        not in the registry.
    WeightsCompatibilityError
        The weights entry declares a ``requires`` specifier that is not
        satisfied by the installed ``structflo-cser`` package.
    """
    if model not in REGISTRY:
        known = list(REGISTRY) or ["(none registered)"]
        raise WeightsNotFoundError(f"Unknown model '{model}'.  Known models: {known}")

    # --- Case 1: explicit local path ----------------------------------------
    if version is not None:
        candidate = Path(version)
        if candidate.exists():
            return candidate
        # Looks like a path but doesn't exist on disk
        s = str(version)
        if "/" in s or "\\" in s or s.endswith(".pt"):
            raise WeightsNotFoundError(f"Weights file not found: {candidate}")

    # --- Case 2: version tag (or None → LATEST) -----------------------------
    tag = str(version) if version is not None else LATEST.get(model)

    if tag is None:
        raise WeightsNotFoundError(
            f"No weights have been published yet for model '{model}'.  "
            f"Pass an explicit local path:  version='/path/to/best.pt'"
        )

    model_registry = REGISTRY[model]
    if tag not in model_registry:
        available = list(model_registry) or ["(none published yet)"]
        raise WeightsNotFoundError(
            f"Unknown weights version '{tag}' for model '{model}'.  "
            f"Available: {available}"
        )

    meta = model_registry[tag]

    # --- Compatibility check -------------------------------------------------
    if "requires" in meta:
        _check_compatibility(model, tag, meta["requires"])

    # --- Download (cached by huggingface_hub) --------------------------------
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic weight download.  "
            "Install it with:  uv add huggingface_hub"
        ) from exc

    local_path = hf_hub_download(
        repo_id=meta["repo_id"],
        filename=meta["filename"],
        revision=meta["revision"],
    )
    return Path(local_path)


def list_versions(model: str) -> list[str]:
    """Return all registered version tags for *model*, in registry order."""
    if model not in REGISTRY:
        raise WeightsNotFoundError(f"Unknown model '{model}'.")
    return list(REGISTRY[model])


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(REGISTRY)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_compatibility(model: str, weights_version: str, requires: str) -> None:
    try:
        pkg_ver = Version(importlib.metadata.version("structflo-cser"))
    except importlib.metadata.PackageNotFoundError:
        return  # running from a source checkout — skip the check

    if pkg_ver not in SpecifierSet(requires):
        raise WeightsCompatibilityError(
            f"Weights '{model}/{weights_version}' require "
            f"structflo-cser{requires} (you have {pkg_ver}).  "
            f"Upgrade with:  uv add 'structflo-cser{requires}'"
        )
