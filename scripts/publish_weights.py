#!/usr/bin/env python3
"""Publish trained weights to HuggingFace Hub and update the local registry.

Usage
-----
# Dry-run (shows what would happen, nothing uploaded)
python scripts/publish_weights.py --model cser-detector --version v1.0 --dry-run

# Full publish + auto-patch weights.py
python scripts/publish_weights.py --model cser-detector --version v1.0

# Point to a specific .pt file (default: standard YOLO output location)
python scripts/publish_weights.py --model cser-detector --version v1.0 \\
    --weights-file runs/labels_detect/yolo11l_panels/weights/best.pt

# Publish LPS (Learned Pair Scorer) weights
python scripts/publish_weights.py --model cser-lps --version v1.0

# Pin to a specific package version range (default: current major, e.g. >=0.1.0,<1.0.0)
python scripts/publish_weights.py --model cser-lps --version v1.0 \\
    --requires ">=0.1.0,<1.0.0"
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model → HF Hub repo mapping
# Add a new entry here when a new model is introduced.
# ---------------------------------------------------------------------------
MODEL_REPOS: dict[str, dict] = {
    "cser-detector": {
        "repo_id":  "sidxz/structflo-cser-detector",
        "filename": "best.pt",
    },
    "cser-lps": {
        "repo_id":  "sidxz/structflo-cser-lps",
        "filename": "best.pt",
    },
    "cser-relmatcher": {
        "repo_id":  "sidxz/structflo-cser-relmatcher",
        "filename": "best.pt",
    },
}

# Default weights file paths per model (relative to project root)
DEFAULT_WEIGHTS_PATHS: dict[str, str] = {
    "cser-detector": "runs/labels_detect/yolo11l_panels/weights/best.pt",
    "cser-lps":      "runs/lps/best.pt",
    "cser-relmatcher": "runs/relmatch_det/best.pt",
}

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_PY = PROJECT_ROOT / "structflo" / "cser" / "weights.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_pkg_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("structflo-cser")
    except importlib.metadata.PackageNotFoundError:
        # running from source — read pyproject.toml
        import tomllib
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        return data["project"]["version"]


def default_requires(pkg_version: str) -> str:
    """Return a requires specifier that pins to the current major version."""
    major = pkg_version.split(".")[0]
    next_major = int(major) + 1
    return f">={pkg_version},<{next_major}.0.0"


def upload(repo_id: str, filename: str, weights_file: Path, version: str, dry_run: bool) -> str:
    """Upload weights_file to HF Hub and tag the commit. Returns commit sha."""
    if dry_run:
        print(f"[dry-run] Would upload {weights_file} → {repo_id}/{filename}")
        print(f"[dry-run] Would tag commit as 'weights-{version}'")
        return "dryrun000"

    from huggingface_hub import HfApi
    api = HfApi()

    # Ensure the repo exists (no-op if it already does). New models need this
    # on their first publish; existing repos (detector, lps) are unaffected.
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading {weights_file} ({weights_file.stat().st_size / 1e6:.1f} MB) ...")
    commit = api.upload_file(
        path_or_fileobj=str(weights_file),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"weights {version}",
    )
    sha = commit.oid
    print(f"Uploaded. Commit: {sha[:12]}")

    tag = f"weights-{version}"
    print(f"Creating tag '{tag}' ...")
    api.create_tag(
        repo_id=repo_id,
        repo_type="model",
        tag=tag,
        tag_message=f"weights {version}",
        revision=sha,
    )
    print("Tagged.")
    return sha


# ---------------------------------------------------------------------------
# weights.py patching
# ---------------------------------------------------------------------------

def _load_registry_source() -> str:
    return WEIGHTS_PY.read_text()


def _patch_registry(source: str, model: str, version: str, meta: dict) -> str:
    """Insert a new version entry into REGISTRY[model] and update LATEST."""

    # --- Build the new entry block ------------------------------------------
    indent = "        "  # 8 spaces (inside REGISTRY[model])
    lines = [
        f'{indent}"{version}": {{',
        f'{indent}    "repo_id":  "{meta["repo_id"]}",',
        f'{indent}    "filename": "{meta["filename"]}",',
        f'{indent}    "revision": "weights-{version}",',
        f'{indent}    "sha256":   "{meta["sha256"]}",',
        f'{indent}    "requires": "{meta["requires"]}",',
        f'{indent}}},',
    ]
    new_entry = "\n".join(lines)

    # --- Find the insertion point inside REGISTRY[model] --------------------
    # Look for the closing `},` of the model's sub-dict, just before the
    # outer `}` that closes REGISTRY.  We insert right before the last `    },`
    # that ends the model's block.

    # Pattern: the line `    },` that closes the model's dict, preceded by
    # either the commented-out placeholder block or an existing entry.
    # We find the last occurrence of `    },` inside the model block.

    # Simpler: find `    "cser-detector": {` ... `    },` span and insert
    # the new entry before the closing `    },`.
    model_open = re.search(
        rf'(\s+"{re.escape(model)}":\s*\{{)',
        source,
    )
    if model_open is None:
        raise ValueError(f"Could not find REGISTRY[\"{model}\"] in {WEIGHTS_PY}")

    # Find the matching closing `    },` after the model block opens
    block_start = model_open.end()
    # Scan forward to find the balanced closing brace of this sub-dict
    depth = 1
    i = block_start
    while i < len(source) and depth > 0:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    block_end = i  # points just past the closing `}`

    # Insert new_entry just before the closing `}` of the model's dict
    # (i.e. at block_end - 1, before the `}`)
    close_brace_pos = block_end - 1
    # Back up past any trailing whitespace/newline before the `}`
    insert_pos = close_brace_pos
    # Find the last newline before the closing brace to insert after it
    last_nl = source.rfind("\n", block_start, close_brace_pos)
    insert_pos = last_nl + 1 if last_nl != -1 else close_brace_pos

    patched = source[:insert_pos] + new_entry + "\n" + source[insert_pos:]

    # --- Update LATEST[model] -----------------------------------------------
    patched = re.sub(
        rf'("{re.escape(model)}":\s*)(?:None|"[^"]*")',
        rf'\g<1>"{version}"',
        patched,
    )

    return patched


def patch_weights_py(model: str, version: str, meta: dict, dry_run: bool) -> None:
    source = _load_registry_source()
    patched = _patch_registry(source, model, version, meta)

    if dry_run:
        print(f"\n[dry-run] Would patch {WEIGHTS_PY} with:")
        print(f"  REGISTRY[\"{model}\"][\"{version}\"] = {{...}}")
        print(f"  LATEST[\"{model}\"] = \"{version}\"")
        return

    WEIGHTS_PY.write_text(patched)
    print(f"Patched {WEIGHTS_PY.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Upload trained weights to HF Hub and register them in weights.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REPOS),
        help="Model name (must exist in MODEL_REPOS)",
    )
    p.add_argument(
        "--version",
        required=True,
        help="Version tag, e.g. v1.0  (will become HF tag 'weights-v1.0')",
    )
    p.add_argument(
        "--weights-file",
        default=None,
        help="Path to .pt file.  Defaults to the standard YOLO output location.",
    )
    p.add_argument(
        "--filename",
        default=None,
        help="Filename inside the HF repo (default: the model's registered filename). "
        "Use to host an extra checkpoint variant in the same repo without "
        "changing the registry default.",
    )
    p.add_argument(
        "--no-registry",
        action="store_true",
        help="Upload the file only; do not patch weights.py (for variant uploads).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    p.add_argument(
        "--requires",
        default=None,
        help="PEP 440 specifier for compatible pkg versions, e.g. '>=0.1.0,<1.0.0'. "
             "Defaults to current installed major: >=X.Y.Z,<(X+1).0.0",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without uploading or modifying files.",
    )
    args = p.parse_args()

    # --- Resolve weights file -----------------------------------------------
    weights_file = Path(args.weights_file) if args.weights_file else (
        PROJECT_ROOT / DEFAULT_WEIGHTS_PATHS[args.model]
    )
    if not weights_file.exists():
        p.error(
            f"Weights file not found: {weights_file}\n"
            f"Pass --weights-file to point to the correct location."
        )

    # --- Resolve requires specifier -----------------------------------------
    pkg_ver = current_pkg_version()
    requires = args.requires or default_requires(pkg_ver)

    # --- Summary ------------------------------------------------------------
    repo_info = MODEL_REPOS[args.model]
    repo_filename = args.filename or repo_info["filename"]
    hf_tag = f"weights-{args.version}"
    sha256 = sha256_of(weights_file)

    print(f"Model:         {args.model}")
    print(f"Version:       {args.version}  (HF tag: {hf_tag})")
    print(f"Weights file:  {weights_file}  ({weights_file.stat().st_size / 1e6:.1f} MB)")
    print(f"HF repo:       {repo_info['repo_id']}")
    print(f"HF filename:   {repo_filename}")
    print(f"sha256:        {sha256}")
    print(f"requires:      {requires}")
    print(f"Dry-run:       {args.dry_run}")
    print()

    if not args.dry_run and not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # --- Upload -------------------------------------------------------------
    upload(
        repo_id=repo_info["repo_id"],
        filename=repo_filename,
        weights_file=weights_file,
        version=args.version,
        dry_run=args.dry_run,
    )

    # --- Patch weights.py ---------------------------------------------------
    if args.no_registry:
        print("Skipping weights.py patch (--no-registry).")
    else:
        meta = {
            "repo_id":  repo_info["repo_id"],
            "filename": repo_filename,
            "sha256":   sha256,
            "requires": requires,
        }
        patch_weights_py(args.model, args.version, meta, dry_run=args.dry_run)

    print()
    if args.dry_run:
        print("Dry-run complete. Re-run without --dry-run to publish.")
    else:
        print("Done. Commit the change to weights.py:")
        print("  git add structflo/cser/weights.py")
        print(f"  git commit -m 'weights: publish {args.version}'")


if __name__ == "__main__":
    main()
