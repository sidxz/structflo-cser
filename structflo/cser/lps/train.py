"""Training script for the Learned Pair Scorer.

Entry point: ``sf-train-lps``

Usage::

    sf-train-lps --data-dir data/generated --epochs 30

    # Custom batch / workers
    sf-train-lps --data-dir data/generated --epochs 50 --batch 1024 --workers 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from structflo.cser.lps.dataset import LPSDataset, PageGroupSampler
from structflo.cser.lps.scorer import PairScorer, save_checkpoint

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "generated"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "runs" / "lps"


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """One training epoch.  Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    bar = tqdm(loader, desc=f"Epoch {epoch:>3} train", leave=False, unit="batch")
    for batch in bar:
        geom = batch["geom"].to(device)
        sc = batch["struct_crop"].to(device)
        lc = batch["label_crop"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        logits = model(sc, lc, geom)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = target.size(0)
        total_loss += loss.item() * bs
        preds = (logits.detach().sigmoid() >= 0.5).float()
        correct += (preds == target).sum().item()
        n += bs
        bar.set_postfix(loss=f"{total_loss / n:.4f}", acc=f"{correct / n:.2%}")

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """One validation epoch.  Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    bar = tqdm(loader, desc=f"Epoch {epoch:>3}   val", leave=False, unit="batch")
    for batch in bar:
        geom = batch["geom"].to(device)
        sc = batch["struct_crop"].to(device)
        lc = batch["label_crop"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        logits = model(sc, lc, geom)
        loss = criterion(logits, target)

        bs = target.size(0)
        total_loss += loss.item() * bs
        preds = (logits.sigmoid() >= 0.5).float()
        correct += (preds == target).sum().item()
        n += bs
        bar.set_postfix(loss=f"{total_loss / n:.4f}", acc=f"{correct / n:.2%}")

    return total_loss / max(n, 1), correct / max(n, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    data_dir: Path = _DEFAULT_DATA_DIR,
    output_dir: Path = _DEFAULT_OUT_DIR,
    epochs: int = 30,
    batch_size: int = 1024,
    neg_per_pos: int = 3,
    bbox_jitter: float = 0.02,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    device_str: str = "cuda",
    seed: int = 42,
    resume: Path | None = None,
    finetune: Path | None = None,
    reject_negatives: bool = False,
) -> Path:
    """Train the PairScorer and return the path to the best checkpoint."""
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lps] device      : {device}")
    print(f"[lps] data        : {data_dir}")
    print(f"[lps] output      : {output_dir}")
    print(f"[lps] epochs      : {epochs}  batch: {batch_size}")
    if resume:
        print(f"[lps] resume      : {resume}")
    if finetune:
        print(f"[lps] finetune    : {finetune}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("[lps] building training dataset …")
    t0 = time.time()
    train_ds = LPSDataset(
        data_dir / "train",
        neg_per_pos=neg_per_pos,
        bbox_jitter=bbox_jitter,
        augment=True,  # rotation/flip/brightness augmentation
        seed=seed,
        reject_negatives=reject_negatives,
    )
    if reject_negatives:
        print("[lps] reject-negatives ENABLED (unlabelled + fp_negatives/ as target-0)")
    print(f"[lps] train pairs : {len(train_ds):,}  ({time.time() - t0:.1f}s)")

    print("[lps] building validation dataset …")
    t0 = time.time()
    val_ds = LPSDataset(
        data_dir / "val",
        neg_per_pos=neg_per_pos,
        bbox_jitter=0.0,
        augment=False,
        seed=seed,
    )
    print(f"[lps] val pairs   : {len(val_ds):,}  ({time.time() - t0:.1f}s)")

    pw = train_ds.pos_weight()
    print(f"[lps] pos_weight  : {pw:.2f}")

    # ------------------------------------------------------------------
    # DataLoaders
    # spawn: avoids inheriting CUDA/libjpeg state from the main process.
    # persistent_workers: keeps workers alive across epochs — critical with
    #   spawn (avoids re-importing torch each epoch) and for LRU image cache.
    # prefetch_factor: workers queue ahead so the GPU is never starved.
    # PageGroupSampler: yields all samples from a page consecutively so the
    #   per-worker LRU image cache gets ~20 hits per JPEG decode, not 1.
    # ------------------------------------------------------------------
    train_sampler = PageGroupSampler(train_ds._path_idx, shuffle=True, seed=seed)
    val_sampler = PageGroupSampler(val_ds._path_idx, shuffle=False, seed=seed)

    _nw = num_workers
    loader_kw: dict = dict(
        batch_size=batch_size,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        multiprocessing_context="spawn",
        persistent_workers=(_nw > 0),
    )
    if _nw > 0:
        loader_kw["prefetch_factor"] = 8

    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kw)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PairScorer()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lps] parameters  : {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------
    # Load checkpoint (resume vs finetune)
    # ------------------------------------------------------------------
    start_epoch = 1
    best_acc = 0.0

    if resume is not None and finetune is not None:
        raise ValueError("Cannot use both --resume and --finetune")

    if finetune is not None:
        # Fine-tune: load weights only, fresh optimizer/scheduler/epoch
        ckpt = torch.load(finetune, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[lps] loaded weights from {finetune}  (fresh optimizer, lr={lr})")

    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("val_accuracy", 0.0)
        print(
            f"[lps] resumed from epoch {start_epoch - 1}  (best acc so far: {best_acc:.2%})"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    print(
        f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>7}  {'ValAcc':>6}  {'LR':>8}"
    )
    print("-" * 58)

    for epoch in range(start_epoch, epochs + 1):
        train_sampler.set_epoch(epoch)
        t_start = time.time()
        tr_loss, tr_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        vl_loss, vl_acc = _val_epoch(model, val_loader, criterion, device, epoch)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start
        marker = " *" if vl_acc > best_acc else ""

        print(
            f"{epoch:>5}  {tr_loss:>9.4f}  {tr_acc:>7.2%}  "
            f"{vl_loss:>7.4f}  {vl_acc:>5.2%}  {lr_now:>8.2e}"
            f"  {elapsed:.0f}s{marker}"
        )

        # Always overwrite last — includes optimizer/scheduler state for resume.
        save_checkpoint(
            model,
            last_path,
            epoch=epoch,
            val_accuracy=vl_acc,
            val_loss=vl_loss,
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
        )

        if vl_acc > best_acc:
            best_acc = vl_acc
            save_checkpoint(
                model,
                best_path,
                epoch=epoch,
                val_accuracy=vl_acc,
                val_loss=vl_loss,
            )

    print(f"\n[lps] best val accuracy : {best_acc:.2%}")
    print(f"[lps] best checkpoint   : {best_path}")
    print(f"[lps] last checkpoint   : {last_path}")
    return best_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train the Learned Pair Scorer (LPS) for structure-label association"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of generated data (must contain train/ and val/ subdirs)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for checkpoints (default: runs/lps/)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024, help="Batch size")
    p.add_argument(
        "--neg-per-pos",
        type=int,
        default=3,
        help="Hard negatives per positive pair (default: 3)",
    )
    p.add_argument(
        "--bbox-jitter",
        type=float,
        default=0.02,
        help="Bbox coordinate jitter fraction (default: 0.02)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from scorer_last.pt (restores model, optimizer, scheduler, epoch)",
    )
    p.add_argument(
        "--finetune",
        type=Path,
        default=None,
        help="Fine-tune from a checkpoint (loads weights only, fresh optimizer/scheduler)",
    )
    p.add_argument(
        "--reject-negatives",
        action="store_true",
        help="Add rejection negatives (unlabelled structures + fp_negatives/ sidecar) to the train set",
    )

    args = p.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        neg_per_pos=args.neg_per_pos,
        bbox_jitter=args.bbox_jitter,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        device_str=args.device,
        seed=args.seed,
        resume=args.resume,
        finetune=args.finetune,
        reject_negatives=args.reject_negatives,
    )


if __name__ == "__main__":
    main()
