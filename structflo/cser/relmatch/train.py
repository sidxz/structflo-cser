"""Training script for the Relational Matcher (SetMatcher).

Entry point: ``sf-train-relmatch``

Assignment NLL loss over per-page Sinkhorn outputs; selection on val
per-structure assignment accuracy (each structure routed to its true label or
the dustbin). Geometry-only and tiny, so training is fast — one page per
forward with gradient accumulation rather than padded batching.

Usage::

    sf-train-relmatch --data-dir data/finetune/lps --epochs 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from structflo.cser.relmatch.dataset import DetMatchDataset, RelMatchDataset
from structflo.cser.relmatch.model import SetMatcher, save_checkpoint

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "finetune" / "lps"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "runs" / "relmatch"


def _page_nll(model: SetMatcher, sample: dict, device: torch.device) -> torch.Tensor:
    nodes = sample["nodes"].to(device)
    is_struct = sample["is_struct"].to(device)
    target = sample["target"].to(device)
    Z = model(nodes, is_struct)  # (n_s+1, n_l+1)
    n_s = target.shape[0]
    rows = torch.arange(n_s, device=device)
    return -Z[rows, target].mean()


@torch.no_grad()
def _val_metrics(model: SetMatcher, ds: RelMatchDataset, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for i in range(len(ds)):
        sample = ds[i]
        nodes = sample["nodes"].to(device)
        is_struct = sample["is_struct"].to(device)
        target = sample["target"].to(device)
        Z = model(nodes, is_struct)
        n_s = target.shape[0]
        rows = torch.arange(n_s, device=device)
        total_loss += float(-Z[rows, target].mean()) * n_s
        pred = Z.argmax(dim=1)[:n_s]  # incl. dustbin column
        correct += int((pred == target).sum())
        n += n_s
    return total_loss / max(n, 1), correct / max(n, 1)


def train(
    data_dir: Path = _DEFAULT_DATA_DIR,
    output_dir: Path = _DEFAULT_OUT_DIR,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    accum: int = 16,
    d_model: int = 64,
    n_layers: int = 3,
    n_heads: int = 4,
    sinkhorn_iters: int = 50,
    bbox_jitter: float = 0.02,
    label_dropout: float = 0.1,
    det_data_dir: Path | None = None,
    device_str: str = "cuda",
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[relmatch] device : {device}")
    print(f"[relmatch] output : {output_dir}")

    if det_data_dir is not None:
        print(f"[relmatch] data   : DETECTION boxes @ {det_data_dir}")
        train_ds = DetMatchDataset(
            det_data_dir / "train", augment=True, bbox_jitter=0.01, seed=seed
        )
        val_ds = DetMatchDataset(det_data_dir / "val", augment=False, seed=seed)
    else:
        print(f"[relmatch] data   : GT boxes @ {data_dir}")
        train_ds = RelMatchDataset(
            data_dir / "train",
            augment=True,
            bbox_jitter=bbox_jitter,
            label_dropout_p=label_dropout,
            seed=seed,
        )
        val_ds = RelMatchDataset(data_dir / "val", augment=False, seed=seed)
    print(f"[relmatch] train pages: {len(train_ds):,}   val pages: {len(val_ds):,}")

    model = SetMatcher(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        sinkhorn_iters=sinkhorn_iters,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[relmatch] parameters : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_acc = 0.0
    order = np.arange(len(train_ds))

    print(f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>8}  {'ValAcc':>7}  {'LR':>8}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        t0 = time.time()
        running = 0.0
        n = 0
        optimizer.zero_grad()
        for step, idx in enumerate(order, 1):
            sample = train_ds[int(idx)]
            loss = _page_nll(model, sample, device)
            (loss / accum).backward()
            running += float(loss)
            n += 1
            if step % accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()

        vl_loss, vl_acc = _val_metrics(model, val_ds, device)
        lr_now = scheduler.get_last_lr()[0]
        marker = " *" if vl_acc > best_acc else ""
        print(
            f"{epoch:>5}  {running / max(n, 1):>9.4f}  {vl_loss:>8.4f}  "
            f"{vl_acc:>6.2%}  {lr_now:>8.2e}  {time.time() - t0:.0f}s{marker}"
        )

        save_checkpoint(
            model, last_path, epoch=epoch, val_accuracy=vl_acc, val_loss=vl_loss
        )
        if vl_acc > best_acc:
            best_acc = vl_acc
            save_checkpoint(
                model, best_path, epoch=epoch, val_accuracy=vl_acc, val_loss=vl_loss
            )

    print(f"\n[relmatch] best val acc : {best_acc:.2%}")
    print(f"[relmatch] best ckpt    : {best_path}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Relational Matcher (SetMatcher)")
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=16, help="pages per optimizer step")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--sinkhorn-iters", type=int, default=50)
    p.add_argument("--bbox-jitter", type=float, default=0.02)
    p.add_argument("--label-dropout", type=float, default=0.1)
    p.add_argument(
        "--det-data-dir",
        type=Path,
        default=None,
        help="Train on cached detection boxes (from prepare_det_data.py) instead of GT boxes",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accum=args.accum,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        sinkhorn_iters=args.sinkhorn_iters,
        bbox_jitter=args.bbox_jitter,
        label_dropout=args.label_dropout,
        det_data_dir=args.det_data_dir,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
