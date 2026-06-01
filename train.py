"""Train fusion models to approximate full-video CLIP embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FusionFeatureDataset
from losses import combined_loss
from models import CrossModalTransformer, LateFusionMLP


def serializable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def build_model(name: str):
    if name == "mlp":
        return LateFusionMLP()
    if name == "transformer":
        return CrossModalTransformer()
    raise ValueError(f"Unknown model: {name}")


def run_epoch(model, loader, optimizer, device, model_name: str, alpha: float, scaler=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "mse": 0.0, "info_nce": 0.0, "cosine": 0.0}
    count = 0
    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        target = F.normalize(batch["target"].to(device, non_blocking=True), dim=-1)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            pred = model(image, audio)
            if model_name == "transformer":
                loss, parts = combined_loss(pred, target, alpha)
            else:
                mse = F.mse_loss(pred, target)
                loss, parts = mse, {"mse": float(mse.detach().cpu()), "info_nce": 0.0}
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        bs = image.size(0)
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["mse"] += parts["mse"] * bs
        totals["info_nce"] += parts["info_nce"] * bs
        totals["cosine"] += float(F.cosine_similarity(pred, target, dim=-1).mean().detach().cpu()) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--clip", type=Path, default=Path("features/cache/clip_features.npz"))
    parser.add_argument("--audio", type=Path, default=Path("features/cache/audio_features.npz"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8, help="Early stop after N non-improving epochs.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Tiny quick check: 3 epochs, larger batch, early stop.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="InfoNCE weight for transformer.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.fast_dev_run:
        args.epochs = min(args.epochs, 3)
        args.batch_size = max(args.batch_size, 128)
        args.patience = min(args.patience, 2)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    run_dir = args.out_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = FusionFeatureDataset(args.split_dir / "train.csv", args.clip, args.audio)
    val_ds = FusionFeatureDataset(args.split_dir / "val.csv", args.clip, args.audio)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(args.model).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.model, args.alpha, scaler if device.type == "cuda" else None)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, args.model, args.alpha, scaler if device.type == "cuda" else None)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=None))
        if val_metrics["cosine"] > best:
            best = val_metrics["cosine"]
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({"model": state_dict, "args": serializable_args(args)}, run_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best val cosine: {best:.4f}")
                break

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best val cosine: {best:.4f}; checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
