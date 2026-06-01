"""Generate training curves and t-SNE embedding visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from dataset import FusionFeatureDataset
from evaluate import predict
from train import build_model


def plot_curves(history_path: Path, out_path: Path) -> None:
    with history_path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    epochs = [r["epoch"] for r in hist]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in hist], label="train loss")
    plt.plot(epochs, [r["val_loss"] for r in hist], label="val loss")
    plt.plot(epochs, [r["val_cosine"] for r in hist], label="val cosine")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_tsne(model_name: str, checkpoint: Path, split_csv: Path, clip: Path, audio: Path, out_path: Path, device: str) -> None:
    ds = FusionFeatureDataset(split_csv, clip, audio)
    model = build_model(model_name)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    pred, target, _ = predict(model, ds, device)
    n = min(300, len(pred))
    points = np.concatenate([pred[:n], target[:n]], axis=0)
    labels = np.array(["predicted"] * n + ["ground truth"] * n)
    xy = TSNE(n_components=2, perplexity=min(30, max(5, n // 5)), init="pca", random_state=13).fit_transform(points)

    plt.figure(figsize=(7, 6))
    for label, marker in [("predicted", "o"), ("ground truth", "x")]:
        mask = labels == label
        plt.scatter(xy[mask, 0], xy[mask, 1], s=18, alpha=0.75, label=label, marker=marker)
    plt.legend()
    plt.title(f"t-SNE: {model_name} vs full-video CLIP proxy")
    plt.xticks([])
    plt.yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mlp/best.pt"))
    parser.add_argument("--history", type=Path, default=Path("runs/mlp/history.json"))
    parser.add_argument("--clip", type=Path, default=Path("features/cache/clip_features.npz"))
    parser.add_argument("--audio", type=Path, default=Path("features/cache/audio_features.npz"))
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/figures"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    plot_curves(args.history, args.out_dir / f"{args.model}_curves.png")
    plot_tsne(args.model, args.checkpoint, args.split_csv, args.clip, args.audio, args.out_dir / f"{args.model}_tsne.png", args.device)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
