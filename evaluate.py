"""Evaluate approximation quality, retrieval, and efficiency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FusionFeatureDataset, load_retrieval_bank
from metrics import estimate_flops, latency_ms, parameter_count, regression_metrics, retrieval_metrics
from train import build_model


@torch.no_grad()
def predict(model, dataset, device: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    preds, targets, ids = [], [], []
    model.to(device).eval()
    for batch in loader:
        pred = model(batch["image"].to(device), batch["audio"].to(device))
        preds.append(pred.cpu().numpy())
        targets.append(batch["target"].numpy())
        ids.extend(batch["video_id"])
    return np.concatenate(preds), np.concatenate(targets), [str(v) for v in ids]


def model_report(args) -> dict[str, float | int | None]:
    model = build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return {
        "parameters": parameter_count(model),
        "latency_ms_per_sample": latency_ms(model, device=args.device, iters=args.latency_iters),
        "flops_estimate": None if args.skip_flops else estimate_flops(model, device=args.device),
    }


def evaluate_fusion(args) -> dict:
    dataset = FusionFeatureDataset(args.split_csv, args.clip, args.audio)
    bank = load_retrieval_bank(args.split_csv, args.clip)
    model = build_model(args.model)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    pred, target, video_ids = predict(model, dataset, args.device)

    text = bank["text"]
    text_ids = bank["text_video_ids"].tolist()
    result = {
        "latent": regression_metrics(pred, target),
        "video_to_text": retrieval_metrics(pred, text, video_ids, text_ids),
        "text_to_video": retrieval_metrics(text, pred, text_ids, video_ids),
        "efficiency": model_report(args),
        "full_8frame_clip_baseline": {
            "note": "Use feature extraction timing from extract_clip_features.py on your hardware. "
            "This baseline encodes 8 CLIP frames per video versus one image+audio forward pass here."
        },
    }
    return result


def evaluate_imagebind(args) -> dict:
    split_bank = load_retrieval_bank(args.split_csv, args.clip)
    ib = np.load(args.imagebind, allow_pickle=True)
    split_ids = set(split_bank["video_ids"])
    keep_video = np.array([str(v) in split_ids for v in ib["video_ids"]])
    ids = ib["video_ids"][keep_video].astype(str).tolist()
    image = ib["image"][keep_video].astype("float32")
    audio = ib["audio"][keep_video].astype("float32")
    fused = image + audio
    fused = fused / np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8, None)

    result = {
        "image_audio_alignment_cosine": float(np.mean(np.sum(image * audio, axis=1))),
        "num_videos": len(ids),
    }
    if "text" in ib and "text_video_ids" in ib:
        keep_text = np.array([str(v) in split_ids for v in ib["text_video_ids"]])
        text = ib["text"][keep_text].astype("float32")
        text_ids = ib["text_video_ids"][keep_text].astype(str).tolist()
        result["video_to_text"] = retrieval_metrics(fused, text, ids, text_ids)
        result["text_to_video"] = retrieval_metrics(text, fused, text_ids, ids)
    else:
        result["note"] = "ImageBind retrieval requires text features in imagebind_features.npz."
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mlp/best.pt"))
    parser.add_argument("--clip", type=Path, default=Path("features/cache/clip_features.npz"))
    parser.add_argument("--audio", type=Path, default=Path("features/cache/audio_features.npz"))
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/eval.json"))
    parser.add_argument("--imagebind", type=Path, default=None)
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument("--skip-flops", action="store_true", help="Skip ptflops for faster evaluation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = {args.model: evaluate_fusion(args)}
    if args.imagebind:
        results["imagebind_zero_shot"] = evaluate_imagebind(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
