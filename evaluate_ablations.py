import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import CachedFeatureDataset
from evaluation.metrics import collect_embeddings, retrieval_metrics
from models.baseline import ImageOnlyBaseline, LateFusionBaseline
from models.advanced import TransformerFusionDistiller
from utils.checkpoint import load_model_checkpoint
from utils.config import load_config


def build_trained_model(cfg, checkpoint):
    if cfg["model"]["name"] == "baseline":
        model = LateFusionBaseline(**{k: v for k, v in cfg["model"].items() if k != "name"})
    else:
        model = TransformerFusionDistiller(**{k: v for k, v in cfg["model"].items() if k != "name"})
    load_model_checkpoint(checkpoint, model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    dataset = CachedFeatureDataset(cfg["data"][f"{args.split}_manifest"], cfg["data"]["feature_dir"])
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    results = {}
    image_only = ImageOnlyBaseline().to(device).eval()
    pred, target, _ = collect_embeddings(image_only, loader, device)
    results["image_only_clip"] = retrieval_metrics(pred, target)

    trained = build_trained_model(cfg, args.checkpoint).to(device).eval()
    pred, target, _ = collect_embeddings(trained, loader, device)
    results["trained_fusion"] = retrieval_metrics(pred, target)

    for name, zero_image, zero_audio in [
        ("trained_no_image", True, False),
        ("trained_no_audio", False, True),
    ]:
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                image = batch["image_emb"].to(device)
                audio = batch["audio_emb"].to(device)
                target = batch["teacher_emb"].to(device)
                if zero_image:
                    image = torch.zeros_like(image)
                if zero_audio:
                    audio = torch.zeros_like(audio)
                preds.append(trained(image, audio).cpu())
                targets.append(target.cpu())
        results[name] = retrieval_metrics(torch.cat(preds), torch.cat(targets))

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/tables") / f"{args.split}_ablation_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
