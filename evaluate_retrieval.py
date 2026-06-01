import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import CachedFeatureDataset
from evaluation.metrics import collect_embeddings, retrieval_metrics
from models import LateFusionBaseline, TransformerFusionDistiller
from utils.checkpoint import load_model_checkpoint
from utils.config import load_config


def build_model(cfg):
    if cfg["model"]["name"] == "baseline":
        return LateFusionBaseline(**{k: v for k, v in cfg["model"].items() if k != "name"})
    return TransformerFusionDistiller(**{k: v for k, v in cfg["model"].items() if k != "name"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    manifest = cfg["data"][f"{args.split}_manifest"]
    dataset = CachedFeatureDataset(manifest, cfg["data"]["feature_dir"])
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(cfg).to(device)
    load_model_checkpoint(args.checkpoint, model, map_location=device)
    pred, target, _ = collect_embeddings(model, loader, device)
    metrics = retrieval_metrics(pred, target)

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/tables") / f"{args.split}_retrieval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
