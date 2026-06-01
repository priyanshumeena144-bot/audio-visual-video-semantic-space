import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import LateFusionBaseline, TransformerFusionDistiller
from utils.config import load_config


def build_model(cfg):
    if cfg["model"]["name"] == "baseline":
        return LateFusionBaseline(**{k: v for k, v in cfg["model"].items() if k != "name"})
    return TransformerFusionDistiller(**{k: v for k, v in cfg["model"].items() if k != "name"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print({"total_parameters": total, "trainable_parameters": trainable})


if __name__ == "__main__":
    main()
