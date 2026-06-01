import argparse
import sys
import time
from pathlib import Path

import torch

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
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    model = build_model(cfg).to(device).eval()
    image = torch.randn(1, cfg["model"].get("image_dim", 512), device=device)
    audio = torch.randn(1, cfg["model"].get("audio_dim", 768), device=device)

    for _ in range(20):
        _ = model(image, audio)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            _ = model(image, audio)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print({"latency_ms": elapsed * 1000.0 / args.iters})


if __name__ == "__main__":
    main()
