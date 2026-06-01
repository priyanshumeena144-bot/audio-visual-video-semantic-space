import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import LateFusionBaseline
from training.trainer import Trainer
from utils.config import load_config
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = LateFusionBaseline(**model_cfg)
    Trainer(model, cfg, device).fit()


if __name__ == "__main__":
    main()
