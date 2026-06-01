from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CachedFeatureDataset
from evaluation.metrics import collect_embeddings, retrieval_metrics
from models.losses import combined_loss
from utils.checkpoint import save_checkpoint
from utils.logger import format_metrics


class Trainer:
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg["training"]["epochs"])
        self.scaler = GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")

    def make_loader(self, split, shuffle):
        dataset = CachedFeatureDataset(self.cfg["data"][f"{split}_manifest"], self.cfg["data"]["feature_dir"])
        return DataLoader(
            dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["data"]["num_workers"],
            pin_memory=self.device.type == "cuda",
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            image = batch["image_emb"].to(self.device, non_blocking=True)
            audio = batch["audio_emb"].to(self.device, non_blocking=True)
            teacher = batch["teacher_emb"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                pred = self.model(image, audio)
                loss, parts = combined_loss(pred, teacher, self.cfg["loss"])

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item()
            if step % self.cfg["training"]["log_every"] == 0:
                pbar.set_postfix(loss=running / step)
        return {"train_loss": running / max(1, len(loader))}

    def validate(self, loader):
        pred, target, _ = collect_embeddings(self.model, loader, self.device)
        return retrieval_metrics(pred, target)

    def fit(self):
        train_loader = self.make_loader("train", shuffle=True)
        val_loader = self.make_loader("val", shuffle=False)
        best = -1.0
        save_dir = Path(self.cfg["training"]["save_dir"])
        run_name = self.cfg.get("run_name", self.cfg["model"]["name"])

        for epoch in range(1, self.cfg["training"]["epochs"] + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            merged = {**train_metrics, **val_metrics}
            print(format_metrics({"epoch": epoch, **merged}))

            score = val_metrics.get(self.cfg["training"]["best_metric"], val_metrics["R@5"])
            save_checkpoint(save_dir / f"{run_name}_last.pt", self.model, self.optimizer, epoch, merged, self.cfg)
            if score > best:
                best = score
                save_checkpoint(save_dir / f"{self.cfg['model']['name']}_best.pt", self.model, self.optimizer, epoch, merged, self.cfg)

