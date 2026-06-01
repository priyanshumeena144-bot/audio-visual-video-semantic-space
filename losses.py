"""Training losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(pred, target)
    nce = info_nce(pred, target)
    loss = mse + alpha * nce
    return loss, {"mse": float(mse.detach().cpu()), "info_nce": float(nce.detach().cpu())}

