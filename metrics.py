"""Regression, retrieval, latency, and model-size metrics."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_n = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8, None)
    target_n = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-8, None)
    return {
        "cosine": float(np.mean(np.sum(pred_n * target_n, axis=1))),
        "mse": float(np.mean((pred - target) ** 2)),
    }


def retrieval_metrics(query: np.ndarray, candidates: np.ndarray, query_ids: list[str], candidate_ids: list[str]) -> dict[str, float]:
    sims = cosine_similarity(query, candidates)
    ranks = []
    candidate_ids_arr = np.asarray(candidate_ids)
    for i, qid in enumerate(query_ids):
        order = np.argsort(-sims[i])
        positives = np.where(candidate_ids_arr[order] == qid)[0]
        ranks.append(int(positives[0]) + 1 if len(positives) else len(candidate_ids))
    ranks_arr = np.asarray(ranks)
    return {
        "R@1": float(np.mean(ranks_arr <= 1) * 100.0),
        "R@5": float(np.mean(ranks_arr <= 5) * 100.0),
        "R@10": float(np.mean(ranks_arr <= 10) * 100.0),
        "MedR": float(np.median(ranks_arr)),
    }


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def latency_ms(model: torch.nn.Module, image_dim: int = 512, audio_dim: int = 128, device: str = "cpu", iters: int = 200) -> float:
    model = model.to(device).eval()
    image = torch.randn(1, image_dim, device=device)
    audio = torch.randn(1, audio_dim, device=device)
    for _ in range(20):
        model(image, audio)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(image, audio)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def estimate_flops(model: torch.nn.Module, device: str = "cpu") -> int | None:
    try:
        from ptflops import get_model_complexity_info

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x[:, :512], x[:, 512:])

        macs, _ = get_model_complexity_info(
            Wrapped(model).to(device),
            (640,),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return int(2 * macs)
    except Exception:
        return None

