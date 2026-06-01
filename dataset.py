"""Feature dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _index_by_id(ids: np.ndarray) -> dict[str, int]:
    return {str(v): i for i, v in enumerate(ids)}


class FusionFeatureDataset(Dataset):
    def __init__(self, split_csv: Path, clip_npz: Path, audio_npz: Path):
        split = pd.read_csv(split_csv)
        clip = np.load(clip_npz, allow_pickle=True)
        audio = np.load(audio_npz, allow_pickle=True)
        clip_index = _index_by_id(clip["video_ids"])
        audio_index = _index_by_id(audio["video_ids"])
        requested_ids = split["video_id"].astype(str).drop_duplicates().tolist()
        self.video_ids = [v for v in requested_ids if v in clip_index and v in audio_index]
        missing = len(requested_ids) - len(self.video_ids)
        if missing:
            print(f"Warning: skipped {missing} split videos missing from feature caches.")
        if not self.video_ids:
            raise ValueError("No split videos were found in both CLIP and audio feature caches.")

        self.image = torch.tensor(np.stack([clip["image"][clip_index[v]] for v in self.video_ids]), dtype=torch.float32)
        self.audio = torch.tensor(np.stack([audio["audio"][audio_index[v]] for v in self.video_ids]), dtype=torch.float32)
        self.target = torch.tensor(np.stack([clip["video"][clip_index[v]] for v in self.video_ids]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "video_id": self.video_ids[idx],
            "image": self.image[idx],
            "audio": self.audio[idx],
            "target": self.target[idx],
        }


def load_retrieval_bank(split_csv: Path, clip_npz: Path) -> dict[str, np.ndarray | list[str]]:
    split = pd.read_csv(split_csv)
    clip = np.load(clip_npz, allow_pickle=True)
    video_ids = split["video_id"].astype(str).drop_duplicates().tolist()
    keep_text = np.isin(clip["text_video_ids"].astype(str), np.array(video_ids))

    vid_index = _index_by_id(clip["video_ids"])
    return {
        "video_ids": video_ids,
        "video": np.stack([clip["video"][vid_index[v]] for v in video_ids]).astype("float32"),
        "text": clip["text"][keep_text].astype("float32"),
        "text_video_ids": clip["text_video_ids"][keep_text].astype(str),
        "captions": clip["captions"][keep_text].astype(str),
    }
