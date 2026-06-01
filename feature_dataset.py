from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Dataset backed by precomputed CLIP, AST and teacher embeddings."""

    def __init__(self, manifest_path, feature_dir):
        self.manifest_path = Path(manifest_path)
        self.feature_dir = Path(feature_dir)
        self.df = pd.read_csv(self.manifest_path)

        split = self.manifest_path.stem
        self.image_features = torch.load(self.feature_dir / f"{split}_image_clip.pt", map_location="cpu")
        self.audio_features = torch.load(self.feature_dir / f"{split}_audio_ast.pt", map_location="cpu")
        self.teacher_features = torch.load(self.feature_dir / f"{split}_teacher_video_clip8.pt", map_location="cpu")

        self.video_ids = self.df["video_id"].astype(str).tolist()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        return {
            "video_id": video_id,
            "image_emb": self.image_features[video_id].float(),
            "audio_emb": self.audio_features[video_id].float(),
            "teacher_emb": self.teacher_features[video_id].float(),
        }

