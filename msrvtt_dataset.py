from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class MSRVTTManifestDataset(Dataset):
    """Lightweight path dataset used for preprocessing and inspection."""

    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        row["video_path"] = str(Path(row["video_path"]))
        return row

