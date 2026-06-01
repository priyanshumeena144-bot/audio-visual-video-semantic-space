import torch
from torch import nn
import torch.nn.functional as F


class LateFusionBaseline(nn.Module):
    def __init__(self, image_dim=512, audio_dim=768, output_dim=512, dropout=0.2):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + output_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, image_emb, audio_emb):
        image_emb = F.normalize(image_emb, dim=-1)
        audio_emb = F.normalize(self.audio_proj(audio_emb), dim=-1)
        fused = torch.cat([image_emb, audio_emb], dim=-1)
        out = self.fusion(fused)
        return F.normalize(out, dim=-1)


class ImageOnlyBaseline(nn.Module):
    def forward(self, image_emb, audio_emb=None):
        return F.normalize(image_emb, dim=-1)


class AudioOnlyBaseline(nn.Module):
    def __init__(self, audio_dim=768, output_dim=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(audio_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, image_emb, audio_emb):
        return F.normalize(self.proj(audio_emb), dim=-1)

