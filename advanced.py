import torch
from torch import nn
import torch.nn.functional as F


class TransformerFusionDistiller(nn.Module):
    def __init__(
        self,
        image_dim=512,
        audio_dim=768,
        hidden_dim=512,
        output_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim) if image_dim != hidden_dim else nn.Identity()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.modality_embed = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, image_emb, audio_emb):
        image_tok = F.normalize(self.image_proj(image_emb), dim=-1).unsqueeze(1)
        audio_tok = F.normalize(self.audio_proj(audio_emb), dim=-1).unsqueeze(1)
        cls_tok = self.cls.expand(image_emb.size(0), -1, -1)

        tokens = torch.cat([cls_tok, image_tok, audio_tok], dim=1)
        tokens = tokens + self.modality_embed
        out = self.transformer(tokens)
        z = self.proj(out[:, 0])
        return F.normalize(z, dim=-1)

