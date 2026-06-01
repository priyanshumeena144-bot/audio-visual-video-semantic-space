from pathlib import Path

import librosa
import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import ASTFeatureExtractor, ASTModel


class CLIPImageEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_image_paths(self, paths):
        images = [self.preprocess(Image.open(path).convert("RGB")) for path in paths]
        batch = torch.stack(images).to(self.device)
        emb = self.model.encode_image(batch)
        return torch.nn.functional.normalize(emb, dim=-1).cpu()


class ASTAudioEncoder:
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", sample_rate=16000, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.sample_rate = sample_rate
        self.extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(self.device).eval()

    def _load_audio(self, path):
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        if wav.size == 0:
            wav = np.zeros(self.sample_rate, dtype=np.float32)
        return wav

    @torch.no_grad()
    def encode_audio_paths(self, paths):
        waves = [self._load_audio(path) for path in paths]
        inputs = self.extractor(waves, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs).last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(out, dim=-1).cpu()


class TeacherVideoEncoder:
    def __init__(self, clip_encoder):
        self.clip_encoder = clip_encoder

    @torch.no_grad()
    def encode_frame_dirs(self, frame_dirs):
        outputs = []
        for frame_dir in frame_dirs:
            paths = sorted(Path(frame_dir).glob("sparse_*.jpg"))
            if not paths:
                paths = sorted(Path(frame_dir).glob("*.jpg"))
            if not paths:
                raise FileNotFoundError(f"No frames found in {frame_dir}")
            emb = self.clip_encoder.encode_image_paths(paths)
            outputs.append(torch.nn.functional.normalize(emb.mean(dim=0), dim=-1))
        return torch.stack(outputs)

