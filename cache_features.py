import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoders import ASTAudioEncoder, CLIPImageEncoder, TeacherVideoEncoder
from utils.config import load_config
from utils.seed import set_seed


def chunks(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def cache_split(split, cfg, clip_encoder, audio_encoder, teacher_encoder):
    manifest = Path(cfg["data"]["manifest_dir"]) / f"{split}.csv"
    df = pd.read_csv(manifest)
    frame_dir = Path(cfg["data"]["frame_dir"])
    audio_dir = Path(cfg["data"]["audio_dir"])
    feature_dir = Path(cfg["data"]["feature_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)

    image_features = {}
    audio_features = {}
    teacher_features = {}
    rows = df.to_dict("records")
    batch_size = cfg["runtime"]["batch_size"]

    for batch in tqdm(list(chunks(rows, batch_size)), desc=f"cache {split}"):
        ids = [str(r["video_id"]) for r in batch]
        keyframes = [frame_dir / vid / "keyframe.jpg" for vid in ids]
        audios = [audio_dir / f"{vid}.wav" for vid in ids]
        frame_dirs = [frame_dir / vid for vid in ids]

        img_emb = clip_encoder.encode_image_paths(keyframes)
        aud_emb = audio_encoder.encode_audio_paths(audios)
        teach_emb = teacher_encoder.encode_frame_dirs(frame_dirs)

        for vid, iemb, aemb, temb in zip(ids, img_emb, aud_emb, teach_emb):
            image_features[vid] = iemb
            audio_features[vid] = aemb
            teacher_features[vid] = temb

    torch.save(image_features, feature_dir / f"{split}_image_clip.pt")
    torch.save(audio_features, feature_dir / f"{split}_audio_ast.pt")
    torch.save(teacher_features, feature_dir / f"{split}_teacher_video_clip8.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cache_features.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    clip_encoder = CLIPImageEncoder(
        cfg["encoders"]["clip_model"],
        cfg["encoders"]["clip_pretrained"],
        cfg["device"],
    )
    audio_encoder = ASTAudioEncoder(
        cfg["encoders"]["ast_model"],
        cfg["encoders"]["sample_rate"],
        cfg["device"],
    )
    teacher_encoder = TeacherVideoEncoder(clip_encoder)

    for split in cfg["data"]["splits"]:
        cache_split(split, cfg, clip_encoder, audio_encoder, teacher_encoder)


if __name__ == "__main__":
    main()
