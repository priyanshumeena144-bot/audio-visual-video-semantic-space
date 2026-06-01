import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_records(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "annotations" in data:
        return data["annotations"]
    if "sentences" in data:
        return data["sentences"]
    if "videos" in data:
        return data["videos"]
    raise ValueError(f"Unsupported MSR-VTT JSON format: {json_path}")


def normalize_record(record, raw_dir):
    video_id = str(record.get("video_id") or record.get("id") or record.get("clip_id") or record.get("video"))
    if video_id.endswith(".mp4"):
        stem = Path(video_id).stem
    else:
        stem = video_id
    caption = record.get("caption") or record.get("sentence") or record.get("text") or ""
    video_path = Path(raw_dir) / "raw_videos" / f"{stem}.mp4"
    return {"video_id": stem, "video_path": str(video_path), "caption": caption}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/msrvtt")
    parser.add_argument("--out-dir", default="data/processed/msrvtt/manifests")
    parser.add_argument("--limit-train", type=int, default=3000)
    parser.add_argument("--limit-test", type=int, default=1000)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json = raw_dir / "msrvtt_train_7k.json"
    if not train_json.exists():
        train_json = raw_dir / "msrvtt_train_9k.json"
    test_json = raw_dir / "msrvtt_test_1k.json"

    train_records = [normalize_record(r, raw_dir) for r in read_records(train_json)]
    train_df = pd.DataFrame(train_records).drop_duplicates("video_id")
    if args.limit_train:
        train_df = train_df.head(args.limit_train)

    train_df, val_df = train_test_split(train_df, test_size=0.12, random_state=42)

    if test_json.exists():
        test_records = [normalize_record(r, raw_dir) for r in read_records(test_json)]
        test_df = pd.DataFrame(test_records).drop_duplicates("video_id")
        if args.limit_test:
            test_df = test_df.head(args.limit_test)
    else:
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print({"train": len(train_df), "val": len(val_df), "test": len(test_df), "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()

