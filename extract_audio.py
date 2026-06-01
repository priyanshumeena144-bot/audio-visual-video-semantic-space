import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def extract_audio(video_path, out_path, sample_rate=16000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(out_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="data/processed/msrvtt/audio")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    failures = []
    for row in tqdm(df.to_dict("records")):
        video_id = str(row["video_id"])
        ok = extract_audio(row["video_path"], Path(args.out_dir) / f"{video_id}.wav", args.sample_rate)
        if not ok:
            failures.append(video_id)
    print({"failures": len(failures)})


if __name__ == "__main__":
    main()

