import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def save_frame(video_path, out_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, min(frame_index, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), frame)


def extract_middle(video_path, out_dir, video_id):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return save_frame(video_path, Path(out_dir) / video_id / "keyframe.jpg", total // 2)


def extract_sparse(video_path, out_dir, video_id, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return False
    ok_all = True
    for i in range(num_frames):
        idx = int((i + 1) * total / (num_frames + 1))
        ok = save_frame(video_path, Path(out_dir) / video_id / f"sparse_{i:02d}.jpg", idx)
        ok_all = ok_all and ok
    return ok_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="data/processed/msrvtt/frames")
    parser.add_argument("--mode", default="both", choices=["middle", "sparse", "both"])
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    failures = []
    for row in tqdm(df.to_dict("records")):
        video_path = Path(row["video_path"])
        video_id = str(row["video_id"])
        if args.mode in {"middle", "both"}:
            if not extract_middle(video_path, args.out_dir, video_id):
                failures.append(video_id)
        if args.mode in {"sparse", "both"}:
            if not extract_sparse(video_path, args.out_dir, video_id, args.num_frames):
                failures.append(video_id)
    print({"failures": len(set(failures))})


if __name__ == "__main__":
    main()

