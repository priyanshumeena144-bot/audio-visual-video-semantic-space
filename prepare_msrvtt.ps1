python preprocessing/build_manifest.py --raw-dir data/raw/msrvtt --out-dir data/processed/msrvtt/manifests
python preprocessing/extract_frames.py --manifest data/processed/msrvtt/manifests/train.csv --out-dir data/processed/msrvtt/frames --mode both
python preprocessing/extract_frames.py --manifest data/processed/msrvtt/manifests/val.csv --out-dir data/processed/msrvtt/frames --mode both
python preprocessing/extract_frames.py --manifest data/processed/msrvtt/manifests/test.csv --out-dir data/processed/msrvtt/frames --mode both
python preprocessing/extract_audio.py --manifest data/processed/msrvtt/manifests/train.csv --out-dir data/processed/msrvtt/audio
python preprocessing/extract_audio.py --manifest data/processed/msrvtt/manifests/val.csv --out-dir data/processed/msrvtt/audio
python preprocessing/extract_audio.py --manifest data/processed/msrvtt/manifests/test.csv --out-dir data/processed/msrvtt/audio

