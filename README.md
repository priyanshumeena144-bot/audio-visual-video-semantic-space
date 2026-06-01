# Audio-Visual Approximation of Video Semantic Space

This project tests whether the semantic embedding of a full video can be approximated from only one static frame plus the audio track. The target embedding is a practical proxy for dense video semantics: the mean-pooled CLIP ViT-B/32 embedding over 8 uniformly sampled frames.

## Setup

```bash
cd project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The code runs on CPU for small subsets, but a single CUDA GPU is recommended for feature extraction and training.

## Fast Smoke Run

For a quick end-to-end check, use a small subset, fewer CLIP frames, cached features, short audio clips, and early-stopped training:

```bash
python data/download_msr_vtt.py --captions path\to\captions.json --video-dir path\to\videos --out data/manifests/msrvtt_fast.csv --subset-size 100
python data/preprocess.py --manifest data/manifests/msrvtt_fast.csv --out-dir data/splits_fast --limit 100
python features/extract_clip_features.py --manifest data/manifests/msrvtt_fast.csv --out-dir features/cache_fast --frames 4 --batch-size 128
python features/extract_audio_features.py --manifest data/manifests/msrvtt_fast.csv --out-dir features/cache_fast --max-seconds 5
python train.py --model mlp --clip features/cache_fast/clip_features.npz --audio features/cache_fast/audio_features.npz --split-dir data/splits_fast --fast-dev-run
python evaluate.py --model mlp --checkpoint runs/mlp/best.pt --clip features/cache_fast/clip_features.npz --audio features/cache_fast/audio_features.npz --split-csv data/splits_fast/test.csv --skip-flops --latency-iters 20
```

Use `--overwrite` on feature scripts only when you want to recompute cached features.

## Data

Download MSR-VTT videos and captions from an authorized benchmark mirror, then build a local 1k-video manifest:

```bash
python data/download_msr_vtt.py --captions path\to\captions.json --video-dir path\to\videos --out data/manifests/msrvtt_1k.csv --subset-size 1000
python data/preprocess.py --manifest data/manifests/msrvtt_1k.csv --out-dir data/splits
```

The split is deterministic: 70% train, 15% validation, 15% test by video id.

## Feature Extraction

```bash
python features/extract_clip_features.py --manifest data/manifests/msrvtt_1k.csv --out-dir features/cache
python features/extract_audio_features.py --manifest data/manifests/msrvtt_1k.csv --out-dir features/cache --use-clap
```

If CLAP cannot be loaded, the audio script falls back to deterministic log-mel statistics reduced to 128 dimensions, preserving the VGGish-style interface required by the baseline.

Speed knobs:

- `--frames 4` makes a faster approximate target for debugging; use `--frames 8` for final runs.
- `--limit N` extracts only the first `N` videos from a manifest.
- `--max-seconds 5` shortens audio decoding.
- Existing feature files are reused automatically unless `--overwrite` is passed.

Optional ImageBind benchmark:

```bash
python features/extract_imagebind_features.py --manifest data/manifests/msrvtt_1k.csv --out-dir features/cache
```

Install ImageBind from Meta's official repository before running that optional script.

## Training

Baseline late-fusion MLP:

```bash
python train.py --model mlp --epochs 30 --batch-size 128 --patience 6
```

Cross-modal Transformer:

```bash
python train.py --model transformer --epochs 40 --batch-size 128 --alpha 0.5 --patience 6
```

Checkpoints and `history.json` files are written under `runs/<model>/`.
On CUDA, training uses automatic mixed precision. Add `--fast-dev-run` for a 3-epoch sanity check or `--compile` to try `torch.compile`.

## Evaluation

```bash
python evaluate.py --model mlp --checkpoint runs/mlp/best.pt --out runs/mlp_eval.json
python evaluate.py --model transformer --checkpoint runs/transformer/best.pt --out runs/transformer_eval.json
```

With optional ImageBind features, including zero-shot image+audio to text retrieval:

```bash
python evaluate.py --model transformer --checkpoint runs/transformer/best.pt --imagebind features/cache/imagebind_features.npz --out runs/transformer_eval_imagebind.json
```

Metrics include latent cosine similarity, MSE, video-to-text and text-to-video Recall@1/5/10, median rank, parameter count, latency, and FLOPs estimate.

## Visualization

```bash
python visualize.py --model mlp --checkpoint runs/mlp/best.pt --history runs/mlp/history.json
python visualize.py --model transformer --checkpoint runs/transformer/best.pt --history runs/transformer/history.json
```

Figures are saved to `runs/figures/`.

## Results Table

Fill this table after running the pipeline on your local MSR-VTT subset:

| Model | Cosine ↑ | MSE ↓ | V2T R@1 ↑ | V2T R@5 ↑ | T2V R@1 ↑ | T2V R@5 ↑ | MedR ↓ | Latency ms ↓ | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full 8-frame CLIP proxy | 1.000 | 0.000 | TBD | TBD | TBD | TBD | TBD | TBD | 0 trainable |
| MLP late fusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Cross-modal Transformer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ImageBind zero-shot | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0 trainable |

## Assumptions and Limitations

- The full-video embedding is a proxy, not human semantic ground truth.
- MSR-VTT captions are many-to-one; retrieval treats all captions sharing the video id as positives.
- Audio is optional in many web videos. Missing audio is represented as zeros.
- ImageBind is optional because its installation is platform-specific and not reliably pip-installable across all environments.
- For exact full-CLIP speed comparison, time `extract_clip_features.py` on your hardware; the learned models avoid the 8-frame CLIP image forward passes at inference after features are available.
