# Audio-Visual Approximation of Video Semantic Space

## Abstract

We study whether a video-level semantic embedding can be approximated from a single static frame and the corresponding audio track. Using MSR-VTT, we define the target semantic space as the mean of CLIP ViT-B/32 embeddings over eight uniformly sampled frames. Two learned approximators are evaluated: a late-fusion MLP and a compact cross-modal Transformer. The task is measured through latent-space proximity, video-text retrieval, and computational efficiency.

## 1. Introduction

Dense video encoders are expensive because they repeatedly process frames, often with high-capacity vision backbones. Many retrieval settings, however, only need a semantic embedding rather than frame-level temporal reasoning. A middle frame captures objects and scene context, while audio can expose events, speech, ambience, and actions that may be visually ambiguous. This project asks whether those two low-cost signals can recover enough of the full-video semantic embedding for competitive retrieval.

## 2. Related Work

CLIP aligns images and text through contrastive pretraining and is widely used as a zero-shot semantic embedding model. AudioCLIP extends CLIP-style alignment to audio-image-text settings. Contrastive Multiview Coding learns shared representations from different views of the same underlying sample. ImageBind generalizes this idea across image, text, audio, depth, thermal, and IMU modalities, making it a strong zero-shot reference for multimodal fusion.

## 3. Methodology

### Target Space

For each video, eight frames are sampled uniformly. Each frame is encoded with CLIP ViT-B/32, L2-normalized, mean-pooled, and normalized again. This vector is the ground-truth proxy video embedding.

### Inputs

The visual input is the middle-frame CLIP ViT-B/32 embedding. The audio input is a 128-dimensional embedding produced by CLAP if available or a VGGish-style log-mel fallback reduced to the same dimension.

### Model 1: Late-Fusion MLP

The 512-dimensional image vector and 128-dimensional audio vector are concatenated. The resulting 640-dimensional vector is passed through a two-layer MLP with ReLU and dropout:

`[image; audio] -> Linear(640, 512) -> ReLU -> Dropout -> Linear(512, 512)`.

The model is trained with MSE loss against the full-video CLIP proxy.

### Model 2: Cross-Modal Transformer

Image and audio embeddings are projected into a shared 256-dimensional space. They form two modality tokens with learned modality embeddings. A two-layer Transformer encoder performs cross-modal fusion through self-attention over the tokens. Mean pooling and a linear head produce the final 512-dimensional predicted video embedding. Training uses MSE plus symmetric InfoNCE:

`L = MSE(pred, target) + alpha * InfoNCE(pred, target)`.

## 4. Experimental Setup

The dataset is a 1000-video MSR-VTT subset with all captions retained for selected videos. Splits are 70/15/15 by video id. Retrieval uses caption positives by shared video id. Evaluation reports cosine similarity, MSE, Recall@1/5/10, median rank, inference latency, parameter count, and FLOPs estimate. Training curves and t-SNE plots are generated for qualitative analysis.

## 5. Results and Analysis

| Method | Cosine ↑ | MSE ↓ | V2T R@1 ↑ | V2T R@5 ↑ | T2V R@1 ↑ | T2V R@5 ↑ | MedR ↓ | Latency ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full 8-frame CLIP proxy | 1.000 | 0.000 | TBD | TBD | TBD | TBD | TBD | TBD |
| Late-fusion MLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Cross-modal Transformer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ImageBind zero-shot | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Expected behavior is that the MLP establishes a strong low-cost regression baseline because the middle frame is already close to the mean frame embedding for visually stable videos. The Transformer should improve on ambiguous cases by allowing audio and image features to interact before projection. Gains should be most visible in retrieval metrics rather than raw MSE when InfoNCE sharpens neighborhood structure.

## 6. Computational Efficiency

At inference, the learned fusion models operate on one image embedding and one audio embedding, avoiding eight CLIP frame passes. The MLP has the lowest latency and parameter count. The Transformer is slightly heavier but remains small compared with repeatedly invoking CLIP ViT-B/32 over video frames. Exact speedups depend on hardware and feature caching strategy.

## 7. Limitations

The target is a CLIP-derived proxy and may underrepresent temporal events. Audio extraction can fail or add little signal for silent videos. The 1000-video subset is suitable for feasibility but not a final benchmark-scale claim. ImageBind support is optional because installation and pretrained weight access can vary by environment. When installed, the ImageBind script extracts image, audio, and text embeddings so zero-shot retrieval is evaluated in the same video-to-text and text-to-video format.

## 8. Conclusion

Single-frame plus audio fusion is a practical route to approximate video semantic embeddings with lower compute than dense frame processing. The MLP provides a minimum viable baseline, while the cross-modal Transformer adds a stronger inductive bias for multimodal interaction and contrastive retrieval structure. Future work should evaluate larger MSR-VTT splits, use stronger audio encoders, add text-supervised fine-tuning, and compare against modern video-language models.
