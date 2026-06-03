# Demo Video

This project includes a slide-based MP4 generator for submission demos. It does
not need a website or screen recorder.

Generate the demo video from the project folder:

```powershell
cd "C:\Users\ASUS\OneDrive\Desktop\Aims internship\project"
python demo\make_demo_video.py --eval-json runs_smoke\mlp_eval.json --out demo\project_demo.mp4
```

The output file is:

```text
demo/project_demo.mp4
```

Suggested narration while showing the video:

```text
This project studies whether a full-video semantic embedding can be approximated
using only one static frame and the audio track. The target video embedding is a
mean-pooled CLIP embedding over sampled frames. I implemented two models: a
late-fusion MLP baseline and a cross-modal Transformer trained with MSE plus
InfoNCE. The project includes preprocessing, CLIP/audio/ImageBind feature
extraction, training, retrieval evaluation, visualization, README, and a
technical report. Since MSR-VTT is external, the included smoke test verifies
the pipeline end-to-end before running the full dataset.
```
