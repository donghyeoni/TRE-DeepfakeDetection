# TRE Diffusion Image Detection

Detecting AI-generated images with a **Temporal Reconstruction Error (TRE)**
feature and a **temporal / spatial attention** classifier (DNSAMNet).

## Overview

This project targets **generative-model image detection** — telling apart real
photographs from images produced by generative models (diffusion models, GANs,
etc.). Despite the "deepfake" name used throughout the original code, the
datasets are **generated-image benchmarks (GenImage / ForenSynths), not
face-swap deepfakes**.

The core idea:

1. Run **DDPM/DDIM inversion** with a pretrained Stable Diffusion model: encode
   an image into a latent, invert it to noise, then reconstruct it step by step.
2. Record how the latent changes across denoising timesteps. The sequence of
   **step-wise differences is the Temporal Reconstruction Error (TRE)** — a
   `(T, D, H, W)` tensor (feature convention `(B, T=20, D=8, H, W)`).
3. Classify the TRE sequence as real/fake with **DNSAMNet**, which combines
   temporal attention (over the `T=20` axis) and spatial attention (a focusing
   map over `H×W`), fuses them multiplicatively, and feeds an MLP classifier.

Real and generated images leave different fingerprints in how well a diffusion
model can reconstruct them across timesteps; the attention classifier learns to
read that signal.

## Method

- **Diffusion inversion** (`src/data/inversion.py`): edit-friendly DDPM/DDIM
  inversion — `inversion_forward_process` records the noise sequence, and
  `inversion_reverse_process` reconstructs from it.
- **TRE features** (`src/data/tre_features.py`): `over_denosing()` reconstructs
  the image step by step and returns the `T` step-wise latent differences.
- **DNSAMNet** (`src/models/`):
  - `attention.py` — hand-written multi-head self-attention (`MHSA`, `MHSABlock`).
  - `temporal_attention.py` — `TemporalAggregation` (attention over `T`).
  - `spatial_attention.py` — `SpatialFocusing` (pool → conv stack → attention →
    spatial map).
  - `dnsamnet.py` — `Classifier` + `TSC` (Temporal × Spatial → Classifier).
  - `resnet_baseline.py` — an earlier `nn.MultiheadAttention` + ResNet-18
    baseline, kept for reference.

## Dataset

Data is **not included** in this repository, and neither are the precomputed
`.pt` TRE features nor the trained weights. Download the source datasets from
their official releases:

- **GenImage** — https://github.com/GenImage-Dataset/GenImage
  Generators used here: `sdv1.4`, `sdv1.5`, `adm`, `biggan`, `glide`,
  `midjourney`, `vqdm`, `wukong`.
- **ForenSynths** (CNNDetection) — https://github.com/PeterWang512/CNNDetection

Expected on-disk layout (override roots via `src/config.py` or environment
variables `GENIMAGE_ROOT` / `TRE_FEATURE_ROOT`):

```
<GENIMAGE_ROOT>/sdv1.4/train/<class>/*.png     # training images (real/fake)
<GENIMAGE_ROOT>/<generator>/val/<class>/*.png  # evaluation images

<TRE_FEATURE_ROOT>/train/<class>/*.pt          # precomputed TRE features
<TRE_FEATURE_ROOT>/test/<generator>/<class>/*.pt
```

## Repository structure

```
tre-diffusion-image-detection/
├── src/
│   ├── config.py                 # dataset roots, generators, SD model id, T/steps, batch, lr
│   ├── data/
│   │   ├── inversion.py          # DDPM/DDIM inversion core (consolidated from 3 notebooks)
│   │   ├── tre_features.py       # over_denosing() -> step-wise latent diffs (TRE)
│   │   ├── build_dataset.py      # CLI: iterate images, extract TRE, save .pt
│   │   └── dataset.py            # transforms, LatentDiffDataset, ImageFolder/DatasetFolder builders
│   ├── models/
│   │   ├── attention.py          # MHSA, MHSABlock
│   │   ├── temporal_attention.py # TemporalAggregation
│   │   ├── spatial_attention.py  # SpatialFocusing
│   │   ├── dnsamnet.py           # Classifier + TSC (full model)
│   │   └── resnet_baseline.py    # ResNet18_4ch, AttentionClassifier (baseline)
│   ├── train.py                  # fit() loop, optimizer, checkpoint save
│   └── eval.py                   # acc(), ap() per-generator + mean, plots
├── notebooks/                    # original notebooks kept as demos
├── docs/                         # project report (PDF)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended — diffusion inversion is the
bottleneck of the pipeline.

## Usage

Run modules from the repository root so package imports resolve.

1. **Build TRE features** from a folder of images:

   ```bash
   python -m src.data.build_dataset \
       --images data/genimage/sdv1.4/val \
       --out data/tre_features/test/sdv4 \
       --limit 1000
   ```

2. **Train** the detector on the precomputed features:

   ```bash
   python -m src.train --epochs 100 --lr 1e-5 --checkpoint weights/DNSAM_new.pt
   ```

3. **Evaluate** accuracy and Average Precision per generator:

   ```bash
   python -m src.eval --checkpoint weights/DNSAM_new.pt
   ```

## Notes

### Bugs fixed during modularization

- **Accuracy threshold**: the model outputs a sigmoid probability, but the
  notebooks thresholded predictions at `> 0` (always true). Training and
  evaluation now threshold at `> 0.5`.
- **`SpatialFocusing` undefined name**: the notebook referenced `num_head`
  inside `SpatialFocusing.__init__` where the argument was `num_heads`, raising a
  `NameError`. Fixed to use `num_heads`.
- **`TREClassifier` name error**: a test cell instantiated the non-existent
  `TREClassifier`; the class is `TSC`.
- **Checkpoint filename typo**: `Attn_cocnat.pt` → `Attn_concat.pt` (the save
  and load paths disagreed).
- **Hard-coded paths**: machine-specific paths (`/home/rmlab`, `/home/mplab`,
  `/home/dh/venv`, Colab Drive) were replaced with values in `src/config.py`.

### Canonical module choice

There were three divergent copies of `TemporalAggregation` / `SpatialFocusing`
in `DNSAMNet.ipynb` (cells ~17/20/38) plus an `nn.MultiheadAttention`-based
prototype in `LoadDataset.ipynb`. The **cell-38 variant** (residual FFN blocks +
convolutional stack) is used as canonical in `src/models/`. The
`nn.MultiheadAttention` prototype is preserved in `resnet_baseline.py`.

### Feature-shape note

The DNSAMNet classifier operates on `(B, T=20, D=8, H=16, W=16)` tensors. The
raw Stable Diffusion latent produced by `over_denosing` has `D=4` channels at
`32×32`. Set `FEATURE_DIM` / `FEATURE_H` / `FEATURE_W` in `src/config.py` to
match whatever your precomputed `.pt` features actually contain.

### Reproducibility

No metrics or results are reported here — see the project report in `docs/` for
the experiments run by the authors. Data, precomputed features, and trained
weights are not distributed with this repository.
