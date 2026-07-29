"""Central configuration for the TRE diffusion image-detection pipeline.

Every path, model id and hyper-parameter used across the code base lives here so
the modules stay free of the hard-coded, machine-specific paths that were
scattered through the original notebooks (``/home/rmlab``, ``/home/mplab``,
``/home/dh/venv`` and Colab Drive paths).

Paths can be overridden with environment variables without editing this file.
"""

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Stable Diffusion / diffusion inversion
# --------------------------------------------------------------------------- #
# The pretrained Stable Diffusion checkpoint used for DDIM/DDPM inversion.
# The notebooks used both "CompVis/stable-diffusion-v1-4" and
# "runwayml/stable-diffusion-v1-5"; v1-4 is the default.
SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")

# Number of denoising/inversion steps.  T (the temporal length of a TRE
# feature) equals this value.
NUM_INFERENCE_STEPS = 20
T = NUM_INFERENCE_STEPS

# Stochasticity of the DDPM/DDIM inversion (eta = 1.0 -> DDPM, eta = 0 -> DDIM).
ETA = 1.0

# VAE latent scaling factor used by Stable Diffusion.
VAE_SCALING_FACTOR = 0.18215

# --------------------------------------------------------------------------- #
# TRE feature tensor convention: (B, T, D, H, W)
# --------------------------------------------------------------------------- #
# The DNSAMNet classifier operates on (B, T=20, D=8, H=16, W=16) tensors.
# The raw Stable Diffusion latent produced by ``over_denosing`` has D=4 channels
# and a 32x32 spatial resolution (256 / 8); adapt these values to whatever the
# precomputed .pt features actually contain.
FEATURE_DIM = 8      # latent-channel dimension expected by the classifier
FEATURE_H = 16
FEATURE_W = 16

# --------------------------------------------------------------------------- #
# Dataset roots
# --------------------------------------------------------------------------- #
# Raw image datasets (GenImage / ForenSynths), laid out as ImageFolder trees:
#     <GENIMAGE_ROOT>/<generator>/train/<class>/*.png
#     <GENIMAGE_ROOT>/<generator>/val/<class>/*.png
GENIMAGE_ROOT = Path(os.environ.get("GENIMAGE_ROOT", "data/genimage"))

# Precomputed TRE feature tensors (.pt), laid out as a DatasetFolder tree:
#     <TRE_FEATURE_ROOT>/train/<class>/*.pt
#     <TRE_FEATURE_ROOT>/test/<generator>/<class>/*.pt
TRE_FEATURE_ROOT = Path(os.environ.get("TRE_FEATURE_ROOT", "data/tre_features"))

# The eight GenImage generators used for evaluation.
GENERATORS = [
    "adm",
    "biggan",
    "glide",
    "midjourney",
    "sdv4",
    "sdv5",
    "vqdm",
    "wukong",
]

# The generator whose training split is used to train the detector.
TRAIN_GENERATOR = "sdv4"  # Stable Diffusion v1.4 training split

# Class label mapping used when saving features.
LABEL_MAP = {0: "real", 1: "fake"}

# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 100
EMBED_DIM = 128
NUM_HEADS = 8

# Where the trained detector weights are written / read from.
CHECKPOINT = Path(os.environ.get("CHECKPOINT", "weights/DNSAM_new.pt"))
