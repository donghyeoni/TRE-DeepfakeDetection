"""CLI: extract TRE features from a folder of images and save them as ``.pt``.

For every image in a torchvision ``ImageFolder`` tree, the diffusion model is
inverted and the step-wise latent differences (TRE feature) are saved as a
``(T, D, H, W)`` tensor under ``<out>/<class>/imageN.pt``.

Example
-------
    python -m src.data.build_dataset \
        --images data/genimage/sdv1.4/val \
        --out data/tre_features/test/sdv4 \
        --limit 1000
"""

import argparse
import gc
import os

import torch
from tqdm import tqdm

from .. import config
from .dataset import image_folder
from .tre_features import over_denosing


def load_pipeline(model_id=config.SD_MODEL_ID, device=config.DEVICE):
    """Load a Stable Diffusion pipeline configured with a DDIM scheduler."""
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def build(images_dir, out_dir, limit=None, num_inference_steps=config.NUM_INFERENCE_STEPS,
          model_id=config.SD_MODEL_ID, device=config.DEVICE):
    """Extract and save TRE features for every image under ``images_dir``."""
    pipe = load_pipeline(model_id=model_id, device=device)
    dataset = image_folder(images_dir, train=False)

    for label in config.LABEL_MAP.values():
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)

    indices = range(len(dataset))
    if limit is not None:
        perm = torch.randperm(len(dataset))[:limit]
        indices = perm.tolist()

    for idx in tqdm(indices, total=len(indices)):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            step_diffs = over_denosing(
                pipe, image, device=device, num_inference_steps=num_inference_steps, mode="data"
            )
        diffs = torch.stack(step_diffs, dim=0)

        save_dir = os.path.join(out_dir, config.LABEL_MAP[label])
        save_path = os.path.join(save_dir, f"image{idx}.pt")
        torch.save(diffs.cpu(), save_path)

        del step_diffs, diffs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Extract TRE features from images.")
    parser.add_argument("--images", required=True, help="Input ImageFolder directory.")
    parser.add_argument("--out", required=True, help="Output directory for .pt features.")
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample at most N images.")
    parser.add_argument("--steps", type=int, default=config.NUM_INFERENCE_STEPS,
                        help="Number of inversion/denoising steps (= T).")
    parser.add_argument("--model-id", default=config.SD_MODEL_ID, help="Stable Diffusion model id.")
    args = parser.parse_args()

    build(
        args.images,
        args.out,
        limit=args.limit,
        num_inference_steps=args.steps,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
