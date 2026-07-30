"""Temporal Reconstruction Error (TRE) feature extraction.

Given an image, the diffusion model is inverted (see :mod:`src.data.inversion`)
and then reconstructed step by step.  The per-step differences of the
reconstructed latents form the TRE sequence -- a ``(T, D, H, W)`` tensor that is
the input to the DNSAMNet detector.
"""

import gc

import torch

from .. import config
from .inversion import DDPM_invert_sample, inversion_reverse_process


@torch.no_grad()
def over_denosing(
    pipe,
    image,
    device=config.DEVICE,
    prompt="",
    num_inference_steps=config.NUM_INFERENCE_STEPS,
    prog_bar=False,
    mode="data",
):
    """Reconstruct ``image`` step by step and return the step-wise latent diffs.

    Parameters
    ----------
    pipe : StableDiffusionPipeline
        A pipeline whose scheduler is a DDIM scheduler.
    image : torch.Tensor
        A ``(1, 3, H, W)`` image tensor in ``[0, 1]``.
    mode : {"data", "image"}
        ``"data"`` returns the list of ``T`` step-wise latent differences (the
        TRE feature); ``"image"`` decodes and returns the reconstructed image.
    """
    latent, zs, wts = DDPM_invert_sample(
        pipe,
        image,
        prompt="",
        num_inference_steps=num_inference_steps,
        steps=num_inference_steps,
        device=device,
    )

    # step_latents holds T+1 reconstructed latents; step_diffs holds T diffs.
    step_latents = [latent.cpu()]
    step_diffs = []

    for step in range(len(zs)):
        restored_latent, _ = inversion_reverse_process(
            pipe,
            xT=wts[step + 1],       # latent at the current timestep
            etas=config.ETA,        # noise scale
            prompts=prompt,         # guidance prompt
            prog_bar=False,
            zs=zs[: step + 1],      # noise predictions up to the current step
        )

        step_latents.append(restored_latent.cpu().detach())

        del restored_latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if mode == "data":
        for i in range(len(step_latents) - 1):
            step_diffs.append(step_latents[i].squeeze(0) - step_latents[i + 1].squeeze(0))
        return step_diffs

    # mode == "image": decode the final latent back to pixel space.
    w0 = step_latents[0].to(device) * 1 / pipe.vae.config.scaling_factor
    images = pipe.vae.decode(w0, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def compute_tre(pipe, image, device=config.DEVICE, prompt="", num_inference_steps=config.NUM_INFERENCE_STEPS):
    """Return the TRE feature for a single image as a ``(T, D, H, W)`` tensor."""
    step_diffs = over_denosing(
        pipe,
        image,
        device=device,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        mode="data",
    )
    return torch.stack(step_diffs, dim=0)
