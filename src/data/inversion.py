"""DDPM / DDIM inversion core.

This module consolidates the diffusion-inversion routines that were duplicated
verbatim across ``TRE.ipynb``, ``LoadDataset.ipynb`` and ``tre_dh.ipynb``.  The
maths follows the "edit-friendly DDPM inversion" formulation (Huberman-Spiegelglas
et al.) as used in the original notebooks:

    x0  --(inversion_forward_process)-->  x_T, noise sequence z_t
    x_T --(inversion_reverse_process)-->  reconstruction

The recorded per-step differences of the reconstruction form the Temporal
Reconstruction Error (TRE) feature (see :mod:`src.data.tre_features`).
"""

import torch
from tqdm import tqdm

from .. import config


# --------------------------------------------------------------------------- #
# Text / noise helpers
# --------------------------------------------------------------------------- #
def encode_text(model, prompts):
    """Encode a prompt (or list of prompts) with the pipeline's text encoder."""
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def mu_tilde(model, xt, x0, timestep):
    """mu_tilde(x_t, x_0) -- DDPM paper eq. 7."""
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    )
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
        (alpha_t ** 0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)
    ) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """Sample the noisy latents x_{1:T} from x_0, i.e. P(x_{1:T} | x_0)."""
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5

    _, C, H, W = x0.size()

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps + 1, C, H, W)).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    return xts


def get_variance(model, timestep):
    """DDPM posterior variance sigma_t^2 (see DDIM paper eq. 16)."""
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


# --------------------------------------------------------------------------- #
# Forward (noising) step + full forward process
# --------------------------------------------------------------------------- #
def forward_step(model, model_output, timestep, sample):
    """A single deterministic noising step used when eta == 0."""
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps,
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t

    # "predicted x_0" -- eq. (12) of https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    next_sample = model.scheduler.add_noise(
        pred_original_sample, model_output, torch.LongTensor([next_timestep])
    )
    return next_sample


def inversion_forward_process(
    model,
    x0,
    etas=None,
    prog_bar=False,
    prompt="",
    cfg_scale=3.5,
    num_inference_steps=50,
    eps=None,
):
    """Invert an image latent x0 into (x_T, noise sequence z_t, noisy latents x_t)."""
    if not prompt == "":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)

    _, C, H, W = x0.size()

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros((num_inference_steps, C, H, W)).to(x0.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1

        if not eta_is_zero:
            xt = xts[idx + 1][None]

        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
            if not prompt == "":
                cond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=text_embeddings
                )

        if not prompt == "":
            # classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            xt = forward_step(model, noise_pred, t, xt)
        else:
            xtm1 = xts[idx][None]
            # prediction of x0
            pred_original_sample = (
                xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred
            ) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = (
                t
                - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            )
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else model.scheduler.final_alpha_cumprod
            )

            variance = get_variance(model, t)
            pred_sample_direction = (
                1 - alpha_prod_t_prev - etas[idx] * variance
            ) ** 0.5 * noise_pred

            mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

    if zs is not None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts


# --------------------------------------------------------------------------- #
# Reverse (denoising) step + full reverse process
# --------------------------------------------------------------------------- #
def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    """A single reverse (denoising) step -- eq. (12) of the DDIM paper."""
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    variance = get_variance(model, timestep)
    # Take care of the asymmetric reverse process (asyrp)
    model_output_direction = model_output
    pred_sample_direction = (
        1 - alpha_prod_t_prev - eta * variance
    ) ** 0.5 * model_output_direction
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** 0.5 * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(
    model,
    xT,
    etas=0,
    prompts="",
    cfg_scales=None,
    prog_bar=False,
    zs=None,
    controller=None,
    asyrp=False,
):
    """Reconstruct from xT using the recorded noise sequence zs."""
    batch_size = max(1, len(prompts))

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = (
        encode_text(model, "") if batch_size == 0 else encode_text(model, [""] * batch_size)
    )

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0] :]) if prog_bar else timesteps[-zs.shape[0] :]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}

    for t in op:
        idx = (
            model.scheduler.num_inference_steps
            - t_to_idx[int(t)]
            - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        )

        with torch.no_grad():
            uncond_out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=uncond_embedding
            )
            if prompts:
                cond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=text_embeddings
                )

        z = zs[idx] if zs is not None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)
            noise_pred = uncond_out.sample + cfg_scales_tensor * (
                cond_out.sample - uncond_out.sample
            )
        else:
            noise_pred = uncond_out.sample

        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)

        if controller is not None:
            xt = controller.step_callback(xt)

    return xt, zs


# --------------------------------------------------------------------------- #
# High level entry points
# --------------------------------------------------------------------------- #
@torch.no_grad()
def DDPM_invert_sample(
    pipe,
    image,
    prompt="",
    num_inference_steps=config.NUM_INFERENCE_STEPS,
    steps=config.NUM_INFERENCE_STEPS,
    device=config.DEVICE,
):
    """Encode ``image`` to a latent, invert it and return latent-space tensors.

    Returns ``(w0, zs, wts)`` where ``w0`` is the reconstructed latent, ``zs`` is
    the recorded noise sequence and ``wts`` are the per-timestep noisy latents.
    """
    from diffusers import DDIMScheduler

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_inference_steps)

    w0 = pipe.vae.encode(image * 2 - 1)
    w0 = config.VAE_SCALING_FACTOR * w0.latent_dist.sample()

    wt, zs, wts = inversion_forward_process(
        pipe, w0, etas=config.ETA, prompt=prompt, num_inference_steps=num_inference_steps
    )
    w0, _ = inversion_reverse_process(
        pipe, xT=wts[steps], etas=config.ETA, prog_bar=False, zs=zs[:steps]
    )

    return w0, zs, wts


@torch.no_grad()
def ddpm_reconstruct(
    pipe,
    image,
    prompt="",
    num_inference_steps=config.NUM_INFERENCE_STEPS,
    steps=config.NUM_INFERENCE_STEPS,
    device=config.DEVICE,
):
    """Invert and decode ``image`` back to pixel space (used for visualisation)."""
    w0, _, _ = DDPM_invert_sample(
        pipe,
        image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        steps=steps,
        device=device,
    )
    w0 = 1 / pipe.vae.config.scaling_factor * w0
    images = pipe.vae.decode(w0, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images
