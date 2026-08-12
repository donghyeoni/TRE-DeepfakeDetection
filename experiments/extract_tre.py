"""Batched TRE feature extraction (semantics-preserving port of src.data).

Modes: default = paper-faithful z_t replay (TRE ~ float residue).
--fresh = reconstruct with pre-drawn COMMON noise eps shared across prefixes
(real TRE signal; forward UNet pass skipped).

List lines: <abs_image_path>\t<label:real|fake>
"""
import argparse, os, sys, time, gc
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/home/j-i15a204/tre/repo")
from src import config
from src.data.inversion import get_variance, reverse_step

DEVICE = torch.device("cuda:0")
T = 20

def load_pipe():
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(T)
    pipe.set_progress_bar_config(disable=True)
    return pipe

@torch.no_grad()
def encode_uncond(pipe, B):
    ti = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    emb = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]
    return emb.expand(B, -1, -1)

@torch.no_grad()
def sample_xts(pipe, x0):
    ab = pipe.scheduler.alphas_cumprod.to(DEVICE)
    ts = pipe.scheduler.timesteps.to(DEVICE)
    B, C, H, W = x0.shape
    t2i = {int(v): k for k, v in enumerate(ts)}
    xts = torch.zeros((T + 1, B, C, H, W), device=DEVICE)
    xts[0] = x0
    for t in reversed(ts):
        idx = T - t2i[int(t)]
        xts[idx] = x0 * (ab[t] ** 0.5) + torch.randn_like(x0) * ((1 - ab[t]) ** 0.5)
    return xts

@torch.no_grad()
def forward_zs(pipe, x0, xts, uncond):
    ab = pipe.scheduler.alphas_cumprod.to(DEVICE)
    ts = pipe.scheduler.timesteps.to(DEVICE)
    B, C, H, W = x0.shape
    t2i = {int(v): k for k, v in enumerate(ts)}
    zs = torch.zeros((T, B, C, H, W), device=DEVICE)
    for t in ts:
        idx = T - t2i[int(t)] - 1
        xt = xts[idx + 1]
        noise_pred = pipe.unet(xt, t, encoder_hidden_states=uncond).sample
        xtm1 = xts[idx]
        pred_x0 = (xt - (1 - ab[t]) ** 0.5 * noise_pred) / ab[t] ** 0.5
        prev_t = t - pipe.scheduler.config.num_train_timesteps // T
        ab_prev = ab[prev_t] if prev_t >= 0 else pipe.scheduler.final_alpha_cumprod.to(DEVICE)
        var = get_variance(pipe, t)
        dir_xt = (1 - ab_prev - 1.0 * var) ** 0.5 * noise_pred
        mu_xt = ab_prev ** 0.5 * pred_x0 + dir_xt
        zs[idx] = (xtm1 - mu_xt) / (1.0 * var ** 0.5)
    zs[0] = torch.zeros_like(zs[0])
    return zs

@torch.no_grad()
def reverse_from(pipe, xT, noise, k, uncond):
    ts = pipe.scheduler.timesteps.to(DEVICE)
    sub = ts[-k:]
    t2i = {int(v): j for j, v in enumerate(sub)}
    xt = xT
    for t in sub:
        idx = T - t2i[int(t)] - (T - k + 1)
        noise_pred = pipe.unet(xt, t, encoder_hidden_states=uncond).sample
        xt = reverse_step(pipe, noise_pred, t, xt, eta=1.0, variance_noise=noise[idx])
    return xt

@torch.no_grad()
def tre_batch(pipe, imgs, uncond, fresh=False):
    x0 = pipe.vae.encode(imgs * 2 - 1).latent_dist.sample() * config.VAE_SCALING_FACTOR
    xts = sample_xts(pipe, x0)
    if fresh:
        noise = torch.randn((T,) + tuple(x0.shape), device=DEVICE)
        noise[0] = 0  # mirror zs[0]=0 of the replay path
    else:
        noise = forward_zs(pipe, x0, xts, uncond)
    lat_prev = reverse_from(pipe, xts[T], noise, T, uncond).cpu()
    diffs = []
    for step in range(T):
        lat = reverse_from(pipe, xts[step + 1], noise, step + 1, uncond).cpu()
        diffs.append(lat_prev - lat)
        lat_prev = lat
    return torch.stack(diffs, dim=1)  # (B, T, C, H, W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    items = [l.strip().split("\t") for l in open(args.list) if l.strip()]
    for lbl in ("real", "fake"):
        os.makedirs(os.path.join(args.out, lbl), exist_ok=True)
    todo = []
    for i, (p, lbl) in enumerate(items):
        if i % args.nshards != args.shard:
            continue
        dst = os.path.join(args.out, lbl, f"{i:06d}.pt")
        if not os.path.exists(dst):
            todo.append((p, lbl, dst))
    print(f"total {len(items)}, shard {args.shard}/{args.nshards}, remaining {len(todo)}, fresh={args.fresh}", flush=True)

    pipe = load_pipe()
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    uncond_full = encode_uncond(pipe, args.batch)
    t0 = time.time(); done = 0
    for s in range(0, len(todo), args.batch):
        chunk = todo[s : s + args.batch]
        imgs = []
        for p, lbl, dst in chunk:
            try:
                imgs.append(tf(Image.open(p).convert("RGB")))
            except Exception as e:
                print("SKIP", p, e, flush=True); imgs.append(torch.zeros(3, 256, 256))
        batch = torch.stack(imgs).to(DEVICE)
        uncond = uncond_full[: len(chunk)]
        feats = tre_batch(pipe, batch, uncond, fresh=args.fresh)
        for j, (p, lbl, dst) in enumerate(chunk):
            torch.save(feats[j].to(torch.float16).clone(), dst)
        done += len(chunk)
        rate = done / (time.time() - t0)
        print(f"{done}/{len(todo)} {rate:.2f} img/s eta {((len(todo)-done)/max(rate,1e-9))/3600:.1f}h", flush=True)
        del feats, batch
        if done % (args.batch * 20) == 0:
            gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
