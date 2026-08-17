# Reproduction guide

Everything this repository claims, and how to re-derive it from nothing but the
public GenImage benchmark and a CUDA machine. Read
[`finding-tre-collapse.md`](finding-tre-collapse.md) first if you want the
argument; this file is the operational counterpart.

---

## 0. What was established

Five experiments, all at full benchmark scale (train 60k, evaluation 8
generators x 12k, 16k for sdv5). Every run shares one classifier and one set of
hyperparameters, so the only variable is the feature definition or the training
composition.

| # | Experiment | Question | Answer |
| --- | --- | --- | --- |
| A | Replayed-noise TRE (as published) | does the original feature carry signal? | No — it is mathematically zero, only float residue survives (std 2.7e-4). 57.7% mean |
| B | Fresh common noise | does injecting fresh noise restore the difference? | No — the noise dominates the difference. 50.1% mean, train acc 96% |
| C | Deterministic, eta = 0 | does removing the stochastic term help? | Yes, but family-bound: 78-79% on SD-derived generators, ~49.5% elsewhere, biggan inverts to 39.8% |
| 1 | Leave-one-generator-out | is C's bias a training artefact? | No — held-out accuracy is chance (mean 51.7%) no matter how the training set is mixed |
| 3 | 3-class head (real / diffusion / GAN) | is biggan's inversion structural? | Yes — 39.8% -> 63.4% once GAN fakes get their own class |
| 2 | Two-inverter ensemble | does a second inverter break the family bond? | Not answered — the only public second inverter (SD 1.5) is the same family, and it changes nothing: 78.8 -> 78.9% on SD, 49.5 -> 52.7% elsewhere |

Raw numbers live in `results/*.json`; nothing in the prose is rounded from a
different source.

> The paper in `docs/` reports Ours_TRE 66.0 / TAF 68.0 / TOD 68.4%. Those runs
> never completed and the numbers are not reliable — do not use them as a target.
> STRE and other baselines are cited from their own papers, not reproduced here.

---

## 1. Environment

On a freshly assigned machine, `experiments/bootstrap.sh` does everything in this
section and the next — venv, repo, archives, file lists — and prints the
extraction command to run afterwards:

```bash
git clone https://github.com/donghyeoni/tre-deepfake-detection
bash tre-deepfake-detection/experiments/bootstrap.sh          # env + test data (25 GB)
WITH_TRAIN=1 bash tre-deepfake-detection/experiments/bootstrap.sh   # + train data (96 GB)
```

Every script resolves paths from **`$TRE_HOME`** (default `~/tre`); nothing is
hardcoded to a particular host or account. Override it, or the narrower
`TRE_REPO` / `TRE_DATA` / `TRE_FEATURES`, if the layout differs. The rest of this
section explains what bootstrap installs and why those versions.

```
Python 3.11
torch 2.3.1+cu121      torchvision 0.18.1
diffusers 0.31.0       transformers 4.44.2
numpy<2                accelerate, safetensors
```

Pinning matters in two places:

- **numpy must stay below 2.** `np.trapz` was removed in numpy 2 and the
  training loop dies mid-epoch without it.
- **diffusers 0.31.0 with transformers 4.44.2.** diffusers 0.27 calls
  `huggingface_hub.cached_download`, which no longer exists; newer transformers
  needs `torch.library.custom_op`, which torch 2.3.1 does not have.

```bash
python -m venv venv
./venv/bin/pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
./venv/bin/pip install "numpy<2" diffusers==0.31.0 transformers==4.44.2 \
    accelerate safetensors huggingface_hub hf_transfer
```

Feature extraction runs in **fp32** and stores **fp16**. Scheme A's entire signal
sits at 1e-4, below fp16 resolution, so lowering the compute precision destroys
the very thing that run measures. C tolerates fp16 compute, but keeping both in
fp32 is what makes the comparison controlled.

---

## 2. Data

GenImage, from the [official release](https://github.com/GenImage-Dataset/GenImage)
or the HF mirror `jzousz/GenImage` (same archives, ~72 MB/s with
`HF_HUB_ENABLE_HF_TRANSFER=1`).

| Archive | Size | Contents |
| --- | --- | --- |
| `genimage_test.zip` | 25.4 GB | all 8 generators' test splits (100k images) |
| `stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.z01..z29 + .zip` | 96.4 GB | the sdv1.4 training split |

The training archive is a 30-part zip; `unzip` will not open it, and `unzip` is
often missing on these images anyway. Use 7-Zip:

```bash
wget https://github.com/ip7z/7zip/releases/download/24.09/7z2409-linux-x64.tar.xz
tar -xf 7z2409-linux-x64.tar.xz 7zz
./7zz x -y -o<dest> imagenet_ai_0419_sdv4.zip     # picks up .z01..z29 automatically
```

On-disk layout after unpacking:

```
data/test_images/test/<generator>_imagenet/{ai,nature}/*.PNG|JPEG
data/sdv4/imagenet_ai_0419_sdv4/train/{ai,nature}/*
```

`ai` is the generated class, `nature` the real one. Generators:
`adm, biggan, glide, midjourney, sdv4, sdv5, vqdm, wukong`.

### File lists

`experiments/build_lists.py` writes `data/lists/{train,test_<gen>}.txt`, one
`<abs_path>\t<real|fake>` per line: 30k `ai` + 30k `nature` sampled with
`random.seed(42)` for training, every test image for evaluation.

**A feature file is named after its line index in these lists.** That is the only
link between a tensor and its source image, so if you regenerate lists on a
second machine, verify the ordering matches before merging anything:

```python
md5("\n".join(os.path.basename(l.split("\t")[0]) for l in lines))
```

All eight fingerprints matched across the two servers used here; had they not,
labels would have silently swapped.

---

## 3. Feature extraction

`experiments/extract_tre.py` builds one `(T=20, 4, 32, 32)` fp16 tensor per
image (165 KB). Shared skeleton: VAE-encode to `x0`, walk the latent to noise,
reconstruct from every prefix `k = 1..20`, take differences of consecutive
reconstructions.

| Flag | Scheme | Reverse step |
| --- | --- | --- |
| *(none)* | A | eta=1, the recorded `z_t` replayed — reconstruction is exact by construction |
| `--fresh` | B | eta=1, one pre-drawn random sequence shared across prefixes |
| `--eta0` | C | eta=0, DDIM inversion, no variance term at all |

```bash
python experiments/extract_tre.py \
    --list data/lists/train.txt --out features_eta0/train \
    --batch 48 --eta0 [--shard k --nshards n] [--model <hf-id>]
```

- `--shard k --nshards n` takes every image where `index % n == k`. Existing
  output files are skipped, so runs resume and shards can be re-split mid-flight
  (used here to rebalance across two servers).
- `--model` swaps the inverter (default `CompVis/stable-diffusion-v1-4`).
  **`stabilityai/*` repositories are gated** — anonymous access returns 401, so
  SD 2.1 is unusable without a token; `stable-diffusion-v1-5/stable-diffusion-v1-5`
  is the public stand-in used for the ensemble.

Cost: 250 UNet calls per image for A/B, 230 for C. Measured **0.78 img/s per
L40S** at batch 48 (15.6 GB VRAM), i.e. **~3.1 img/s on four**, so one full pass
over the 160k images is **~14 hours on a 4-GPU node**. Storage is 26 GB per
scheme.

---

## 4. Classifier and protocol

`experiments/train_eval.py`, architecture from the original notebook
(`src/models/resnet_baseline.py`):

```
temporal MHSA (embed = channels, 4 heads, 2 layers) over the T axis
  x  spatial focusing (mean+max pool -> attention -> 1x1 conv -> softmax map)
  -> ResNet18 with a channel-matched first conv, 2 classes
CrossEntropy, Adam 1e-4, batch 16, 20 epochs, 3% held-out validation split
```

Head count is 4 because `embed_dim=4` must be divisible by it; the notebook's 8
is impossible. Two latent bugs had to be fixed before anything ran — see §7.

```bash
python experiments/train_eval.py --features features_eta0 --epochs 20 \
    --ckpt weights/ours_tre_eta0.pt --results results_eta0.json
python experiments/train_eval.py --features features_eta0 --eval-only \
    --ckpt weights/ours_tre_eta0.pt --results results_eta0.json
```

Evaluation reports accuracy **and** average precision per generator; accuracy
alone hides the biggan inversion (its AP falls to 0.36, which is the clearer
signal that the direction flipped).

---

## 5. Experiment catalogue

### A / B / C — the three feature definitions

Extract with the flag from §3, train, evaluate. ~15 h extraction + ~1 h training
each on a 4-GPU node. Results: `results/repro.json`, `results/fresh.json`,
`results/eta0.json`.

The decisive diagnostic is cheap and worth running first on 16 images before
committing GPU-hours — it separates the three schemes immediately:

| scheme | TRE std | mean abs, real | mean abs, generated | ratio |
| --- | --- | --- | --- | --- |
| A | 0.00023 | 0.000167 | 0.000148 | 1.12 |
| B | 0.868 | 0.6377 | 0.6390 | **0.998** |
| C | 0.202 | 0.1183 | 0.1011 | **1.17** |

A ratio at 1.0 (B) means the feature cannot separate the classes no matter what
classifier follows. `experiments/` has this as the smoke path used before each
full run.

### Experiment 1 — leave-one-generator-out

Trains on five generators' first halves (30k) and tests on the sixth, six times.
No extraction: it reuses the scheme-C features. ~20 min per run, four in
parallel. `results/logo.json`.

### Experiment 3 — 3-class head

Same features, head widened to real / diffusion-fake / GAN-fake, trained on the
six generators' first halves; binary accuracy collapses the fake classes.
Zero extraction cost, ~1 h. `results/threeclass.json`.
Per-generator figures in that file include the training halves — the clean
number is `best_val_binary_acc`.

### Experiment 2 — two-inverter ensemble

Extract scheme C a second time with a different inverter, concatenate on the
channel axis (`(20, 8, 32, 32)`), train the same skeleton widened to 8 channels
(`experiments/ensemble_train.py`). An SD1.5-only control runs alongside so that
"ensemble effect" and "different inverter effect" stay separable. Extraction is
another ~14 GPU-hours; training ~1 h. `results/ensemble.json`.

```bash
python experiments/extract_tre.py --list data/lists/train.txt \
    --out features_sd15/train --batch 48 --eta0 \
    --model stable-diffusion-v1-5/stable-diffusion-v1-5
python experiments/ensemble_train.py --mode ensemble --gens sdv4 sdv5 \
    --ckpt weights/ensemble.pt --results results_ensemble.json
python experiments/ensemble_train.py --mode b --gens sdv4 sdv5 \
    --ckpt weights/sd15_only.pt --results results_sd15only.json     # control
```

**This run does not answer its own question.** SD 1.5 is a continuation of
SD 1.4, so both inverters belong to one family; the intended SD 2.1 arm is
unreachable because `stabilityai/*` is gated. A real test needs an inverter from
a different family — pixel-space ADM, or a GAN inversion — and the negative
result here should be read as "same-family ensembling is useless", not as
"ensembling is useless".

---

## 6. Running it across two machines

Splitting **by generator** (not by shard) keeps each generator's features whole
on one machine, which matters because average precision needs the full score
list — accuracy alone could be averaged across machines, AP cannot.

The pattern used here:

1. Machine 1: training set + sdv4/sdv5. Machine 2: the other six generators.
2. Train wherever the training features are.
3. Move the **checkpoint** (45 MB), never the features (26 GB): attach it to a
   GitHub release and `wget` it on the other machine, verifying sha256.
4. Evaluate each machine's own generators, merge the two small JSONs.

---

## 7. Bugs and traps

**Fixed in this repository** (both were latent because the original notebooks
never completed a run):

- `AttentionClassifier` passed a 5-D `(B,T,C,H,W)` tensor to `SpatialFocusing`,
  which unpacks four dimensions — the channel axis is now averaged first.
- `build_dataset.py` indexed `LABEL_MAP` by the numeric ImageFolder label.
  GenImage class dirs sort as `[ai, nature]`, so index 0 is `ai` — fakes were
  being written to `real/` and vice versa. Now mapped by class name.
- Accuracy was thresholded at `> 0` on a sigmoid output (always true) — `> 0.5`.
- `SpatialFocusing.__init__` referenced an undefined `num_head`.
- The library's own eta=0 branch in `inversion_forward_process` is unusable: it
  leaves `xts` undefined and walks the timesteps in descending order.
  `extract_tre.py --eta0` implements standard ascending-timestep DDIM inversion
  instead.

**Operational traps that cost hours here:**

- Killing a driver script leaves its child extractor alive. The orphan keeps
  writing to the log at its own file offset, overwriting completion markers, and
  quietly halves the GPU's throughput. Always `pgrep -af` and match the *actual*
  running command before killing.
- A shell orchestrator that launches workers with `&` and later calls `wait`
  will fall straight through if the workers were killed externally — the next
  stage then runs on incomplete data.
- Patch every machine. `--model` was added on one server only, and the other
  spent a cycle exiting instantly with `unrecognized arguments`.
- A `*.png` line in `.gitignore` silently swallows README figures; `!assets/*.png`
  is needed.
- Competing downloads starve HF metadata requests — a model snapshot can fail
  while a 25 GB archive saturates the link. Serialise them, and gate GPU stages
  behind a smoke test so a failed model fetch never leaves the GPUs idle.

---

## 8. Open threads

`ideas-generalization.md` carries the full list. The short version after
experiments 1 and 3:

- The family bond is a property of the feature, so multi-inverter ensembles
  only help if the inverters differ in *family*, not in version — gated
  `stabilityai` repositories are the practical obstacle.
- The 3-class result suggests structure worth exploiting: a GAN/diffusion split
  before the real/fake decision.
- Nothing here has been tested under JPEG, resize or blur. Any generalisation
  claim needs that table.
