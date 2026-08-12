# Reproduction & improvement experiment harness

Scripts used to re-run the full-scale experiment (GenImage; train: sdv1.4
30k real + 30k fake, seed 42; test: 8 generators x 12k) on a 4x L40S server.
Paths are server-specific — adjust the constants at the top of each script
before reuse.

| File | Role |
| --- | --- |
| `build_lists.py` | Build train/test list files (`<path>\t<label>`) from unpacked GenImage |
| `extract_tre.py` | Batched TRE extraction. Default = paper-faithful z_t replay; `--fresh` = common-random-noise reconstruction (the improvement condition). `--shard k --nshards n` for multi-GPU splits |
| `driver_gpu.sh` / `driver_gpu_fresh.sh` | Per-GPU shard runners (train list, then each generator's test list) |
| `orchestrate.sh` | Stage chain: wait for train features → train → wait for extraction → eval (`results.json`) → fresh extraction → fresh train/eval (`results_fresh.json`) |
| `train_eval.py` | Ours_TRE classifier (temporal MHSA x spatial focusing → ResNet18-4ch, CE, Adam 1e-4) + per-generator Accuracy/AP evaluation |

Environment: Python 3.11, torch 2.3.1+cu121, torchvision 0.18.1,
diffusers 0.31.0, transformers 4.44.2, numpy<2. Features are stored as
fp16 `(T=20, 4, 32, 32)` tensors, one `.pt` per image.

Why two extraction modes exist — see `docs/finding-tre-collapse.md`.
