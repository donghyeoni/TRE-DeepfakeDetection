# Results

This pipeline (Stable Diffusion inversion → TRE features → DNSAMNet classifier)
requires a pretrained diffusion model, the generated-image benchmarks
(GenImage / ForenSynths), and a GPU. It is therefore **not re-run in this
repository**.

Instead, the figures and logs embedded in the original notebooks are preserved
under [`results/notebook_reference/`](results/notebook_reference/):

- **`tre_dh__cell00_*.png`** (10 figures) — TRE / reconstruction-error and
  attention visualizations from the exploratory notebook.
- **`DNSAMNet__cell15_1.png`**, **`TRE__cell06_1.png`** — model / feature
  figures from the DNSAMNet and TRE notebooks.
- **`*.log`** — the notebooks' captured stdout (shapes, intermediate values).

## Reproducing

```bash
# 1. Build TRE features from your image folders (needs Stable Diffusion + GPU)
python -m src.data.build_dataset --data-root /path/to/GenImage --out features/

# 2. Train the DNSAMNet classifier on the precomputed features
python -m src.train --features features/

# 3. Evaluate per-generator accuracy / average precision
python -m src.eval --features features/ --weights weights/best.pt
```

See the project report in `docs/` for the authors' quantitative results
(per-generator accuracy and average precision).
