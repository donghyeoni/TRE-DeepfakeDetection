"""Does a 3-class head remove biggan's inversion?

Scheme C scored 39.8% on biggan -- below chance, i.e. the learned real/fake
direction points the wrong way for a GAN. Here the head predicts
real / diffusion-fake / GAN-fake instead, so "fake" no longer has to be one
direction. Detection accuracy is then measured by collapsing the two fake
classes back into one.

Trained on the first half of each generator's features (biggan is the only GAN),
evaluated on the second half plus the full set per generator. Note that the
per-generator figures therefore include training material -- the clean number is
the validation binary accuracy.

    python threeclass_train.py

Set TRE_REPO and TRE_FEATURES if they are not at the defaults below.
"""
import argparse, glob, json, os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

REPO = os.environ.get("TRE_REPO", "/home/j-i15a401/tre/repo")
sys.path.insert(0, REPO)
from src.models.resnet_baseline import (TemporalAggregation, SpatialFocusing,
                                        ResNet18_4ch, AttentionClassifier)

GENS = ["adm", "biggan", "glide", "midjourney", "vqdm", "wukong"]
GAN = {"biggan"}
ROOT = os.environ.get("TRE_FEATURES", "/home/j-i15a401/tre/features_eta0/test")
DEVICE = torch.device("cuda:0")
# 0 = real, 1 = diffusion-fake, 2 = GAN-fake
LABELS = {"real": 0, "diffusion": 1, "gan": 2}


class Half(Dataset):
    def __init__(self, gen, half):
        fake_cls = LABELS["gan"] if gen in GAN else LABELS["diffusion"]
        self.items = []
        for cls, y in (("real", LABELS["real"]), ("fake", fake_cls)):
            fs = sorted(glob.glob(os.path.join(ROOT, gen, cls, "*.pt")))
            mid = len(fs) // 2
            self.items += [(p, y) for p in (fs[:mid] if half == "a" else fs[mid:])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return torch.load(p, weights_only=True).float(), y


def build_model():
    return AttentionClassifier(
        TemporalAggregation(T=20, D=4, num_heads=4, num_layers=2),
        SpatialFocusing(H=32, W=32, D=1, num_heads=1, num_layers=2),
        ResNet18_4ch(num_classes=3),
    ).to(DEVICE)


@torch.no_grad()
def evaluate(model, ds):
    """3-class accuracy plus binary (real vs any fake) accuracy and AP."""
    model.eval()
    c3 = c2 = n = 0
    scores, bin_labels = [], []
    for x, y in DataLoader(ds, batch_size=64, num_workers=6):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        prob = logits.softmax(-1)
        c3 += (logits.argmax(-1) == y).sum().item()
        fake_prob = prob[:, 1] + prob[:, 2]
        yb = (y > 0).long()
        c2 += ((fake_prob > 0.5).long() == yb).sum().item()
        n += len(y)
        scores += fake_prob.cpu().tolist(); bin_labels += yb.cpu().tolist()
    tp = fp = 0; P = sum(bin_labels); ap = 0.0
    for s, l in sorted(zip(scores, bin_labels), reverse=True):
        if l == 1:
            tp += 1; ap += tp / (tp + fp) / max(P, 1)
        else:
            fp += 1
    return round(c3 / n, 4), round(c2 / n, 4), round(ap, 4), n


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--epochs", type=int, default=20)
    ap_.add_argument("--batch", type=int, default=16)
    ap_.add_argument("--lr", type=float, default=1e-4)
    ap_.add_argument("--outdir", default=os.path.dirname(ROOT.rstrip("/")) or ".")
    args = ap_.parse_args()

    train_ds = ConcatDataset([Half(g, "a") for g in GENS])
    val_ds = ConcatDataset([Half(g, "b") for g in GENS])
    print(f"train={len(train_ds)}  val={len(val_ds)}  (3-class head)", flush=True)

    model = build_model()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=8, pin_memory=True)
    ckpt = os.path.join(args.outdir, "weights", "threeclass.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    best = 0.0
    for ep in range(args.epochs):
        model.train(); seen = hit = 0; loss_sum = 0.0
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); seen += len(y)
            hit += (out.argmax(-1) == y).sum().item()
        a3, a2, apv, _ = evaluate(model, val_ds)
        print(f"epoch {ep+1}/{args.epochs} loss {loss_sum/seen:.4f} acc3 {hit/seen:.4f}"
              f" val_acc3 {a3} val_bin {a2} val_ap {apv} ({time.time()-t0:.0f}s)", flush=True)
        if a2 >= best:
            best = a2; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    out = {"head": "3-class (real / diffusion-fake / GAN-fake)",
           "train_size": len(train_ds), "best_val_binary_acc": best,
           "per_generator": {}}
    print("\n  generator     acc3     binary acc / ap", flush=True)
    for g in GENS:
        a3, a2, apv, n = evaluate(model, ConcatDataset([Half(g, "a"), Half(g, "b")]))
        out["per_generator"][g] = {"n": n, "acc3": a3, "binary_acc": a2, "ap": apv}
        print(f"  {g:11s} {a3:.4f}   {a2:.4f} / {apv:.4f}", flush=True)
    res = os.path.join(args.outdir, "results_threeclass.json")
    json.dump(out, open(res, "w"), indent=1)
    print("wrote", res, flush=True)


if __name__ == "__main__":
    main()
