"""Leave-one-generator-out on the eta=0 features.

Question: is scheme C's family bias a property of the feature, or an artefact of
having trained on sdv1.4 alone? Here the classifier is trained on five
generators and tested on the sixth, so training diversity is the only thing that
changed relative to the single-generator run.

Each generator's 12k test features are split in half per class: the first half
("a") is training material, the second half ("b") is held out. The held-out
generator contributes nothing to training and is evaluated on both halves.

    python logo_train.py --holdout adm

Set TRE_REPO and TRE_FEATURES if they are not at the defaults below.
"""
import argparse, glob, json, os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

TRE_HOME = os.environ.get("TRE_HOME", os.path.expanduser("~/tre"))
REPO = os.environ.get("TRE_REPO", os.path.join(TRE_HOME, "repo"))
sys.path.insert(0, REPO)
from src.models.resnet_baseline import (TemporalAggregation, SpatialFocusing,
                                        ResNet18_4ch, AttentionClassifier)

GENS = ["adm", "biggan", "glide", "midjourney", "vqdm", "wukong"]
ROOT = os.environ.get("TRE_FEATURES", os.path.join(TRE_HOME, "features_eta0", "test"))
DEVICE = torch.device("cuda:0")


class Half(Dataset):
    """One generator, one half of each class. half='a' trains, half='b' is held out."""

    def __init__(self, gen, half):
        self.items = []
        for cls, y in (("real", 0), ("fake", 1)):
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
        ResNet18_4ch(num_classes=2),
    ).to(DEVICE)


@torch.no_grad()
def evaluate(model, ds, batch=64):
    model.eval()
    loader = DataLoader(ds, batch_size=batch, num_workers=6)
    correct = n = 0
    scores, labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item(); n += len(y)
        scores += logits.softmax(-1)[:, 1].cpu().tolist(); labels += y.cpu().tolist()
    tp = fp = 0; P = sum(labels); ap = 0.0
    for s, l in sorted(zip(scores, labels), reverse=True):
        if l == 1:
            tp += 1; ap += tp / (tp + fp) / max(P, 1)
        else:
            fp += 1
    return round(correct / max(n, 1), 4), round(ap, 4), n


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--holdout", required=True, choices=GENS)
    ap_.add_argument("--epochs", type=int, default=20)
    ap_.add_argument("--batch", type=int, default=16)
    ap_.add_argument("--lr", type=float, default=1e-4)
    ap_.add_argument("--outdir", default=os.path.dirname(ROOT.rstrip("/")) or ".")
    args = ap_.parse_args()

    train_gens = [g for g in GENS if g != args.holdout]
    train_ds = ConcatDataset([Half(g, "a") for g in train_gens])
    val_ds = ConcatDataset([Half(g, "b") for g in train_gens])
    print(f"holdout={args.holdout}  train={len(train_ds)} from {train_gens}"
          f"  val={len(val_ds)}", flush=True)

    model = build_model()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=8, pin_memory=True)

    best = 0.0
    ckpt = os.path.join(args.outdir, "weights", f"logo_{args.holdout}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    for ep in range(args.epochs):
        model.train(); seen = hit = 0; loss_sum = 0.0
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); seen += len(y)
            hit += (out.argmax(-1) == y).sum().item()
        vacc, vap, _ = evaluate(model, val_ds)
        print(f"epoch {ep+1}/{args.epochs} loss {loss_sum/seen:.4f} acc {hit/seen:.4f}"
              f" val_acc {vacc} val_ap {vap} ({time.time()-t0:.0f}s)", flush=True)
        if vacc >= best:
            best = vacc; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    out = {"holdout": args.holdout, "train_generators": train_gens,
           "train_size": len(train_ds), "best_val_acc": best, "per_generator": {}}
    for g in GENS:
        acc, apv, n = evaluate(model, ConcatDataset([Half(g, "a"), Half(g, "b")]))
        seen_in_training = g != args.holdout
        out["per_generator"][g] = {"n": n, "acc": acc, "ap": apv,
                                   "in_training": seen_in_training}
        print(f"  {g:11s} acc {acc}  ap {apv}"
              f"  {'(trained)' if seen_in_training else '<-- HELD OUT'}", flush=True)
    res = os.path.join(args.outdir, f"results_logo_{args.holdout}.json")
    json.dump(out, open(res, "w"), indent=1)
    print("wrote", res, flush=True)


if __name__ == "__main__":
    main()
