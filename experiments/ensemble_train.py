"""Two-inverter ensemble on the eta=0 features.

Scheme C measured how well ONE inverter (SD 1.4) round-trips an image, and that
turned out to be family-specific. This asks whether pairing it with a second
inverter helps: the two feature tensors are concatenated on the channel axis,
so the model sees (T=20, 8, 32, 32) instead of (T=20, 4, 32, 32).

Modes:
  --mode ensemble   SD1.4 features ++ SD1.5 features   (8 channels)
  --mode b          SD1.5 features alone               (4 channels, control)

    python ensemble_train.py --mode ensemble --gens sdv4 sdv5 \
        --ckpt weights/ensemble.pt --results results_ensemble.json
    python ensemble_train.py --mode ensemble --eval-only \
        --ckpt weights/ensemble.pt --results results_ensemble_s2.json \
        --features-a features_eta0 --features-b features_sd15 \
        --gens adm biggan glide midjourney vqdm wukong

Paths follow $TRE_HOME (default ~/tre); override TRE_REPO / TRE_FEATURES if needed.
"""
import argparse, glob, json, os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

TRE_HOME = os.environ.get("TRE_HOME", os.path.expanduser("~/tre"))
REPO = os.environ.get("TRE_REPO", os.path.join(TRE_HOME, "repo"))
sys.path.insert(0, REPO)
from src.models.resnet_baseline import TemporalAggregation, SpatialFocusing

DEVICE = torch.device("cuda:0")


class ResNet18_nch(nn.Module):
    def __init__(self, in_ch, num_classes=2):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):            # [B, H, W, C]
        return self.model(x.permute(0, 3, 1, 2))


class Net(nn.Module):
    """Same skeleton as the single-inverter runs, widened to C channels."""

    def __init__(self, C):
        super().__init__()
        self.temporal = TemporalAggregation(T=20, D=C, num_heads=4, num_layers=2)
        self.spatial = SpatialFocusing(H=32, W=32, D=1, num_heads=1, num_layers=2)
        self.classifier = ResNet18_nch(C)

    def forward(self, x):            # [B, T, C, H, W]
        B, T, C, H, W = x.shape
        xt = x.permute(0, 2, 3, 4, 1).reshape(B * H * W, T, C)
        last = self.temporal.attn_blocks(xt)[:, -1, :].view(B, H, W, C)
        return self.classifier(last * self.spatial(x.mean(dim=2)))


class Paired(Dataset):
    """Features for one split, optionally concatenating a second inverter's copy."""

    def __init__(self, root_a, root_b, sub, mode):
        self.mode = mode
        self.items = []
        for cls, y in (("real", 0), ("fake", 1)):
            for pa in sorted(glob.glob(os.path.join(root_a, sub, cls, "*.pt"))):
                pb = os.path.join(root_b, sub, cls, os.path.basename(pa))
                if not os.path.exists(pb):
                    continue          # both inverters are required
                self.items.append((pa, pb, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pa, pb, y = self.items[i]
        if self.mode == "b":
            return torch.load(pb, weights_only=True).float(), y
        a = torch.load(pa, weights_only=True).float()
        b = torch.load(pb, weights_only=True).float()
        return torch.cat([a, b], dim=1), y      # (T, 8, H, W)


@torch.no_grad()
def evaluate(model, ds, batch=64):
    model.eval()
    correct = n = 0
    scores, labels = [], []
    for x, y in DataLoader(ds, batch_size=batch, num_workers=6):
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
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ensemble", "b"], default="ensemble")
    p.add_argument("--features-a", default=os.path.join(TRE_HOME, "features_eta0"))
    p.add_argument("--features-b", default=os.path.join(TRE_HOME, "features_sd15"))
    p.add_argument("--gens", nargs="*", default=["sdv4", "sdv5"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    C = 8 if args.mode == "ensemble" else 4
    model = Net(C).to(DEVICE)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    if not args.eval_only:
        full = Paired(args.features_a, args.features_b, "train", args.mode)
        print(f"mode={args.mode} channels={C} train pairs={len(full)}", flush=True)
        g = torch.Generator().manual_seed(42)
        nval = max(1, int(len(full) * args.val_frac))
        tr, va = torch.utils.data.random_split(full, [len(full) - nval, nval], generator=g)
        loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best = 0.0
        for ep in range(args.epochs):
            model.train(); seen = hit = 0; ls = 0.0
            t0 = time.time()
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x); loss = crit(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                ls += loss.item() * len(y); seen += len(y); hit += (out.argmax(-1) == y).sum().item()
            vacc, vap, _ = evaluate(model, va)
            print(f"epoch {ep+1}/{args.epochs} loss {ls/seen:.4f} acc {hit/seen:.4f}"
                  f" val_acc {vacc} val_ap {vap} ({time.time()-t0:.0f}s)", flush=True)
            if vacc >= best:
                best = vacc; torch.save(model.state_dict(), args.ckpt)
        print("best val acc:", best, flush=True)

    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    out = {"mode": args.mode, "channels": C, "per_generator": {}}
    for gen in args.gens:
        ds = Paired(args.features_a, args.features_b, os.path.join("test", gen), args.mode)
        if not len(ds):
            print(f"  {gen}: no paired features, skipped", flush=True); continue
        acc, ap, n = evaluate(model, ds)
        out["per_generator"][gen] = {"n": n, "acc": acc, "ap": ap}
        print(f"  {gen:11s} acc {acc}  ap {ap}  (n={n})", flush=True)
    if out["per_generator"]:
        out["mean_acc"] = round(sum(v["acc"] for v in out["per_generator"].values())
                                / len(out["per_generator"]), 4)
    json.dump(out, open(args.results, "w"), indent=1)
    print("wrote", args.results, flush=True)


if __name__ == "__main__":
    main()
