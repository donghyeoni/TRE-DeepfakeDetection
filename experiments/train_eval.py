"""Train the paper's Ours_TRE classifier and evaluate per generator.

Model: AttentionClassifier = temporal MHSA (embed 4, heads 4, 2 layers)
x spatial focusing -> ResNet18(4ch, 2 classes). CrossEntropy, Adam 1e-4,
batch 16 (LoadDataset.ipynb cell 16 recipe; head count fixed to 4 because
embed_dim=4 requires num_heads | 4).
"""
import argparse, glob, os, sys, time, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRE_HOME = os.environ.get("TRE_HOME", os.path.expanduser("~/tre"))
sys.path.insert(0, os.environ.get("TRE_REPO", os.path.join(TRE_HOME, "repo")))
from src.models.resnet_baseline import TemporalAggregation, SpatialFocusing, ResNet18_4ch, AttentionClassifier

DEVICE = torch.device("cuda:0")

class FeatDataset(Dataset):
    def __init__(self, root):
        self.items = [(p, 0) for p in sorted(glob.glob(os.path.join(root, "real", "*.pt")))]
        self.items += [(p, 1) for p in sorted(glob.glob(os.path.join(root, "fake", "*.pt")))]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return torch.load(p, weights_only=True).float(), y

def build_model():
    temporal = TemporalAggregation(T=20, D=4, num_heads=4, num_layers=2)
    spatial = SpatialFocusing(H=32, W=32, D=1, num_heads=1, num_layers=2)
    return AttentionClassifier(temporal, spatial, ResNet18_4ch(num_classes=2)).to(DEVICE)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = n = 0
    scores, labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        prob = logits.softmax(-1)[:, 1]
        correct += (logits.argmax(-1) == y).sum().item(); n += len(y)
        scores += prob.cpu().tolist(); labels += y.cpu().tolist()
    # average precision (fake = positive)
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0; P = sum(labels); ap = 0.0
    for s, l in pairs:
        if l == 1:
            tp += 1; ap += tp / (tp + fp) / max(P, 1)
        else:
            fp += 1
    return correct / max(n, 1), ap

def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("--features", default=os.path.join(TRE_HOME, "features"))
    apr.add_argument("--epochs", type=int, default=20)
    apr.add_argument("--batch", type=int, default=16)
    apr.add_argument("--lr", type=float, default=1e-4)
    apr.add_argument("--val-frac", type=float, default=0.03)
    apr.add_argument("--ckpt", default=os.path.join(TRE_HOME, "weights", "ours_tre.pt"))
    apr.add_argument("--eval-only", action="store_true")
    apr.add_argument("--results", default=os.path.join(TRE_HOME, "results.json"))
    args = apr.parse_args()

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    model = build_model()

    if not args.eval_only:
        full = FeatDataset(os.path.join(args.features, "train"))
        print("train features:", len(full), flush=True)
        g = torch.Generator().manual_seed(42)
        nval = max(1, int(len(full) * args.val_frac))
        tr, va = torch.utils.data.random_split(full, [len(full) - nval, nval], generator=g)
        trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
        val = DataLoader(va, batch_size=64, num_workers=4)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best = 0.0
        for ep in range(args.epochs):
            model.train(); tot = seen = closs = 0
            t0 = time.time()
            for x, y in trl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x); loss = crit(out, y)
                opt.zero_grad(); loss.backward(); opt.step()
                closs += loss.item() * len(y); seen += len(y)
                tot += (out.argmax(-1) == y).sum().item()
            vacc, vap = evaluate(model, val)
            print(f"epoch {ep+1}/{args.epochs} loss {closs/seen:.4f} acc {tot/seen:.4f} val_acc {vacc:.4f} val_ap {vap:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if vacc >= best:
                best = vacc; torch.save(model.state_dict(), args.ckpt)
        print("best val acc:", best, flush=True)

    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    results = {}
    for gen in ["adm", "biggan", "glide", "midjourney", "sdv4", "sdv5", "vqdm", "wukong"]:
        root = os.path.join(args.features, "test", gen)
        if not os.path.isdir(root): continue
        ds = FeatDataset(root)
        if not len(ds): continue
        acc, ap = evaluate(model, DataLoader(ds, batch_size=64, num_workers=8))
        results[gen] = {"n": len(ds), "acc": round(acc, 4), "ap": round(ap, 4)}
        print(gen, results[gen], flush=True)
    if results:
        mean_acc = sum(v["acc"] for v in results.values()) / len(results)
        results["mean_acc"] = round(mean_acc, 4)
        json.dump(results, open(args.results, "w"), indent=1)
        print("MEAN ACC", mean_acc, flush=True)

if __name__ == "__main__":
    main()
