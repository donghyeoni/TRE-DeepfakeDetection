"""Evaluation: per-generator accuracy and Average Precision, plus curves.

Accuracy uses a ``0.5`` threshold to match the model's sigmoid output (the
notebook used ``> 0``, which is wrong for a probability -- see the README).  The
model already returns probabilities, so AP does not re-apply a sigmoid.
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from . import config
from .models.dnsamnet import TSC


def acc(model, test_loader, device=config.DEVICE):
    """Classification accuracy over ``test_loader``."""
    model.eval()
    total_samples = 0
    val_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[-2], batch[-1]
            inputs, labels = inputs.to(device), labels.to(device)
            total_samples += len(labels)
            labels = labels.view(-1, 1).float()

            outputs = model(inputs)
            pred = (outputs > 0.5).float()  # sigmoid output -> threshold at 0.5
            val_acc += (pred == labels).sum().item()

    return val_acc / max(total_samples, 1)


def ap(model, test_loader, device=config.DEVICE):
    """Average Precision over ``test_loader`` (model already outputs probs)."""
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[-2], batch[-1]
            inputs = inputs.to(device)
            probs = model(inputs).cpu()  # already a sigmoid probability
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probs.numpy())

    return average_precision_score(y_true, y_score)


def evaluate(model, test_loaders, device=config.DEVICE):
    """Return ``(acc_per_gen, ap_per_gen)`` dicts plus their means, and print them."""
    acc_per_gen = {}
    ap_per_gen = {}
    for gen, loader in test_loaders.items():
        acc_per_gen[gen] = acc(model, loader, device)
        ap_per_gen[gen] = ap(model, loader, device)

    print("== Accuracy ==")
    for gen, value in acc_per_gen.items():
        print(f"{gen} : {value:.3f}")
    if acc_per_gen:
        print(f"mean : {np.mean(list(acc_per_gen.values())):.3f}\n")

    print("== Average Precision ==")
    for gen, value in ap_per_gen.items():
        print(f"{gen} : {value:.3f}")
    if ap_per_gen:
        print(f"mean : {np.mean(list(ap_per_gen.values())):.3f}")

    return acc_per_gen, ap_per_gen


def plot_history(history):
    """Plot training/validation loss and accuracy curves."""
    import matplotlib.pyplot as plt

    num_epochs = len(history)
    unit = max(num_epochs / 10, 1)

    for col, name in [(1, "loss"), (2, "accuracy")]:
        train_col = col
        test_col = col + 2
        plt.figure(figsize=(6, 5))
        plt.plot(history[:, 0], history[:, train_col], "b", label="train")
        plt.plot(history[:, 0], history[:, test_col], "k", label="test")
        plt.xticks(np.arange(0, num_epochs + 1, unit))
        plt.xlabel("epoch")
        plt.ylabel(name)
        plt.title(f"learning curve ({name})")
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the DNSAMNet TRE detector.")
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    from .data.dataset import build_feature_loaders

    device = config.DEVICE
    _, test_loaders = build_feature_loaders(batch_size=args.batch_size)

    model = TSC().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    for param in model.parameters():
        param.requires_grad = False

    evaluate(model, test_loaders, device)


if __name__ == "__main__":
    main()
