"""Training loop for the DNSAMNet detector.

The model outputs a sigmoid probability, so training uses ``BCELoss`` and the
accuracy threshold is ``0.5`` (the notebook's ``> 0`` threshold was a bug for a
sigmoid output -- see the README notes).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import config
from .models.dnsamnet import TSC


def fit(model, optimizer, criterion, epochs, train_loader, test_loader, device,
        history=None, scheduler=None):
    """Train ``model`` and return the training-history array.

    ``history`` columns: ``[epoch, train_loss, train_acc, val_loss, val_acc]``.
    """
    if history is None:
        history = np.zeros((0, 5))

    base_epochs = len(history)

    for epoch in range(base_epochs, epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        model.train()
        total_samples = 0
        avg_train_loss = avg_train_acc = 0.0

        for batch in tqdm(train_loader):
            # DatasetFolder yields (input, label); some loaders yield (path, input, label).
            inputs, labels = batch[-2], batch[-1]
            inputs, labels = inputs.to(device), labels.to(device).float()
            total_samples += len(labels)
            labels = labels.view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            pred = (outputs > 0.5).float()  # sigmoid output -> threshold at 0.5
            train_acc += (pred == labels).sum().item()

            avg_train_loss = train_loss / total_samples
            avg_train_acc = train_acc / total_samples

        model.eval()
        total_samples = 0
        avg_val_loss = avg_val_acc = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[-2], batch[-1]
                inputs, labels = inputs.to(device), labels.to(device).float()
                total_samples += len(labels)
                labels = labels.view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                pred = (outputs > 0.5).float()
                val_acc += (pred == labels).sum().item()

                avg_val_loss = val_loss / total_samples
                avg_val_acc = val_acc / total_samples

        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {avg_train_loss:.5f} "
            f"acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f} val_acc: {avg_val_acc:.5f}"
        )
        contents = np.array([epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, contents))

        if scheduler is not None:
            scheduler.step()

    return history


def main():
    parser = argparse.ArgumentParser(description="Train the DNSAMNet TRE detector.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--val-generator", default="glide",
                        help="Generator whose test loader is used for validation during training.")
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT))
    args = parser.parse_args()

    from .data.dataset import build_feature_loaders

    device = config.DEVICE
    train_loader, test_loaders = build_feature_loaders(batch_size=args.batch_size)
    val_loader = test_loaders.get(args.val_generator, next(iter(test_loaders.values())))

    model = TSC().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = fit(model, optimizer, criterion, args.epochs, train_loader, val_loader, device)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
