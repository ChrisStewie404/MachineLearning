import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from resnn import ResidualMLPClassifier
from util import load_dataset


def get_loaders(csv_path, batch_size, val_ratio, random_state):
    inputs, labels = load_dataset(csv_path, labeled=True)
    inputs = torch.from_numpy(inputs.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.int64))

    train_x, val_x, train_y, val_y = train_test_split(
        inputs,
        labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=labels,
    )

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = ResidualMLPClassifier(
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_loader, val_loader = get_loaders(
        args.train_csv, args.batch_size, args.val_ratio, args.random_state
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc and args.save_dir:
            best_val_acc = val_acc
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(args.save_dir) / f"resnn-best.pth")
            print(f"  Saved best checkpoint (val_acc={val_acc:.3f})")

    if args.save_dir:
        torch.save(model.state_dict(), Path(args.save_dir) / f"resnn-final.pth")
        print("Saved final checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ResidualMLPClassifier")
    parser.add_argument("--train-csv", type=str, default="../../data/train.csv")
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--n-classes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./resnn_checkpoints")
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()
    train(args)
