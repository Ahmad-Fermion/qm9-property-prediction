"""Train a graph neural network on QM9 quantum-chemical properties.

Example:
    python src/train.py --subset 2000 --epochs 2
    python src/train.py --subset 20000 --epochs 50 --hidden-dim 128
    python src/train.py --subset -1 --epochs 100 --batch-size 128
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import MPNN


class SelectTarget:
    """Select one QM9 target column and store it as a graph-level scalar."""

    def __init__(self, target_idx: int):
        self.target_idx = target_idx

    def __call__(self, data):
        data.y = data.y[:, self.target_idx].view(1, 1).float()
        return data


def parse_args():
    parser = argparse.ArgumentParser(description="Train MPNN on QM9.")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory.")
    parser.add_argument("--target-idx", type=int, default=4, help="QM9 target index. Default: 4 = HOMO-LUMO gap.")
    parser.add_argument("--target-name", type=str, default="gap", help="Target name for plots/logs.")
    parser.add_argument("--subset", type=int, default=20000, help="Number of molecules to use. Use -1 for full QM9.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(args):
    # Distance appends interatomic distance to edge_attr.
    transform = T.Compose([SelectTarget(args.target_idx), T.Distance(norm=False)])
    dataset = QM9(root=args.data_dir, transform=transform)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)

    if args.subset > 0:
        indices = indices[: args.subset]

    n = len(indices)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)

    train_dataset = dataset[indices[:n_train]]
    val_dataset = dataset[indices[n_train : n_train + n_val]]
    test_dataset = dataset[indices[n_train + n_val :]]

    return train_dataset, val_dataset, test_dataset


def compute_target_stats(dataset) -> Tuple[float, float]:
    ys = torch.cat([dataset[i].y.view(-1) for i in range(len(dataset))])
    y_mean = ys.mean().item()
    y_std = ys.std().item()
    if y_std == 0:
        raise ValueError("Target standard deviation is zero.")
    return y_mean, y_std


def run_epoch(model, loader, optimizer, device, y_mean, y_std, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0

    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    with torch.set_grad_enabled(train):
        for batch in iterator:
            batch = batch.to(device)
            y = (batch.y - y_mean) / y_std
            pred = model(batch)
            loss = F.mse_loss(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []

    for batch in tqdm(loader, leave=False, desc="test"):
        batch = batch.to(device)
        pred = model(batch) * y_std + y_mean
        preds.append(pred.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)

    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    return preds, targets, mae, r2


def save_plots(train_losses, val_losses, preds, targets, mae, r2, target_name):
    Path("figures").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="Train MSE")
    ax.plot(val_losses, label="Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on normalized target")
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/learning_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.35, s=10, rasterized=True)
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, label="Ideal")
    ax.set_xlabel(f"DFT {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"Predicted vs DFT | MAE={mae:.3f}, R2={r2:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/predicted_vs_actual.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    print(f"Device: {device}")
    print("Loading QM9 dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(args)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    y_mean, y_std = compute_target_stats(train_dataset)
    print(f"Target {args.target_name}: mean={y_mean:.6f}, std={y_std:.6f}")

    sample = train_dataset[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1]
    print(f"Node dim: {node_dim} | Edge dim: {edge_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MPNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, y_mean, y_std, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, y_mean, y_std, train=False)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE {train_loss:.5f} | "
            f"Val MSE {val_loss:.5f} | "
            f"Best Val {best_val:.5f}"
        )

    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    preds, targets, mae, r2 = evaluate(model, test_loader, device, y_mean, y_std)

    print(f"Test MAE: {mae:.6f}")
    print(f"Test R2 : {r2:.6f}")

    save_plots(train_losses, val_losses, preds, targets, mae, r2, args.target_name)

    metrics = {
        "target_idx": args.target_idx,
        "target_name": args.target_name,
        "subset": args.subset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.lr,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "target_mean": y_mean,
        "target_std": y_std,
        "test_mae": float(mae),
        "test_r2": float(r2),
        "best_val_mse_normalized": float(best_val),
        "model_parameters": int(n_params),
    }

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved checkpoints/best_model.pt")
    print("Saved figures/learning_curve.png")
    print("Saved figures/predicted_vs_actual.png")
    print("Saved results/metrics.json")


if __name__ == "__main__":
    main()
