#!/usr/bin/env python3
"""Training entry point for navigation models."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split

SRC_PATH = Path(__file__).resolve().parents[2] / 'src'
import sys
sys.path.append(str(SRC_PATH))

from dnn_guidance.data_loader import PathfindingDataset, _pair_files
from dnn_guidance.trainer import train_one_epoch, validate_one_epoch
from dnn_guidance.loss import DiceFocalLoss
from dnn_guidance.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a navigation model")
    parser.add_argument("config", type=str, help="Training configuration YAML")
    parser.add_argument(
        "--model",
        type=str,
        default="unet_film",
        choices=["unet_film", "hr_film_net"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Optional YAML file with model hyperparameters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples_dir = Path(cfg["samples_dir"])  # paths relative to CWD
    gt_dir = Path(cfg["ground_truth_dir"])
    pairs = _pair_files(samples_dir, gt_dir)
    train_pairs, val_pairs = train_test_split(pairs, test_size=cfg["val_split"], random_state=seed)

    train_ds = PathfindingDataset(samples_dir, gt_dir, augment=True)
    val_ds = PathfindingDataset(samples_dir, gt_dir, augment=False)
    train_ds.pairs = train_pairs
    val_ds.pairs = val_pairs

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
    )

    device = torch.device(cfg.get("device", "cpu"))
    model_name = cfg.get("model_name", args.model)
    model = create_model(model_name, args.model_config)
    model.to(device)

    opt_cfg = cfg["optimizer"]
    optim_cls = getattr(optim, opt_cfg["name"])
    optimizer = optim_cls(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    scheduler = None
    sched_cfg = cfg.get("scheduler")
    if sched_cfg and sched_cfg.get("name") == "CosineAnnealing":
        t_max = cfg["epochs"] - sched_cfg.get("warmup_epochs", 0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    loss_cfg = cfg["loss"]
    loss_fn = DiceFocalLoss(
        dice_weight=loss_cfg["dice_weight"],
        focal_weight=loss_cfg["focal_weight"],
        focal_gamma=loss_cfg["focal_gamma"],
    )

    scaler = GradScaler(enabled=cfg.get("use_amp", True))
    best_dice = -1.0
    checkpoints_dir = Path(cfg.get("checkpoints_dir", "checkpoints"))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / f"{model_name}_best.pth"

    for epoch in range(cfg["epochs"]):
        print(f"--- Epoch {epoch + 1}/{cfg['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_dice = validate_one_epoch(model, val_loader, loss_fn, device)
        if scheduler:
            scheduler.step()

        print(
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)

    print(f"Best Dice: {best_dice:.4f}. Model saved to {best_path}")


if __name__ == "__main__":
    main()
