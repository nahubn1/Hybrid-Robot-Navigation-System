"""Training and validation utilities for DNN-guided navigation models."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

__all__ = ["dice_score", "train_one_epoch", "validate_one_epoch"]


def dice_score(logits: torch.Tensor, targets: torch.Tensor, *, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Compute the Dice coefficient for binary segmentation.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape ``[B, 1, H, W]``.
    targets : torch.Tensor
        Binary ground truth masks with the same shape as ``logits``.
    threshold : float, optional
        Probability threshold to convert predictions to binary values. Defaults
        to ``0.5``.
    eps : float, optional
        Small constant for numerical stability. Defaults to ``1e-6``.

    Returns
    -------
    torch.Tensor
        Dice score averaged over the batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
) -> float:
    """Run a single training epoch.

    Sets the model to training mode and iterates over ``loader`` performing the
    forward and backward passes using mixed precision. Gradients are scaled with
    ``scaler`` for stability.

    Returns the average training loss.
    """
    model.train()
    epoch_loss = 0.0
    num_samples = 0

    for (grid, robot), targets in tqdm(loader, desc="Train", leave=False):
        grid = grid.to(device)
        robot = robot.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(grid, robot)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = grid.size(0)
        epoch_loss += loss.item() * batch_size
        num_samples += batch_size

    return epoch_loss / max(1, num_samples)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run a single validation epoch.

    The model is evaluated without gradient computation. Both the validation
    loss and Dice score are averaged over the entire dataset and returned.
    """
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for (grid, robot), targets in tqdm(loader, desc="Val", leave=False):
            grid = grid.to(device)
            robot = robot.to(device)
            targets = targets.to(device)

            logits = model(grid, robot)
            loss = loss_fn(logits, targets)
            dice = dice_score(logits, targets)

            batch_size = grid.size(0)
            val_loss += loss.item() * batch_size
            val_dice += dice.item() * batch_size
            num_samples += batch_size

    avg_loss = val_loss / max(1, num_samples)
    avg_dice = val_dice / max(1, num_samples)
    return avg_loss, avg_dice

