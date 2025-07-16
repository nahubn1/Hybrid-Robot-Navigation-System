"""Custom composite loss for training path segmentation models.

This module implements a weighted combination of Soft Dice Loss
and Binary Focal Loss, designed to address severe class imbalance
between sparse path pixels and the background. The formulation is
inspired by the following papers:

- Dice loss: "V-Net: Fully Convolutional Neural Networks for Volumetric
  Medical Image Segmentation" (`https://arxiv.org/abs/1606.04797`).
- Focal loss: "Focal Loss for Dense Object Detection"
  (`https://arxiv.org/abs/1708.02002`).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Parameters
    ----------
    eps : float, optional
        Small constant added to numerator and denominator for numerical
        stability. Defaults to ``1e-6``.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss given logits and targets."""
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(1, probs.dim()))
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
        dice_score = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1 - dice_score.mean()


class BinaryFocalLoss(nn.Module):
    """Binary focal loss implemented on logits.

    Parameters
    ----------
    gamma : float, optional
        Focusing parameter controlling the down-weighting of easy examples.
        Defaults to ``2.0``.
    alpha : float, optional
        Balancing factor for the positive class. Defaults to ``0.25``.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss given logits and targets."""
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Weighted sum of :class:`SoftDiceLoss` and :class:`BinaryFocalLoss`."""

    def __init__(
        self,
        dice_weight: float = 0.6,
        focal_weight: float = 0.4,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = SoftDiceLoss()
        self.focal = BinaryFocalLoss(gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted composite loss."""
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
