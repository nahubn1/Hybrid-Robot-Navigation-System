from pathlib import Path
import sys

import torch

SRC_PATH = Path(__file__).resolve().parents[3] / "src"
sys.path.append(str(SRC_PATH))

from dnn_guidance.loss import DiceFocalLoss, SoftDiceLoss
from dnn_guidance.trainer import dice_score


def test_dice_focal_loss_computation():
    loss_fn = DiceFocalLoss()
    B, C, H, W = 1, 1, 16, 16
    targets = torch.randint(0, 2, (B, C, H, W)).float()
    logits_perfect = torch.logit(targets.clamp(1e-4, 1 - 1e-4))
    loss_perfect = loss_fn(logits_perfect, targets)
    assert loss_perfect.ndim == 0
    assert torch.isclose(loss_perfect, torch.tensor(0.0), atol=1e-4)

    logits_wrong = torch.logit((1 - targets).clamp(1e-4, 1 - 1e-4))
    loss_wrong = loss_fn(logits_wrong, targets)
    assert loss_wrong.ndim == 0
    assert loss_wrong.item() > 0.0


def test_dice_score_matches_soft_dice_loss():
    B, C, H, W = 2, 1, 8, 8
    logits = torch.randn(B, C, H, W)
    targets = torch.rand(B, C, H, W)

    loss_fn = SoftDiceLoss()
    expected = 1 - loss_fn(logits, targets)
    result = dice_score(logits, targets)

    assert torch.isclose(result, expected, atol=1e-6)
