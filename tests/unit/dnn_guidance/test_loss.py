from pathlib import Path
import sys

import torch

SRC_PATH = Path(__file__).resolve().parents[3] / "src"
sys.path.append(str(SRC_PATH))

from dnn_guidance.loss import DiceFocalLoss


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
