import torch
from portable_bridge_unet.losses import build_loss


def test_losses_are_finite_and_backward():
    for name in ("bce", "dice", "bce_dice", "focal"):
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = (torch.rand(2, 1, 32, 32) > 0.7).float()
        loss = build_loss(name)(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
