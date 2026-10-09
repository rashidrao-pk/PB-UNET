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


def test_dice_amp_sized_reductions_are_finite():
    from portable_bridge_unet.losses import DiceLoss
    logits = torch.zeros(2, 1, 256, 256, dtype=torch.float16, requires_grad=True)
    loss = DiceLoss()(logits, torch.ones_like(logits))
    assert torch.isfinite(loss)
    assert .32 < loss.item() < .34
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_invalid_loss_weights():
    import pytest
    from portable_bridge_unet.losses import BCEDiceLoss
    for weights in ((-1, 2), (0, 0), (float("nan"), 1)):
        with pytest.raises(ValueError):
            BCEDiceLoss(*weights)
