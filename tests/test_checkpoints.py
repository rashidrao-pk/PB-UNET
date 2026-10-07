import pytest
import torch
from portable_bridge_unet.models import build_model
from portable_bridge_unet.checkpoints import load_checkpoint

@pytest.mark.parametrize("name", ["unet", "portable_bridge"])
def test_training_step_and_checkpoint_roundtrip(tmp_path, name):
    torch.manual_seed(42)
    model = build_model(name, in_channels=1, filters=[4, 8, 16])
    x = torch.randn(2, 1, 32, 32)
    target = torch.randint(0, 2, x.shape).float()
    optimizer = torch.optim.Adam(model.parameters())
    before = next(model.parameters()).detach().clone()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))
    model.eval()
    path = tmp_path / "best.pt"
    torch.save(dict(model_name=name, model_state=model.state_dict(),
                    in_channels=1, filters=[4, 8, 16], image_size=32, threshold=0.4), path)
    restored, metadata = load_checkpoint(path)
    assert metadata["threshold"] == 0.4
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))

def test_invalid_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({}, path)
    with pytest.raises(ValueError, match="missing fields"):
        load_checkpoint(path)
