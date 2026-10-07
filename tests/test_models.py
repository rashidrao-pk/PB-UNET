import torch

from portable_bridge_unet.models import BaselineUNet, PortableBridgeUNet, count_trainable_parameters


def test_parameter_counts():
    assert count_trainable_parameters(BaselineUNet()) == 1_211_649
    assert count_trainable_parameters(PortableBridgeUNet()) == 2_356_609


def test_output_shapes():
    x = torch.randn(2, 3, 64, 64)
    for model in (BaselineUNet(), PortableBridgeUNet()):
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1, 64, 64)
