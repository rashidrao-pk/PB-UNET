import torch

from portable_bridge_unet.models import (
    MODEL_NAMES,
    BaselineUNet,
    PortableBridgeLegacyUNet,
    build_model,
    count_trainable_parameters,
)


def test_legacy_parameter_counts():
    assert count_trainable_parameters(BaselineUNet()) == 1_211_649
    assert count_trainable_parameters(PortableBridgeLegacyUNet()) == 2_356_609


def test_all_model_output_shapes():
    x = torch.randn(1, 3, 64, 64)
    for name in MODEL_NAMES:
        model = build_model(name)
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 64, 64), name
