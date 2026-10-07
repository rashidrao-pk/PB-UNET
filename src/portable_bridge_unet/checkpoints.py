"""Load saved architectures, including unversioned historical PB checkpoints."""
import warnings
import torch
from .models import build_model, PortableBridgeUNet

def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    missing = {"model_name", "model_state"} - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    name = checkpoint["model_name"]
    settings = dict(in_channels=int(checkpoint.get("in_channels", 3)),
                    filters=checkpoint.get("filters", [32, 64, 128]))
    historical_pb = name.lower().replace("-", "_") in (
        "portable_bridge", "pb_unet", "portable_bridge_unet")
    if historical_pb and checkpoint.get("format_version", 1) < 2:
        if any(key.startswith("final_refine.") for key in checkpoint["model_state"]):
            warnings.warn("Unversioned refined PB checkpoint: assuming historical mixed interpolation "
                          "(bilinear bridge/final, nearest decoder). Retrain if provenance differs.", UserWarning)
            model = PortableBridgeUNet(**settings, decoder_mode="nearest", spatial_encoder_dropout=False)
            resolved = "historical_portable_bridge_mixed"
        else:
            model = build_model("portable_bridge_legacy", **settings)
            resolved = "portable_bridge_legacy"
    else:
        model = build_model(name, **settings)
        resolved = name
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), {**checkpoint, "resolved_model_name": resolved}
