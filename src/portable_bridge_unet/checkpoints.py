"""Load the architecture and weights saved by the trainer."""
import torch
from .models import build_model

def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    missing = {"model_name", "model_state"} - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    model = build_model(
        checkpoint["model_name"], in_channels=int(checkpoint.get("in_channels", 3)),
        filters=checkpoint.get("filters", [32, 64, 128]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
