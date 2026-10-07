"""Save reviewable pre-training artifacts using the configured data and model."""
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
import platform
import cv2
import numpy as np
import torch
from .data import SegmentationDataset, pair_by_stem
from .models import build_model, count_trainable_parameters
from .utils import save_json

def make_preview(rgb, binary, overlay):
    """Frame each panel and keep titles outside the image pixels."""
    panels = []
    for title, content in (
        ("Image", rgb),
        ("Binary mask", np.repeat(binary[..., None], 3, axis=2)),
        ("Mask overlay", overlay),
    ):
        height, width = content.shape[:2]
        # A minimum panel width keeps labels legible for small smoke-test inputs.
        panel_width = max(width, 180) + 16
        panel = np.full((height + 52, panel_width, 3), 245, dtype=np.uint8)
        left = (panel_width - width) // 2
        panel[44:44 + height, left:left + width] = content
        cv2.rectangle(panel, (left - 1, 43), (left + width, 44 + height), (140, 140, 140), 1)
        cv2.rectangle(panel, (0, 0), (panel_width - 1, height + 51), (100, 100, 100), 1)
        text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(panel, title, ((panel_width - text_width) // 2, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        panels.append(panel)
    gap = np.full((panels[0].shape[0], 10, 3), 255, dtype=np.uint8)
    return np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)

def save_smoke_test(config, dataset_report, out_dir, num_samples=6):
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(out_dir) / stamp
    out.mkdir(parents=True)
    save_json(config, out / "resolved_config.json")
    report = {
        "created_utc": stamp, "dataset": dataset_report,
        "pretraining_ready": False, "errors": [], "samples": [],
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "opencv": cv2.__version__},
        "preprocessing": {
            "prepared_data": "PNG inputs; Sunnybrook preparation uses DICOM rescale then per-frame min-max uint8",
            "target": "LV cavity from inner contours when using prepare_sunnybrook.py",
            "image_size": config.get("image_size", 256),
            "channels": config.get("channels", 3),
            "image_resize": "bilinear", "mask_resize": "nearest",
            "image_scaling": "float32 / 255", "mask_binarization": "value / 255 > 0.5",
            "training_augmentation": config.get("train", {}).get("augment", False),
            "preview_augmentation": False,
        },
        "split": config.get("split", {}),
        "training": config.get("train", {}),
        "checkpoint": {"required_before_training": False, "loaded": False},
    }
    sample_tensor = None
    if dataset_report["training_ready"]:
        try:
            samples = pair_by_stem(sorted(glob(config["dataset"]["images"], recursive=True)),
                                   sorted(glob(config["dataset"]["masks"], recursive=True)))
            ds = SegmentationDataset(samples, config.get("image_size", 256), config.get("channels", 3))
            # Evenly spaced deterministic samples across the sorted dataset.
            indices = np.linspace(0, len(ds) - 1, min(num_samples, len(ds)), dtype=int)
            for number, index in enumerate(indices):
                item = ds[int(index)]
                tensor, mask = item["image"], item["mask"]
                if not torch.isfinite(tensor).all() or not torch.isfinite(mask).all():
                    raise ValueError(f"non-finite data: {item['image_path']}")
                rgb = tensor.numpy().transpose(1, 2, 0)
                if rgb.shape[2] == 1:
                    rgb = np.repeat(rgb, 3, axis=2)
                rgb = np.round(rgb * 255).astype(np.uint8)
                binary = (mask[0].numpy() * 255).astype(np.uint8)
                overlay = rgb.copy()
                selected = binary > 0
                overlay[selected] = (0.6 * rgb[selected] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
                preview = make_preview(rgb, binary, overlay)
                name = f"sample_{number:02d}.png"
                if not cv2.imwrite(str(out / name), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"could not save {name}")
                report["samples"].append({
                    "preview": name, "image": item["image_path"], "mask": item["mask_path"],
                    "image_shape": list(tensor.shape), "mask_shape": list(mask.shape),
                    "image_range": [float(tensor.min()), float(tensor.max())],
                    "mask_values": mask.unique().tolist(), "foreground_fraction": float(mask.mean()),
                })
                if sample_tensor is None:
                    sample_tensor = tensor.unsqueeze(0)
        except (ValueError, OSError, RuntimeError) as exc:
            report["errors"].append(f"sample check: {exc}")
    else:
        report["errors"].append("prepared dataset is unavailable or invalid")
    try:
        model = build_model(config.get("model", "portable_bridge"),
                            in_channels=config.get("channels", 3),
                            filters=config.get("filters", [32, 64, 128])).cpu().eval()
        (out / "model_architecture.txt").write_text(str(model) + "\n")
        report["model"] = {
            "name": config.get("model", "portable_bridge"),
            "trainable_parameters": count_trainable_parameters(model),
            "filters": config.get("filters", [32, 64, 128]),
            "device": "cpu", "weights": "random initialization; no training or checkpoint loading",
        }
        if sample_tensor is not None:
            with torch.inference_mode():
                output = model(sample_tensor)
            expected = (1, 1, config.get("image_size", 256), config.get("image_size", 256))
            if tuple(output.shape) != expected or not torch.isfinite(output).all():
                raise ValueError(f"invalid model output: {tuple(output.shape)}")
            report["model"].update(forward_pass=True, input_shape=list(sample_tensor.shape),
                                   output_shape=list(output.shape), finite_output=True)
        else:
            report["model"]["forward_pass"] = False
    except (ValueError, RuntimeError) as exc:
        report["errors"].append(f"model check: {exc}")
    report["pretraining_ready"] = bool(
        dataset_report["raw_available"] and dataset_report["training_ready"]
        and report["samples"] and report.get("model", {}).get("forward_pass")
        and not report["errors"])
    save_json(report, out / "report.json")
    (out / "README.md").write_text(
        "# Pre-training smoke test\n\n"
        f"Ready: {report['pretraining_ready']}\n\n"
        "Each sample preview shows the resized image, binary target mask, and red mask overlay "
        "from left to right. Previews disable random augmentation.\n\n"
        "See report.json for checks and preprocessing settings, resolved_config.json for paths "
        "and experiment settings, and model_architecture.txt for the configured model. "
        "The forward pass uses random weights and does not measure segmentation quality. "
        "This check does not verify GPU memory capacity or full training convergence.\n")
    return out, report
