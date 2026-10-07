"""Shared YAML configuration; command-line arguments always take precedence.

Relative paths are resolved against the configuration file's directory.
"""
from pathlib import Path
import argparse
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "paper_legacy.yaml"
PATH_KEYS = {"images", "masks", "out_dir", "checkpoint", "split_csv", "image", "output"}

def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    config = dict(config)
    for section in ("dataset", "train", "evaluate", "predict", "smoke_test"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        values = dict(values)
        for key, value in values.items():
            if value is not None and (key in PATH_KEYS or key.endswith("_root")):
                p = Path(value).expanduser()
                values[key] = str(p if p.is_absolute() else path.parent / p)
        config[section] = values
    return config

def parse_config_args(parser, section, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=str(DEFAULT_CONFIG))
    selected, _ = probe.parse_known_args(argv)
    config = load_config(selected.config)
    parser.add_argument("--config", default=selected.config)
    defaults = {k: v for k, v in config.items() if not isinstance(v, dict)}
    if section != "train":
        # Architecture and threshold come from the checkpoint unless explicitly overridden.
        defaults = {k: v for k, v in defaults.items() if k in ("batch_size", "num_workers", "device")}
    defaults.update(config[section])
    if section == "train":
        defaults.update({k: v for k, v in config["dataset"].items() if k in ("images", "masks")})
        defaults["lr"] = config.get("learning_rate", 1e-4)
        defaults["val_fraction"] = config.get("split", {}).get("validation", 0.15)
        defaults["test_fraction"] = config.get("split", {}).get("test", 0.15)
        defaults["seed"] = config.get("split", {}).get("seed", 42)
    destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in defaults.items() if k in destinations})
    args = parser.parse_args(argv)
    for key in {"train": ("images", "masks"), "evaluate": ("checkpoint", "split_csv"),
                "predict": ("checkpoint", "image", "output")}[section]:
        if not getattr(args, key, None):
            parser.error(f"--{key.replace('_', '-')} must be provided in config or CLI")
    return args
