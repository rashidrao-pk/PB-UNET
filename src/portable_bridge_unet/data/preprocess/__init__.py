"""Dataset-specific preparation and a common config-driven API."""
from importlib import import_module

PREPARERS = {
    "brats2020": ("brats", "prepare_brats"),
    "brats2021": ("brats", "prepare_brats"),
    "drive": ("drive", "prepare_drive"),
    "sunnybrook": ("sunnybrook", "prepare_sunnybrook"),
    "kvasir_seg": ("kvasir", "prepare_kvasir"),
    "montgomery": ("montgomery", "prepare_montgomery"),
    "isic2016": ("isic2016", "prepare_isic2016"),
}


def prepare_dataset(config, dry_run=False):
    """Prepare the dataset named by ``config['dataset']['name']``.

    Returns a report mapping; invalid sources raise exceptions. Dry runs do not
    write prepared files. Dataset-specific source/split policies are preserved.
    """
    name = config.get("dataset", {}).get("name", "").lower()
    if name not in PREPARERS:
        raise ValueError(f"No preparer for {name!r}; supported: {', '.join(PREPARERS)}")
    module, function = PREPARERS[name]
    return getattr(import_module(f"{__name__}.{module}"), function)(config, dry_run=dry_run)


__all__ = ["prepare_dataset"]
