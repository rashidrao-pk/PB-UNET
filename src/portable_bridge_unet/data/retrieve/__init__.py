"""Dataset retrieval from official sources, separate from preprocessing."""
from importlib import import_module

RETRIEVERS = {"isic2016": "download_isic2016", "montgomery": "download_montgomery"}


def download_dataset(config, dry_run=False):
    """Retrieve configured raw data and return a report.

    Dry runs query official listings but do not download archives or write files.
    Datasets with manual downloads fail explicitly rather than using a mirror.
    """
    name = config.get("dataset", {}).get("name", "").lower()
    if name not in RETRIEVERS:
        raise ValueError(f"No automatic downloader for {name!r}; supported: {', '.join(RETRIEVERS)}. "
                         "Follow the dataset's manual download guide.")
    return getattr(import_module(f"{__name__}.{name}"), RETRIEVERS[name])(config, dry_run=dry_run)


__all__ = ["download_dataset"]
