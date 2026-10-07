"""CLI adapters for the data library; algorithms live in retrieve/preprocess."""
import argparse
import json
from zipfile import BadZipFile

from ..config import DEFAULT_CONFIG, load_config
from .preprocess import prepare_dataset
from .retrieve import download_dataset


def _execute(parser, args, operation, dataset=None):
    try:
        config = load_config(args.config)
        name = config.get("dataset", {}).get("name")
        if dataset is not None:
            if name is not None and name.lower() != dataset:
                raise ValueError(f"This entry point expects {dataset}, but config selects {name}")
            config.setdefault("dataset", {})["name"] = dataset
        function = download_dataset if operation == "retrieve" else prepare_dataset
        report = function(config, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, ImportError, BadZipFile) as exc:
        parser.exit(1, f"{operation.capitalize()} failed: {exc}\nCheck the selected config and dataset guide; see docs/LIBRARY.md.\n")


def dataset_main(operation, dataset, config_name, argv=None):
    """Keep the original dataset-specific script commands as thin adapters."""
    parser = argparse.ArgumentParser(description=f"{operation.capitalize()} {dataset} using the data library")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.parent / config_name))
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate sources/listings without writing dataset files")
    return _execute(parser, parser.parse_args(argv), operation, dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrieve or preprocess datasets using YAML configuration")
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("retrieve", "preprocess"):
        command = commands.add_parser(operation)
        command.add_argument("--config", required=True, help="Experiment YAML; dataset.name selects implementation")
        command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return _execute(parser, args, args.operation)
