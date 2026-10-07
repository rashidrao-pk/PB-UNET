# Data library structure and usage

Dataset algorithms now live inside the installable `portable_bridge_unet`
package. The `scripts/` directory contains CLI adapters and experiment
orchestration; downloading, preprocessing, checking, and figure generation can
also be called directly from Python.

```text
src/portable_bridge_unet/
  data/
    __init__.py          # Public loader API, compatible with previous imports
    loaders.py           # Samples, pairing, splits, Dataset, DataLoader
    cli.py               # Shared command-line adapters
    __main__.py          # python -m portable_bridge_unet.data
    retrieve/
      __init__.py        # download_dataset(config, dry_run=False)
      isic2016.py        # Official archive retrieval and extraction
      montgomery.py      # Official NLM image and mask retrieval
    preprocess/
      __init__.py        # prepare_dataset(config, dry_run=False)
      sunnybrook.py      # DICOM/contour preparation
      kvasir.py          # RGB polyp preparation
      montgomery.py      # Both-lung mask preparation
      isic2016.py        # RGB skin-lesion preparation and official test split
    checks.py            # Raw/prepared availability and decoding checks
    visualization.py     # Bordered/titled smoke previews and model reports
    overview.py          # Combined dataset figure and descriptions
```

`retrieve` is the directory name (standard spelling), rather than `retrive`.
Raw/prepared dataset files remain at the paths configured in YAML; this source
package is not a directory for storing downloaded medical images.

## Use from Python

Install the package using your existing PyTorch environment:

```bash
python -m pip install -e . --no-deps
```

Load a config once and use the library APIs:

```python
from portable_bridge_unet.config import load_config
from portable_bridge_unet.data.retrieve import download_dataset
from portable_bridge_unet.data.preprocess import prepare_dataset
from portable_bridge_unet.data.checks import check_dataset
from portable_bridge_unet.data.visualization import save_smoke_test

config = load_config("configs/isic2016_pb_epito.yaml")
download_report = download_dataset(config, dry_run=True)
# download_dataset(config)  # Download when ready; dry run only resolves links.
# prepare_dataset(config)   # After raw files are available.
inventory = check_dataset(config)
artifacts, report = save_smoke_test(
    config, inventory, config["smoke_test"]["out_dir"], num_samples=6
)
```

For a dataset-specific API:

```python
from portable_bridge_unet.data.preprocess.isic2016 import prepare_isic2016
from portable_bridge_unet.data.retrieve.isic2016 import download_isic2016
```

Library functions accept configuration mappings and return report mappings.
Retrieval and preparation raise exceptions for invalid/missing data; they do not
parse `sys.argv`, print JSON reports, or exit Python. Progress bars remain visible.
Checks and smoke generation retain their existing report behavior, including
failure artifacts. Dry-run retrieval may make network requests for source lists,
but does not download dataset archives or create dataset files. Preprocessing
dry runs validate local sources without writing prepared files.

The dispatchers choose an implementation using `dataset.name`. Automated
retrieval currently supports `isic2016` and `montgomery`; preprocessing supports
`sunnybrook`, `kvasir_seg`, `montgomery`, and `isic2016`. Unsupported choices raise
a clear error. BUSI's externally prepared data and existing manual download
instructions are retained; no new BUSI or Kvasir downloader is implied.

Generate the combined figure without importing a script:

```python
from portable_bridge_unet.data.overview import create_dataset_overview

outputs = create_dataset_overview(
    "runs/smoke_test", "runs/dataset_overview",
    datasets=["sunnybrook", "busi", "kvasir_seg", "montgomery", "isic2016"],
)
print(outputs["png"], outputs["pdf"])
```

## Unified CLI

After installing the package, both forms use the same implementation:

```bash
python -m portable_bridge_unet.data retrieve --config configs/isic2016_pb_epito.yaml --dry-run
python -m portable_bridge_unet.data preprocess --config configs/isic2016_pb_epito.yaml --dry-run

pb-data retrieve --config configs/isic2016_pb_epito.yaml --dry-run
pb-data preprocess --config configs/isic2016_pb_epito.yaml --dry-run
```

Remove `--dry-run` to perform the corresponding operation. The `pb-data` console
entry point is registered when installing/reinstalling this package version.
For module commands directly from a checkout without installation, prefix with
`PYTHONPATH=src` from the repo root. Explicit config paths keep the library usable
outside the repository working directory; relative dataset paths resolve against
the YAML file as before.

## Existing commands and imports

These commands still work and now delegate to the library:

```bash
python scripts/download_isic2016.py --config configs/isic2016_pb_epito.yaml --dry-run
python scripts/prepare_isic2016.py --config configs/isic2016_pb_epito.yaml --dry-run
python scripts/download_montgomery.py --config configs/montgomery_pb_epito.yaml --dry-run
python scripts/prepare_montgomery.py --config configs/montgomery_pb_epito.yaml --dry-run
python scripts/prepare_kvasir.py --config configs/kvasir_pb_epito.yaml --dry-run
python scripts/prepare_sunnybrook.py --config configs/paper_legacy_epito.yaml --dry-run
```

Dataset-specific commands reject a config selecting a different dataset, rather
than silently dispatching to a mismatched preparer. The existing `check_dataset`,
`preview_datasets`, and `plot_dataset_overview` commands also keep their flags.
Training, evaluation, prediction, and benchmark commands remain unchanged.

Existing imports such as `from portable_bridge_unet.data import make_loader`
remain supported. Flat modules `preparation`, `kvasir`, `montgomery`, `isic2016`,
`dataset_check`, and `smoke_test` are compatibility re-exports of the canonical
library functions. New code should use `portable_bridge_unet.data.*`. The
compatibility modules contain no duplicate algorithms.

## Adding a dataset

Add the preparation implementation under `data/preprocess/`, register it in
`PREPARERS`, and supply configs/tests/docs. If automatic retrieval is appropriate,
add it under `data/retrieve/` and register it in `RETRIEVERS`. Keep dataset/split
policies in the implementation and paths in YAML. The shared CLI then exposes
the new dataset without adding another algorithm-bearing script.

BraTS 2020/2021 and DRIVE preparers are also registered in `data.preprocess`. See [BraTS](BRATS.md) and [DRIVE](DRIVE.md) for access instructions, layouts and protocol details. Their authenticated downloads remain manual. `Sample.roi` and CSV `roi` fields optionally carry field-of-view masks through loaders and evaluation.
