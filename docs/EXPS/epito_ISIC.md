```bash
python scripts/download_isic2016.py --config configs/isic2016_pb_epito.yaml
python scripts/prepare_isic2016.py --config configs/isic2016_pb_epito.yaml
python scripts/check_dataset.py --config configs/isic2016_pb_epito.yaml --require-prepared
python scripts/run_benchmark.py --config configs/benchmark_isic2016_epito.yaml
python scripts/plot_dataset_overview.py
```


```bash
python scripts/prepare_isic2016.py --config configs/isic2016_pb_epito.yaml --dry-run
python scripts/prepare_isic2016.py --config configs/isic2016_pb_epito.yaml
python scripts/check_dataset.py --config configs/isic2016_pb_epito.yaml --require-prepared
```