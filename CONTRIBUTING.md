# Contributing

Use an isolated branch and install the editable package with test dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python audit/verify_parameter_counts.py
```

Keep reusable algorithms under `src/portable_bridge_unet/`. Dataset preparation
belongs in `data/preprocess/`, retrieval in `data/retrieve/`; CLI adapters should
remain small. Existing train/benchmark orchestration is still in scripts.
Document public APIs and config changes, and preserve compatibility intentionally.

For behavior changes, add a focused regression test with temporary synthetic
fixtures. Tests must not download restricted data or depend on local datasets,
checkpoints, hard-coded author paths, or GPUs. Record how a fix changes old
results or checkpoints. Avoid tests that merely repeat implementation details.

New datasets need source/access/citation documentation, validation, reproducible
split policy, preprocessing report, configs, library registration, smoke previews,
and tests for malformed inputs and split isolation. Use patient/case grouping
where available. Do not describe image identity as verified patient identity.

New architectures need explicit names, a source citation, implementation
assumptions, shape/gradient tests, checkpoint round trips, and a model-card update.
Label adaptations honestly. Shared widths do not make baselines computationally
matched or faithful to their original papers.

Pull requests should state the problem, final behavior, validation, and any effect
on reproducibility. Do not include private data, credentials, raw medical images,
large run directories, or unsupported performance claims. Report the exact command,
config (with private paths redacted), versions, traceback and minimal fixture when
opening an issue. Review the dataset's terms before sharing samples or weights.
