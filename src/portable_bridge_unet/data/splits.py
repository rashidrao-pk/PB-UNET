"""Explicit patient-grouped split artifacts for prepared Sunnybrook data."""
import csv
from pathlib import Path
from ..utils import save_json
from .loaders import load_samples_csv, split_samples_grouped
from .preprocess.common import sha256


def prepare_grouped_splits(config, dry_run=False):
    root = Path(config['dataset']['prepared_root']).resolve()
    manifest = root / 'manifest.csv'
    samples = load_samples_csv(manifest)
    if len({str(Path(s.image).resolve()) for s in samples}) != len(samples):
        raise ValueError('Duplicate images in the prepared source manifest')
    settings = config.get('split', {})
    parts = split_samples_grouped(samples, settings.get('validation', .15),
                                  settings.get('test', .15), settings.get('seed', 42))
    report = dict(split_unit='patient group', seed=settings.get('seed', 42),
                  source_manifest=str(manifest), source_manifest_sha256=sha256(manifest),
                  counts={name:len(subset) for name, subset in zip(('train','val','test'), parts)},
                  groups={name:sorted({s.group for s in subset}) for name, subset in zip(('train','val','test'), parts)})
    if dry_run:
        return report
    meta = root / 'metadata'
    meta.mkdir(parents=True, exist_ok=True)
    targets = []
    for name, subset in zip(('train','val','test'), parts):
        target = Path(config['train'][f'{name}_split'])
        import io
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['image','mask','group'])
        writer.writerows((s.image,s.mask,s.group) for s in subset)
        content = stream.getvalue().encode()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Existing split differs: {target}; use a separate prepared_root/protocol')
        targets.append((target,content))
    for target,content in targets:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    save_json(report, meta / 'split_preparation.json')
    return report
