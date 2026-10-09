"""DRIVE RGB vessel masks, official test split and first manual annotator."""
from pathlib import Path
import re
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from .common import output_root, sha256, write_metadata, write_png


def source_pairs(config):
    root = Path(config['dataset']['raw_root'])
    parts = {}
    for part in ('training', 'test'):
        images = sorted((root / part / 'images').glob('*.tif'))
        expected = config['dataset'].get(f'expected_{part}_pairs', 20)
        if not images or (expected is not None and len(images) != expected):
            raise ValueError(f'{part}: expected {expected} DRIVE TIFF images, found {len(images)}')
        rows = []
        for image in images:
            match = re.fullmatch(r'(\d+)_(training|test)\.tif', image.name)
            if not match or match[2] != part:
                raise ValueError(f'Unexpected DRIVE filename: {image}')
            identity = match[1]
            mask = root / part / '1st_manual' / f'{identity}_manual1.gif'
            roi = root / part / 'mask' / f'{identity}_{part}_mask.gif'
            for path in (mask, roi):
                if not path.is_file():
                    raise FileNotFoundError(path)
            rows.append((identity, image, mask, roi))
        parts[part] = rows
    if {r[0] for r in parts['training']} & {r[0] for r in parts['test']}:
        raise ValueError('DRIVE training/test image IDs overlap')
    return parts


def read_pair(row):
    _, image, mask, roi = row
    with Image.open(image) as source:
        rgb = np.asarray(source.convert('RGB'))
    arrays = []
    for path in (mask, roi):
        with Image.open(path) as source:
            array = np.asarray(source.convert('L'))
        if array.shape != rgb.shape[:2] or not set(np.unique(array)).issubset({0, 255}) or not array.any():
            raise ValueError(f'Expected nonempty binary mask matching image dimensions: {path}')
        arrays.append(array)
    # The FOV defines the evaluation domain; vessels outside it are excluded.
    return rgb, np.where(arrays[1] > 0, arrays[0], 0).astype(np.uint8), arrays[1]


def validate_source(config):
    parts = source_pairs(config)
    hashes = {}
    for part, rows in parts.items():
        hashes[part] = set()
        for row in rows:
            read_pair(row)
            hashes[part].add(sha256(row[1]))
    if hashes['training'] & hashes['test']:
        raise ValueError('Byte-identical DRIVE training/test images')
    return parts


def prepare_drive(config, dry_run=False):
    source = validate_source(config)
    split = config.get('split', {})
    seed, fraction = split.get('seed', 42), split.get('validation', .2)
    if not 0 < fraction < 1:
        raise ValueError('validation must be between zero and one')
    training, validation = train_test_split(source['training'], test_size=fraction, random_state=seed)
    parts = dict(train=training, val=validation, test=source['test'])
    report = dict(dataset_name='drive', pairs=sum(map(len, parts.values())), train=len(training), validation=len(validation),
                  test=len(parts['test']), seed=seed, official_test_preserved=True, annotator='1st_manual',
                  split_unit='image; patient linkage unavailable', evaluation_domain='provided retinal field of view',
                  preparation='RGB retained as PNG; binary first manual mask intersected with FOV; no resize or enhancement')
    if dry_run:
        return {**report, 'source_valid': True}
    out = output_root(config)
    (out / 'fov').mkdir()
    rows = []
    for split_name, subset in parts.items():
        for identity, image, mask, roi in subset:
            rgb, binary, fov = read_pair((identity, image, mask, roi))
            stem = f'DRIVE_{identity}'
            ip, mp, rp = [out / folder / f'{stem}.png' for folder in ('images', 'masks', 'fov')]
            write_png(ip, rgb[:, :, ::-1])
            write_png(mp, binary)
            write_png(rp, fov)
            rows.append(dict(image=str(ip), mask=str(mp), roi=str(rp), group=identity, split=split_name,
                             source_image=str(image.resolve()), source_mask=str(mask.resolve()), source_roi=str(roi.resolve()),
                             source_image_sha256=sha256(image), source_mask_sha256=sha256(mask), source_roi_sha256=sha256(roi)))
    return write_metadata(out, rows, report)
