"""Case-separated 2D binary experiments from labeled BraTS 2020/2021 NIfTI."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from .common import output_root, sha256, write_metadata, write_png


def source_cases(config):
    dataset = config['dataset']
    root = Path(dataset['raw_root'])
    modality = dataset.get('modality', 'flair')
    if modality not in ('flair', 't1', 't1ce', 't2'):
        raise ValueError('modality must be flair, t1, t1ce, or t2')
    cases = []
    for mask in sorted(root.rglob('*_seg.nii.gz')):
        identity = mask.name.removesuffix('_seg.nii.gz')
        image = mask.with_name(f'{identity}_{modality}.nii.gz')
        if not image.is_file():
            raise FileNotFoundError(image)
        cases.append((identity, image, mask))
    if not cases:
        raise ValueError(f'No labeled *_seg.nii.gz cases in {root}; extract the labeled training release')
    if len({case[0] for case in cases}) != len(cases):
        raise ValueError('Duplicate BraTS case IDs')
    expected = dataset.get('expected_cases')
    if expected is not None and len(cases) != expected:
        raise ValueError(f'expected {expected} cases, found {len(cases)}')
    return cases


def read_case(image_path, mask_path):
    import nibabel as nib
    image = nib.as_closest_canonical(nib.load(str(image_path)))
    mask = nib.as_closest_canonical(nib.load(str(mask_path)))
    data, labels = image.get_fdata(dtype=np.float32), mask.get_fdata(dtype=np.float32)
    if data.ndim != 3 or data.shape != labels.shape or not np.allclose(image.affine, mask.affine):
        raise ValueError(f'Image/mask shape or affine mismatch: {image_path}')
    if not np.isfinite(data).all() or not np.isfinite(labels).all():
        raise ValueError(f'Non-finite BraTS volume: {image_path}')
    if not set(np.unique(labels)).issubset({0, 1, 2, 4}):
        raise ValueError('Expected BraTS 2020/2021 labels 0,1,2,4; other releases need explicit mapping')
    if not np.any(data != 0):
        raise ValueError(f'Empty MRI volume: {image_path}')
    if np.diff(np.percentile(data[data != 0], [1, 99]))[0] <= 0:
        raise ValueError(f"Degenerate MRI intensity range: {image_path}")
    return data, labels, image.affine


def validate_source(config):
    cases = source_cases(config)
    for _, image, mask in tqdm(cases, desc='Validate BraTS', unit='case'):
        read_case(image, mask)
    return cases


def prepare_brats(config, dry_run=False):
    dataset = config['dataset']
    target = dataset.get('region', 'WT').upper()
    regions = {'WT': (1, 2, 4), 'TC': (1, 4), 'ET': (4,)}
    if target not in regions:
        raise ValueError('region must be WT, TC, or ET')
    cases = validate_source(config)
    split = config.get('split', {})
    val, test, seed = split.get('validation', .15), split.get('test', .15), split.get('seed', 42)
    if not (0 < val < 1 and 0 < test < 1 and val + test < 1):
        raise ValueError('validation/test fractions must be positive and sum to less than one')
    remaining, testing = train_test_split(cases, test_size=test, random_state=seed)
    training, validation = train_test_split(remaining, test_size=val / (1 - test), random_state=seed)
    assignments = {case[0]: name for name, subset in [('train', training), ('val', validation), ('test', testing)] for case in subset}
    report = dict(dataset_name=dataset['name'], cases=len(cases), case_counts={name: list(assignments.values()).count(name) for name in ('train', 'val', 'test')},
                  seed=seed, split_unit='case before slicing; internal split of labeled training release',
                  modality=dataset.get('modality', 'flair'), region=target, labels=list(regions[target]),
                  preparation='canonical RAS; axial slices with nonzero MRI; nonzero-volume 1st/99th percentile scaling to uint8; no resize',
                  official_challenge_test=False, background_slices='included when MRI slice is nonzero')
    if dry_run:
        return {**report, 'source_valid': True}
    out = output_root(config)
    rows, case_metadata = [], []
    for identity, image_path, mask_path in tqdm(cases, desc='Prepare BraTS', unit='case'):
        image, labels, affine = read_case(image_path, mask_path)
        low, high = np.percentile(image[image != 0], [1, 99])
        if high <= low:
            raise ValueError(f'Degenerate MRI intensity range: {identity}')
        scaled = np.rint(np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        scaled[image == 0] = 0
        image_hash, mask_hash = sha256(image_path), sha256(mask_path)
        case_metadata.append(dict(case_id=identity, source_image=str(image_path.resolve()), source_mask=str(mask_path.resolve()),
                                  image_sha256=image_hash, mask_sha256=mask_hash, canonical_affine=affine.tolist(),
                                  percentile_1=float(low), percentile_99=float(high), split=assignments[identity]))
        for z in range(image.shape[2]):
            if not np.any(image[:, :, z] != 0):
                continue
            stem = f'{identity}_z{z:03d}'
            ip, mp = out / 'images' / f'{stem}.png', out / 'masks' / f'{stem}.png'
            write_png(ip, scaled[:, :, z])
            write_png(mp, np.isin(labels[:, :, z], regions[target]).astype(np.uint8) * 255)
            rows.append(dict(image=str(ip), mask=str(mp), group=identity, split=assignments[identity], slice_index=z,
                             source_image=str(image_path.resolve()), source_mask=str(mask_path.resolve())))
    report.update(pairs=len(rows), **{name: sum(row['split'] == name for row in rows) for name in ('train', 'val', 'test')})
    report['cases_metadata'] = case_metadata
    return write_metadata(out, rows, report)
