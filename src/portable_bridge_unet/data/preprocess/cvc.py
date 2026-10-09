"""Prepare CVC-ClinicDB strictly as an external evaluation set."""
from pathlib import Path
import numpy as np
from PIL import Image
from .common import output_root, sha256, write_png
from ...utils import save_json
import csv


def source_pairs(config):
    root=Path(config['dataset']['raw_root'])
    for image_dir,mask_dir in [(root/'Original',root/'Ground Truth'),(root/'CVC-ClinicDB/Original',root/'CVC-ClinicDB/Ground Truth'),(root/'images',root/'masks')]:
        if image_dir.is_dir() and mask_dir.is_dir(): break
    else: raise FileNotFoundError(f'No CVC Original/Ground Truth or images/masks folders under {root}')
    def inventory(folder):
        files=[p for p in folder.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp')]
        result={p.stem:p for p in files}
        if len(files)!=len(result): raise ValueError('Duplicate CVC filename stems')
        return result
    images,masks=inventory(image_dir),inventory(mask_dir)
    if not images or images.keys()!=masks.keys(): raise ValueError('CVC image/mask stems differ or are empty')
    expected=config['dataset'].get('expected_pairs',612)
    if expected is not None and len(images)!=expected: raise ValueError(f'expected {expected} CVC pairs, found {len(images)}')
    return [(name,images[name],masks[name]) for name in sorted(images)]


def read_pair(row):
    _,ip,mp=row
    with Image.open(ip) as stream: image=np.asarray(stream.convert('RGB'))
    with Image.open(mp) as stream: mask=np.asarray(stream.convert('L'))
    if image.shape[:2]!=mask.shape: raise ValueError(f'CVC image/mask dimensions differ: {ip}')
    if not set(np.unique(mask)).issubset({0,255}): raise ValueError(f'CVC mask must be binary 0/255: {mp}')
    return image,mask


def prepare_cvc(config,dry_run=False):
    source=source_pairs(config)
    for row in source: read_pair(row)
    report=dict(pairs=len(source),test=len(source),dataset_name='cvc_clinicdb',split_unit='external image test only; patient/video identities unavailable',
                preparation='decoded RGB retained; binary mask retained; PNG export; no resize',target='polyp foreground')
    if dry_run: return {**report,'source_valid':True}
    out=output_root(config)
    rows=[]
    for row in source:
        name,source_image,source_mask=row
        image,mask=read_pair(row)
        ip,mp=out/'images'/f'{name}.png',out/'masks'/f'{name}.png'
        write_png(ip,image[:,:,::-1]);write_png(mp,mask)
        rows.append(dict(image=str(ip),mask=str(mp),sample_id=name,source_image=str(source_image.resolve()),source_mask=str(source_mask.resolve()),
                         source_image_sha256=sha256(source_image),source_mask_sha256=sha256(source_mask),split='test'))
    for filename in ('samples.csv','manifest.csv','test_split.csv'):
        with (out/'metadata'/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['prepared_root']=str(out)
    save_json(report,out/'metadata/preparation.json')
    return report
