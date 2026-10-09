import csv
import json
from pathlib import Path
import cv2
import nibabel as nib
import numpy as np
import pytest
import torch
from PIL import Image

from portable_bridge_unet.data import load_samples_csv, make_loader
from portable_bridge_unet.data.checks import check_dataset
from portable_bridge_unet.data.preprocess import prepare_dataset
from portable_bridge_unet.data.visualization import save_smoke_test
from portable_bridge_unet.evaluation import evaluate_loader
from portable_bridge_unet.metrics import batch_metrics_from_logits


def config_for(tmp_path, name):
    out = tmp_path / 'prepared'
    config = dict(model='unet', filters=[4, 8, 16], image_size=32, channels=3 if name == 'drive' else 1,
                  dataset=dict(name=name, raw_root=str(tmp_path/'raw'), prepared_root=str(out),
                               images=str(out/'images/*.png'), masks=str(out/'masks/*.png')),
                  train={f'{s}_split':str(out/f'metadata/{s}_split.csv') for s in ('train','val','test')},
                  split=dict(validation=.2, test=.2, seed=42))
    return config


def brats_fixture(tmp_path, name='brats2020'):
    config = config_for(tmp_path, name)
    config['dataset']['expected_cases'] = 10
    for i in range(10):
        identity=f'BraTS_{i:03d}'
        root=tmp_path/'raw'/identity
        root.mkdir(parents=True)
        data=np.arange(12*14*4, dtype=np.float32).reshape(12,14,4)+i
        data[:,:,0]=0
        labels=np.zeros(data.shape, np.uint8)
        labels[2:5,3:7,1]=1
        labels[5:8,3:7,1]=2
        labels[8:10,3:7,1]=4
        for suffix,array in [('flair',data),('seg',labels)]:
            nib.save(nib.Nifti1Image(array,np.eye(4)),root/f'{identity}_{suffix}.nii.gz')
    return config


def drive_fixture(tmp_path):
    config=config_for(tmp_path,'drive')
    config['dataset'].update(expected_training_pairs=10, expected_test_pairs=3, expected_pairs=13)
    for part, indices in [('training',range(10,20)),('test',range(1,4))]:
        for folder in ('images','1st_manual','mask'):
            (tmp_path/'raw'/part/folder).mkdir(parents=True)
        for i in indices:
            rng=np.random.default_rng(i)
            rgb=rng.integers(0,256,(32,40,3),dtype=np.uint8)
            mask=np.zeros((32,40),np.uint8); mask[8:24,19:21]=255
            roi=np.zeros_like(mask); roi[4:28,4:36]=255
            for folder,filename,array in [('images',f'{i:02d}_{part}.tif',rgb),('1st_manual',f'{i:02d}_manual1.gif',mask),('mask',f'{i:02d}_{part}_mask.gif',roi)]:
                Image.fromarray(array).save(tmp_path/'raw'/part/folder/filename)
    return config


@pytest.mark.parametrize('name',['brats2020','brats2021'])
def test_brats_case_split_labels_and_smoke(tmp_path,name):
    config=brats_fixture(tmp_path,name)
    dry=prepare_dataset(config,True)
    assert dry['case_counts']==dict(train=6,val=2,test=2)
    assert not (tmp_path/'prepared').exists()
    report=prepare_dataset(config)
    assert report['pairs']==30  # MRI-empty slice skipped; tumor-empty brain slices retained.
    samples=[load_samples_csv(config['train'][f'{s}_split']) for s in ('train','val','test')]
    groups=[{s.group for s in subset} for subset in samples]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    mask=cv2.imread(str(tmp_path/'prepared/masks/BraTS_000_z001.png'),0)
    assert (mask>0).sum()==(3+3+2)*4
    assert json.loads((tmp_path/'prepared/metadata/preparation.json').read_text())['region']=='WT'
    inventory=check_dataset(config)
    assert inventory['raw_available'] and inventory['training_ready']
    _, smoke=save_smoke_test(config,inventory,tmp_path/'smoke',2)
    assert smoke['pretraining_ready']
    assert smoke['samples'][0]['image_shape']==[1,32,32]
    with pytest.raises(ValueError,match='empty prepared_root'):
        prepare_dataset(config)


def test_brats_invalid_labels_fail_before_output(tmp_path):
    config=brats_fixture(tmp_path)
    mask=next((tmp_path/'raw').rglob('*_seg.nii.gz'))
    nib.save(nib.Nifti1Image(np.full((12,14,4),3,np.uint8),np.eye(4)),mask)
    with pytest.raises(ValueError,match='labels'):
        prepare_dataset(config)
    assert not (tmp_path/'prepared').exists()


def test_drive_official_test_fov_loader_and_smoke(tmp_path):
    config=drive_fixture(tmp_path)
    assert prepare_dataset(config,True)['test']==3
    assert not (tmp_path/'prepared').exists()
    report=prepare_dataset(config)
    assert (report['train'],report['validation'],report['test'])==(8,2,3)
    samples=load_samples_csv(config['train']['test_split'])
    assert {s.group for s in samples}=={'01','02','03'}
    assert all(s.roi for s in samples)
    inventory=check_dataset(config)
    assert inventory['training_ready'] and inventory['raw_available']
    _,smoke=save_smoke_test(config,inventory,tmp_path/'smoke',1)
    assert smoke['pretraining_ready']
    batch=next(iter(make_loader(samples,2,image_size=32,num_workers=0,pin_memory=False)))
    assert 0 < batch['roi'].sum() < batch['roi'].numel()


def test_metrics_exclude_outside_fov_including_accuracy():
    target=torch.tensor([[[[1.,0.],[0.,0.]]]])
    roi=torch.tensor([[[[1.,1.],[0.,0.]]]])
    logits=torch.tensor([[[[20.,-20.],[20.,20.]]]])
    masked=batch_metrics_from_logits(logits,target,roi=roi)
    assert all(float(value[0])==1 for value in masked.values())
    assert batch_metrics_from_logits(logits,target)['accuracy'][0]==.5


def test_evaluation_uses_fov(tmp_path):
    config=drive_fixture(tmp_path)
    prepare_dataset(config)
    samples=load_samples_csv(config['train']['test_split'])
    loader=make_loader(samples,2,image_size=32,num_workers=0,pin_memory=False)
    class AllForeground(torch.nn.Module):
        def forward(self,x):
            return torch.full((x.shape[0],1,32,32),20.)
    frame,summary=evaluate_loader(AllForeground(),loader,torch.device('cpu'),compute_hd95=True)
    batch=next(iter(loader))
    expected=float((batch['mask'][0]*batch['roi'][0]).sum()/batch['roi'][0].sum())
    assert frame.iloc[0]['precision']==pytest.approx(expected)
    assert len(frame)==3 and 'hd95' in summary


@pytest.mark.parametrize('region,pixels',[('WT',32),('TC',20),('ET',8)])
def test_brats_region_mapping(tmp_path,region,pixels):
    config=brats_fixture(tmp_path)
    config['dataset']['region']=region
    prepare_dataset(config)
    mask=cv2.imread(str(tmp_path/'prepared/masks/BraTS_000_z001.png'),0)
    assert (mask>0).sum()==pixels


def test_drive_missing_fov_fails_before_output(tmp_path):
    config=drive_fixture(tmp_path)
    next((tmp_path/'raw/training/mask').glob('*.gif')).unlink()
    with pytest.raises(FileNotFoundError):
        prepare_dataset(config)
    assert not (tmp_path/'prepared').exists()
