import csv
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
import yaml
from portable_bridge_unet.experiments import read_benchmark, validate_splits, training_arguments
from portable_bridge_unet.experiment_suite import load_suite, run_suite, build_commands
from portable_bridge_unet.data.preprocess.cvc import prepare_cvc
from portable_bridge_unet.data.splits import prepare_grouped_splits
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('suffix',['','_epito'])
def test_all_dataset_configs_match_and_have_modes(suffix):
    _,entries=load_suite(ROOT/f'configs/benchmark_all{suffix}.yaml')
    assert {row['name'] for row in entries}=={'sunnybrook','busi','kvasir_seg','montgomery','isic2016','brats2020','brats2021','drive','cvc_clinicdb'}
    assert entries[-1]['role']=='external'
    for row in entries:
        if row['role']=='train':
            assert len(row['resolved_benchmark']['models'])==9
            assert row['resolved_multiseed']['seeds']==[42,123,2026]


def fixture_suite(tmp_path,models=None):
    prepared=tmp_path/'prepared'; (prepared/'images').mkdir(parents=True);(prepared/'masks').mkdir()
    splits={}
    for index,name in enumerate(('train','val','test')):
        image=prepared/'images'/f'{name}.png';mask=prepared/'masks'/f'{name}.png'
        cv2.imwrite(str(image),np.random.default_rng(index).integers(0,256,(32,32,3),dtype=np.uint8))
        target=np.zeros((32,32),np.uint8);target[8:24,8:24]=255;cv2.imwrite(str(mask),target)
        path=tmp_path/f'{name}.csv'
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['image','mask','group']);writer.writerow([image,mask,name])
        splits[name]=str(path)
    experiment=dict(model='unet',filters=[4,8,16],image_size=32,channels=3,
        dataset=dict(name='busi',images=str(prepared/'images/*.png'),masks=str(prepared/'masks/*.png')),
        train={f'{s}_split':path for s,path in splits.items()})
    (tmp_path/'experiment.yaml').write_text(yaml.safe_dump(experiment))
    benchmark=dict(models=models or ['unet','portable_bridge'],splits=splits,out_root=str(tmp_path/'benchmark'),
        bootstrap_resamples=20,seed=42,postprocess=False,
        training=dict(epochs=1,batch_size=1,channels=3,image_size=32,filters=[4,8,16],lr=.001,device='cpu',amp=False,num_workers=0))
    for name in ('benchmark','multiseed'):
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump({**benchmark,**({'seeds':[1,2]} if name=='multiseed' else {})}))
    suite=dict(out_root=str(tmp_path/'suite_runs'),datasets=[dict(name='busi',role='train',experiment='experiment.yaml',benchmark='benchmark.yaml',multiseed='multiseed.yaml')])
    path=tmp_path/'suite.yaml';path.write_text(yaml.safe_dump(suite))
    return path


def test_dry_run_does_not_write_or_read_dataset(tmp_path):
    suite=fixture_suite(tmp_path)
    (tmp_path/'train.csv').unlink()
    out=tmp_path/'out'
    report=run_suite(suite,dry_run=True,out_dir=out)
    assert report['status']=='planned' and not out.exists()
    assert '--out-root' in report['commands'][0]


def test_global_preflight_blocks_training_and_records_all_failures(tmp_path,monkeypatch):
    suite=fixture_suite(tmp_path)
    (tmp_path/'train.csv').unlink()
    import portable_bridge_unet.experiment_suite as module
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k: pytest.fail('Must not train before readiness'))
    report=run_suite(suite,out_dir=tmp_path/'out')
    assert report['status']=='blocked' and not report['datasets']['busi']['ready']
    assert 'train.csv' in report['datasets']['busi']['error']
    assert (tmp_path/'out/suite_report.json').is_file()


def test_check_only_saves_previews_without_training(tmp_path):
    suite=fixture_suite(tmp_path)
    report=run_suite(suite,out_dir=tmp_path/'out',check_only=True)
    assert report['status']=='ready'
    preview=Path(report['datasets']['busi']['smoke_artifacts'])/'sample_00.png'
    assert preview.is_file()
    assert not (tmp_path/'out/busi/benchmark').exists()


def test_split_validation_rejects_image_and_group_leakage(tmp_path):
    suite=fixture_suite(tmp_path)
    _,entries=load_suite(suite)
    cfg=entries[0]['resolved_benchmark']
    assert validate_splits(cfg,require_groups=True)==dict(train=1,val=1,test=1)
    (tmp_path/'val.csv').write_text((tmp_path/'train.csv').read_text())
    with pytest.raises(ValueError,match='overlap'): validate_splits(cfg)


def test_training_seed_augmentation_and_amp_are_explicit():
    cfg=dict(splits=dict(train='t',val='v',test='e'),seed=7,training=dict(augment=False,amp=False,deterministic=True))
    args=training_arguments(cfg,seed=123)
    assert args[args.index('--seed')+1]=='123'
    assert '--no-augment' in args and '--no-amp' in args and '--deterministic' in args


def test_external_multiseed_uses_planned_source_checkpoints(tmp_path):
    _,entries=load_suite(ROOT/'configs/benchmark_all_epito.yaml',['kvasir_seg','cvc_clinicdb'])
    commands=build_commands(entries,'multiseed',tmp_path)
    external=commands[-1]
    assert external[external.index('--source-root')+1]==str(tmp_path/'kvasir_seg/multiseed')
    assert external[external.index('--seeds')+1:]==['42','123','2026']


def test_grouped_split_artifacts_keep_patients_separate(tmp_path):
    suite=fixture_suite(tmp_path)
    prepared=tmp_path/'prepared'
    manifest=prepared/'manifest.csv'
    rows=[]
    for i in range(12):
        image=prepared/f'images/patient_{i}.png';mask=prepared/f'masks/patient_{i}.png'
        image.write_bytes((prepared/'images/train.png').read_bytes());mask.write_bytes((prepared/'masks/train.png').read_bytes())
        rows.append(dict(image=str(image),mask=str(mask),group=f'patient_{i}'))
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','mask','group']);writer.writeheader();writer.writerows(rows)
    cfg=dict(dataset=dict(prepared_root=str(prepared)),split=dict(validation=.2,test=.2,seed=42),
        train={f'{name}_split':str(prepared/f'metadata/{name}_split.csv') for name in ('train','val','test')})
    report=prepare_grouped_splits(cfg,True)
    assert not (prepared/'metadata').exists()
    groups=report['groups'];assert not (set(groups['train'])&set(groups['val']) or set(groups['train'])&set(groups['test']))
    prepare_grouped_splits(cfg);prepare_grouped_splits(cfg)
    cfg['split']['seed']=99
    with pytest.raises(ValueError,match='Existing split differs'): prepare_grouped_splits(cfg)


def test_cvc_external_preparation_validates_before_writes(tmp_path):
    raw=tmp_path/'raw';(raw/'Original').mkdir(parents=True);(raw/'Ground Truth').mkdir()
    cv2.imwrite(str(raw/'Original/1.png'),np.full((32,32,3),128,np.uint8))
    cv2.imwrite(str(raw/'Ground Truth/1.png'),np.full((32,32),255,np.uint8))
    cfg=dict(dataset=dict(raw_root=str(raw),prepared_root=str(tmp_path/'prepared'),expected_pairs=1))
    assert prepare_cvc(cfg,True)['source_valid']
    assert not (tmp_path/'prepared').exists()
    prepare_cvc(cfg)
    assert (tmp_path/'prepared/metadata/test_split.csv').is_file()
    assert not (tmp_path/'prepared/metadata/train_split.csv').exists()


def test_case_group_leakage_rejected_even_with_distinct_images(tmp_path):
    suite=fixture_suite(tmp_path)
    _,entries=load_suite(suite)
    path=tmp_path/'val.csv'
    with path.open() as stream: rows=list(csv.DictReader(stream))
    rows[0]['group']='train'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image','mask','group']);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='group overlap'):
        validate_splits(entries[0]['resolved_benchmark'],require_groups=True)


def test_frozen_suite_config_cannot_silently_change(tmp_path):
    suite=fixture_suite(tmp_path)
    run_suite(suite,out_dir=tmp_path/'run',check_only=True)
    path=tmp_path/'benchmark.yaml'
    cfg=yaml.safe_load(path.read_text());cfg['training']['augment']=True
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Frozen config differs'):
        run_suite(suite,out_dir=tmp_path/'run',check_only=True)


def test_reevaluation_requires_original_suite_mode(tmp_path):
    suite=fixture_suite(tmp_path)
    run_suite(suite,out_dir=tmp_path/'run',check_only=True)
    with pytest.raises(ValueError,match='Existing suite mode differs'):
        run_suite(suite,out_dir=tmp_path/'run',mode='multiseed',skip_training=True)
