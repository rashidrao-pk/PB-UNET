"""Plan, preflight and execute configured datasets without mixing test protocols."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import yaml
from tqdm.auto import tqdm
from .config import load_config
from .experiments import read_benchmark, validate_splits
from .data import load_samples_csv, SegmentationDataset
from .data.checks import check_dataset
from .data.preprocess import prepare_dataset
from .data.splits import prepare_grouped_splits
from .data.visualization import save_smoke_test
from .utils import save_json
from .models import MODEL_NAMES
from .data.preprocess.common import sha256

ROOT=Path(__file__).resolve().parents[2]


def load_suite(path, selected=None):
    path=Path(path).expanduser().resolve()
    cfg=yaml.safe_load(path.read_text())
    if not isinstance(cfg,dict) or not isinstance(cfg.get('datasets'),list) or not cfg['datasets']:
        raise ValueError('Suite requires a nonempty datasets list')
    entries=[]
    for row in cfg['datasets']:
        row=dict(row)
        if row.get('role') not in ('train','external'): raise ValueError('Dataset role must be train or external')
        name=row['name']
        if Path(name).name!=name or name in ('.','..'): raise ValueError('Dataset name must be a simple directory name')
        for key in ('experiment','benchmark') + (('multiseed',) if row['role']=='train' else ()):
            p=Path(row[key]).expanduser();row[key]=str(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
            if not Path(row[key]).is_file(): raise FileNotFoundError(row[key])
        experiment=load_config(row['experiment'])
        if experiment['dataset']['name']!=name: raise ValueError(f'Dataset name mismatch: {name}')
        row['resolved_experiment']=experiment
        if row['role']=='train':
            for key in ('benchmark','multiseed'):
                benchmark=read_benchmark(row[key])
                for split in ('train','val','test'):
                    if Path(benchmark['splits'][split]).resolve()!=Path(experiment['train'][f'{split}_split']).resolve():
                        raise ValueError(f'{name}: experiment/{key} {split} paths differ')
                for setting,default in (('channels',3),('image_size',256)):
                    if benchmark.get('training',{}).get(setting,default)!=experiment.get(setting,default):
                        raise ValueError(f'{name}: experiment/{key} {setting} differs')
                row[f'resolved_{key}']=benchmark
        else:
            # Resolve the small external schema here; no dataset read in dry runs.
            external=yaml.safe_load(Path(row['benchmark']).read_text())
            models=external.get('models',[])
            if not models or len(set(models))!=len(models) or set(models)-set(MODEL_NAMES): raise ValueError('External config requires unique supported models')
            for key in ('split_csv','source_root','out_root'):
                p=Path(external[key]).expanduser()
                external[key]=str(p.resolve() if p.is_absolute() else (Path(row['benchmark']).parent/p).resolve())
            if Path(external['split_csv']).resolve()!=Path(experiment['evaluate']['split_csv']).resolve(): raise ValueError('External test CSV mismatch')
            row['resolved_benchmark']=external
        entries.append(row)
    names=[row['name'] for row in entries]
    if len(set(names))!=len(names): raise ValueError('Duplicate dataset names')
    if selected:
        missing=set(selected)-set(names)
        if missing: raise ValueError(f'Unknown datasets: {sorted(missing)}; available: {names}')
        entries=[row for row in entries if row['name'] in selected]
    # Train sources before external evaluation, regardless of selection ordering.
    entries.sort(key=lambda row:row['role']=='external')
    return cfg,entries


def build_commands(entries, mode, out):
    commands=[]
    roots={row['name']:out/row['name']/mode for row in entries if row['role']=='train'}
    for row in entries:
        if row['role']=='train':
            script='run_multiseed.py' if mode=='multiseed' else 'run_benchmark.py'
            commands.append([sys.executable,str(ROOT/'scripts'/script),'--config',row[mode], '--out-root',str(roots[row['name']])])
        else:
            command=[sys.executable,str(ROOT/'scripts/run_external_benchmark.py'),'--config',row['benchmark'],
                     '--out-root',str(out/row['name']/'external')]
            source=row.get('source_dataset')
            if source in roots:
                command.extend(['--source-root',str(roots[source])])
                if mode=='multiseed':
                    source_row=next(item for item in entries if item['name']==source)
                    command.extend(['--seeds',*map(str,source_row['resolved_multiseed'].get('seeds',[42,123,2026]))])
            commands.append(command)
    return commands


def preflight(row, out, prepare=False):
    experiment=row['resolved_experiment']
    dataset=experiment['dataset']
    if prepare:
        if row['role']=='train':
            paths=[Path(experiment['train'][f'{s}_split']) for s in ('train','val','test')]
        else: paths=[Path(experiment['evaluate']['split_csv'])]
        if not all(path.is_file() for path in paths):
            if row['name']=='busi': raise ValueError('BUSI requires existing prepared lesion pairs and split CSVs; see docs/DATASETS/BUSI.md')
            # Sunnybrook may have existing PNGs but still need grouped manifests.
            if row['name']!='sunnybrook' or not (Path(dataset['prepared_root'])/'manifest.csv').is_file():
                prepare_dataset(experiment)
            if row['name']=='sunnybrook': prepare_grouped_splits(experiment)
    if row['role']=='train':
        cfg=row['resolved_benchmark']
        counts=validate_splits(cfg,require_groups=cfg.get('split_unit')=='group')
        if row['name']=='drive':
            if any(not s.roi for path in cfg['splits'].values() for s in load_samples_csv(path)):
                raise ValueError('DRIVE split CSVs require roi field-of-view masks')
    else:
        samples=load_samples_csv(experiment['evaluate']['split_csv'])
        ds=SegmentationDataset(samples,experiment.get('image_size',256),experiment.get('channels',3))
        for index in range(len(ds)): ds[index]
        counts={'test':len(samples)}
    from glob import glob
    prepared={str(Path(path).resolve()) for path in glob(dataset['images'],recursive=True)}
    paths=row['resolved_benchmark']['splits'].values() if row['role']=='train' else [experiment['evaluate']['split_csv']]
    declared=[str(Path(sample.image).resolve()) for path in paths for sample in load_samples_csv(path)]
    if len(declared)!=len(set(declared)) or set(declared)!=prepared:
        raise ValueError('Split CSVs must cover the prepared image inventory exactly once')
    inventory=check_dataset(experiment)
    if not inventory['training_ready']: raise ValueError(inventory.get('preparation_issue','Prepared data not ready'))
    artifacts,smoke=save_smoke_test(experiment,inventory,out/row['name']/'smoke_test',2)
    if not smoke['pretraining_ready']: raise ValueError(f'Model/data smoke test failed: {smoke["errors"]}')
    return dict(ready=True,counts=counts,raw_available=inventory.get('raw_available'),smoke_artifacts=str(artifacts))


def run_suite(path, *, selected=None, mode='benchmark', out_dir=None, dry_run=False, check_only=False, prepare=False, skip_training=False):
    cfg,entries=load_suite(path,selected)
    if mode not in ('benchmark','multiseed'): raise ValueError('mode must be benchmark or multiseed')
    if skip_training and not out_dir: raise ValueError('--skip-training requires --out-dir pointing at the existing suite run')
    source=Path(path).resolve()
    base=Path(cfg.get('out_root','../runs/dataset_suite')).expanduser()
    if not base.is_absolute(): base=source.parent/base
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(out_dir).expanduser().resolve() if out_dir else base.resolve()/stamp
    commands=build_commands(entries,mode,out)
    if skip_training:
        for row,command in zip(entries,commands):
            if row['role']=='train': command.append('--skip-training')
    if dry_run:
        for row,command in zip(entries,commands):
            print(f'{row["name"]} ({row["role"]}): '+' '.join(command+['--dry-run']))
        return dict(status='planned',mode=mode,out_dir=str(out),datasets=[row['name'] for row in entries],commands=commands)
    if skip_training and (out/'suite_report.json').is_file():
        previous=json.loads((out/'suite_report.json').read_text())
        if previous.get('mode')!=mode:
            raise ValueError('Existing suite mode differs; select the original benchmark/multiseed mode')
    if not skip_training and not check_only and any(out.rglob('best_dice.pt')):
        raise ValueError('Existing suite checkpoints found; use a fresh --out-dir or --skip-training')
    out.mkdir(parents=True,exist_ok=True)
    snapshots=out/'configs';snapshots.mkdir(exist_ok=True)
    for row,command in zip(entries,commands):
        resolved=row['resolved_multiseed'] if mode=='multiseed' and row['role']=='train' else row['resolved_benchmark']
        target=snapshots/f'{row["name"]}_{mode}.yaml'
        content=yaml.safe_dump(resolved,sort_keys=False)
        if target.exists() and target.read_text()!=content:
            raise ValueError(f'Frozen config differs: {target}; use the original protocol or a fresh run directory')
        target.write_text(content)
        command[command.index('--config')+1]=str(target)
    report=dict(status='checking',mode=mode,out_dir=str(out),datasets={},commands=commands,
                config_path=str(source),config_sha256=sha256(source))
    for row in tqdm(entries,desc='Dataset preflight',unit='dataset'):
        try: report['datasets'][row['name']]=preflight(row,out,prepare)
        except (OSError,ValueError,KeyError,ImportError) as exc:
            report['datasets'][row['name']]=dict(ready=False,error=str(exc))
        save_json(report,out/'suite_report.json')
    if any(not result['ready'] for result in report['datasets'].values()):
        report['status']='blocked';save_json(report,out/'suite_report.json');return report
    if check_only:
        report['status']='ready';save_json(report,out/'suite_report.json');return report
    report['status']='running';save_json(report,out/'suite_report.json')
    for row,command in tqdm(list(zip(entries,commands)),desc='Dataset experiments',unit='dataset'):
        try:
            subprocess.run(command,cwd=ROOT,check=True)
            report['datasets'][row['name']]['experiment_status']='complete'
        except (OSError,subprocess.CalledProcessError) as exc:
            report['status']='failed';report['datasets'][row['name']]['experiment_error']=str(exc)
            save_json(report,out/'suite_report.json');return report
        save_json(report,out/'suite_report.json')
    report['status']='complete';save_json(report,out/'suite_report.json');return report
