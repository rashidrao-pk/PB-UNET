"""Validation shared by individual benchmarks and dataset-suite orchestration."""
from pathlib import Path
import math
import yaml
from .data import load_samples_csv, SegmentationDataset
from .models import MODEL_NAMES


def read_benchmark(path):
    path=Path(path).expanduser().resolve()
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict):
        raise ValueError(f'Benchmark config must be a mapping: {path}')
    models=config.get('models',list(MODEL_NAMES))
    if not models or len(set(models))!=len(models) or set(models)-set(MODEL_NAMES):
        raise ValueError(f'Benchmark models must be unique supported names: {models}')
    config['models']=models
    if 'seeds' in config:
        seeds=config['seeds']
        if not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds) or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in seeds):
            raise ValueError('seeds must be unique nonnegative integers')
    def resolve(value):
        p=Path(value).expanduser()
        return str(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
    if set(config.get('splits',{}))!= {'train','val','test'}:
        raise ValueError('Benchmark requires train, val and test split paths')
    config['splits']={name:resolve(value) for name,value in config['splits'].items()}
    config['out_root']=resolve(config.get('out_root','../runs/benchmark'))
    t=config.get('training',{})
    if not isinstance(t,dict): raise ValueError('training must be a mapping')
    for key,default in (('epochs',100),('batch_size',8),('image_size',256)):
        value=t.get(key,default)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{key} must be a positive integer')
    if t.get('channels',3) not in (1,3): raise ValueError('channels must be 1 or 3')
    if not 0 <= t.get('threshold',.5) <= 1: raise ValueError('threshold must be within [0,1]')
    if not math.isfinite(t.get('lr',1e-4)) or t.get('lr',1e-4)<=0: raise ValueError('lr must be finite and positive')
    if not isinstance(t.get('num_workers',4),int) or t.get('num_workers',4)<0: raise ValueError('num_workers must be nonnegative')
    filters=t.get('filters',[32,64,128])
    if not filters or any(not isinstance(f,int) or f<1 for f in filters): raise ValueError('filters must be positive integers')
    if t.get('image_size',256) % (2**len(filters)): raise ValueError('image_size must be divisible by 2 ** encoder depth')
    if config.get('bootstrap_resamples',10000)<1: raise ValueError('bootstrap_resamples must be positive')
    from .losses import build_loss
    build_loss(t.get("loss","bce_dice"))
    return config


def validate_splits(config, *, decode=True, require_groups=False):
    parts={name:load_samples_csv(path) for name,path in config['splits'].items()}
    identities={name:[str(Path(s.image).resolve()) for s in subset] for name,subset in parts.items()}
    if any(len(paths)!=len(set(paths)) for paths in identities.values()):
        raise ValueError('Duplicate images within split CSVs')
    for a,b in (('train','val'),('train','test'),('val','test')):
        if set(identities[a]) & set(identities[b]): raise ValueError(f'{a}/{b} image overlap')
    groups={name:{s.group for s in subset if s.group} for name,subset in parts.items()}
    if require_groups:
        if any(not s.group for subset in parts.values() for s in subset): raise ValueError('Nonempty patient/case groups required')
        for a,b in (('train','val'),('train','test'),('val','test')):
            if groups[a]&groups[b]: raise ValueError(f'{a}/{b} patient/case group overlap')
    settings=config.get('training',{})
    if decode:
        for subset in parts.values():
            ds=SegmentationDataset(subset,settings.get('image_size',256),settings.get('channels',3))
            for i in range(len(ds)): ds[i]
    return {name:len(subset) for name,subset in parts.items()}


def training_arguments(config, seed=None):
    t=config.get('training',{})
    arguments=['--seed',str(seed if seed is not None else t.get('seed',config.get('seed',42)))]
    for name,path in config['splits'].items(): arguments.extend([f'--{name}-split',path])
    for option,key,default in [('epochs','epochs',100),('batch-size','batch_size',8),('image-size','image_size',256),
        ('channels','channels',3),('lr','lr',1e-4),('loss','loss','bce_dice'),('num-workers','num_workers',4),
        ('device','device','auto'),('patience','patience',30),('lr-patience','lr_patience',10),
        ('lr-factor','lr_factor',.05),('threshold','threshold',.5)]:
        arguments.extend([f'--{option}',str(t.get(key,default))])
    arguments.extend(['--filters',*map(str,t.get('filters',[32,64,128]))])
    arguments.append('--augment' if t.get('augment',False) else '--no-augment')
    arguments.append('--amp' if t.get('amp',True) else '--no-amp')
    if t.get('deterministic',False): arguments.append('--deterministic')
    return arguments
