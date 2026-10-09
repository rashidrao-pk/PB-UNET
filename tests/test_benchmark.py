import importlib.util
import json
import sys
from pathlib import Path
import yaml


def benchmark_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_benchmark.py'
    spec = importlib.util.spec_from_file_location('benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_training_only_forwards_settings(tmp_path, monkeypatch):
    config = tmp_path / 'benchmark.yaml'
    config.write_text(yaml.safe_dump(dict(models=['unet'], seed=19,
        splits=dict(train='train.csv', val='val.csv', test='test.csv'),
        out_root='runs', training=dict(augment=False, num_workers=0))))
    module = benchmark_module()
    commands = []
    monkeypatch.setattr(module, 'run', lambda cmd, dry_run=False: commands.append(cmd))
    monkeypatch.setattr(module, 'validate_splits', lambda *a, **k: {})
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--config', str(config), '--skip-evaluation'])
    module.main()
    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[cmd.index('--seed') + 1] == '19'
    assert '--no-augment' in cmd
    assert cmd[cmd.index('--train-split') + 1] == str(tmp_path / 'train.csv')
    assert not (tmp_path / 'runs/leaderboard_raw.csv').exists()


def test_benchmark_statistics_use_paired_csvs(tmp_path, monkeypatch):
    import pandas as pd
    config = tmp_path / 'benchmark.yaml'
    config.write_text(yaml.safe_dump(dict(models=['unet', 'portable_bridge'],
        splits=dict(train='train.csv', val='val.csv', test='test.csv'),
        out_root='runs', bootstrap_resamples=20)))
    metrics = ['dice', 'iou', 'precision', 'recall', 'accuracy', 'hd95']
    for model in ['unet', 'portable_bridge']:
        out = tmp_path / 'runs' / model / 'evaluation'
        out.mkdir(parents=True)
        (out / 'summary_raw.json').write_text(json.dumps({m: dict(mean=.5, median=.5) for m in metrics}))
        frame = pd.DataFrame({'image_path': ['b', 'a'], **{m: [.4, .6] for m in metrics}})
        if model != 'unet':
            frame = frame.iloc[::-1]
        frame.to_csv(out / 'per_image_raw.csv', index=False)
    module = benchmark_module()
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--config', str(config), '--skip-training', '--skip-evaluation'])
    module.main()
    assert (tmp_path / 'runs/leaderboard_raw.csv').exists()
    report = json.loads((tmp_path / 'runs/dice_bootstrap_vs_unet.json').read_text())
    assert report['portable_bridge']['mean_delta'] == 0
