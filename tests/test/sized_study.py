"""Size comparisons preserve the study and dispatch only ready, memory-admitted tasks."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from tools.queue import prepare, run, capacity, process


@pytest.fixture
def committed_source(monkeypatch):
    original = subprocess.check_output

    def git(command, **kwargs):
        if command[:3] == ['git', 'status', '--porcelain']:
            return ''
        if command[:3] == ['git', 'ls-files', '-z']:
            tracked = original(command, **kwargs).decode().split('\0')
            tracked.extend(str(p.relative_to(prepare.ROOT)) for p in
                           (prepare.ROOT / 'tools/queue').glob('*.py')
                           if str(p.relative_to(prepare.ROOT)) not in tracked)
            return '\0'.join(tracked).encode()
        return original(command, **kwargs)

    monkeypatch.setattr(prepare.subprocess, 'check_output', git)


def test_size_queue_preserves_complete_study_and_uses_only_its_own_checkpoints(tmp_path, committed_source):
    manifest = prepare.prepare(tmp_path / 'large')
    assert (Path(manifest['source']) / 'tools/queue/run.py').exists()
    assert len(manifest['tasks']) == 73
    parents = {t['id']: t for t in manifest['tasks']}
    for task in manifest['tasks']:
        left, right = [yaml.safe_load(Path(path).read_text()) for path in task['configs'].values()]
        assert left['size'] == right['size'] == 'd8w512'
        assert left['width'] == 512 and left['depth'] == 8 and left['heads'] == 8
        assert left['wandb']['group'] is None
        assert 'd8w512' in left['wandb']['name']
        assert left['microbatch'] == left['logical_batch'] == (256 if task['stage'] == 'pretrain' else 64)
        assert {k: v for k, v in left.items() if k not in {'device', 'budget'}} == {
            k: v for k, v in right.items() if k not in {'device', 'budget'}}
        if task['depends']:
            parent = parents[task['depends']]
            assert left['checkpoint'] == str(Path(parent['output']) / 'training/best.pth')
            assert left['width'] == yaml.safe_load(Path(parent['configs']['0']).read_text())['width']


@pytest.mark.parametrize('width,gpu', [(128, 0), (256, 1)])
def test_in_domain_subset_excludes_tuab_and_transfer_runs(tmp_path, committed_source, width, gpu):
    datasets = {'stew', 'dreamer-arousal', 'tuev-binary'}
    manifest = prepare.prepare(tmp_path / 'subset', width=width, datasets=datasets,
                               domains=['in_domain'], devices=[gpu])
    assert manifest['study']['source'] is None
    assert manifest['study']['cross_domain'] == []
    assert len(manifest['tasks']) == manifest['study']['expected_runs'] == 18
    assert manifest['devices'] == [gpu]
    parents = {t['id']: t for t in manifest['tasks']}
    identities = set()
    for task in manifest['tasks']:
        assert task['dataset'] in datasets and task['domain'] == 'in_domain'
        assert task['stage'] in {'pretrain', 'finetune'}
        assert set(task['configs']) == {str(gpu)}
        config = yaml.safe_load(Path(task['configs'][str(gpu)]).read_text())
        assert config['size'] == f'd8w{width}'
        assert config['device'] == f'cuda:{gpu}'
        assert config['wandb']['group'] is None
        assert config['seed'] in {241, 242, 243}
        identities.add((task['dataset'], task['stage'], config['seed']))
        if task['depends']:
            parent = parents[task['depends']]
            assert parent['dataset'] == task['dataset']
            assert parent['id'].endswith(f's{config["seed"]}')
            assert config['checkpoint'] == str(Path(parent['output']) / 'training/best.pth')
    assert identities == {(d, stage, seed) for d in datasets
                          for stage in ['pretrain', 'finetune'] for seed in [241, 242, 243]}


def test_cross_domain_subset_retains_shared_source_dependency():
    study = json.loads((prepare.ROOT / 'data/study.json').read_text())
    selected = prepare.select_study(study, ['bciciv-2a'], ['cross_domain'])
    assert selected['in_domain'] == []
    assert selected['cross_domain'] == ['bciciv-2a']
    assert selected['source'] == study['source']
    assert selected['expected_runs'] == 7


def test_invalid_subset_does_not_create_output(tmp_path):
    root = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='selection'):
        prepare.prepare(root, datasets=['unknown'])
    assert not root.exists()


def test_admission_credits_only_usage_observed_on_both_sides_of_free_memory_snapshot(monkeypatch):
    snapshots = iter([{('GPU-a', 10): 60}, {('GPU-a', 10): 40}])
    monkeypatch.setattr(capacity, 'process_memory', lambda: next(snapshots))
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: '0, GPU-a, 100\n')
    owner = Mock(pid=10)
    free = run.available({'devices': [0], 'headroom_mib': 10},
                         {'reservations': {'toy/pretrain': 80}},
                         {1: (owner, {'stage': 'pretrain', 'dataset': 'toy'}, 0, None)})
    assert free == {0: 50}


@pytest.mark.parametrize('fail', [False, True])
def test_queue_runs_dependencies_first_and_blocks_failed_descendants(tmp_path, monkeypatch, fail):
    executable = tmp_path / 'fake-python'
    executable.write_text(f'#!{sys.executable}\n' + '''
import json, pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[sys.argv.index('--child') + 1]).read_text())
root = pathlib.Path(config['output']); root.mkdir(parents=True)
if config.get('checkpoint'):
    assert pathlib.Path(config['checkpoint']).exists()
(root / 'training').mkdir()
if config.get('fail'): sys.exit(7)
(root / 'training/best.pth').write_text('checkpoint')
(root / 'training/result.json').write_text(json.dumps({'status':'complete'}))
''')
    executable.chmod(0o755)
    tasks = []
    for name, parent in [('pretrain', None), ('finetune', 'pretrain')]:
        config = tmp_path / f'{name}.yaml'
        settings = {'dataset': str(tmp_path / 'data.h5'), 'output': str(tmp_path / name),
                    'device': 'cpu', 'stage': 'pretrain', 'fail': fail and name == 'pretrain'}
        if parent:
            settings['checkpoint'] = str(tmp_path / parent / 'training/best.pth')
        config.write_text(yaml.safe_dump(settings))
        tasks.append({'id': name, 'stage': name, 'domain': 'in_domain', 'depends': parent, 'dataset': 'toy',
                      'output': settings['output'], 'configs': {'0': str(config)}})
    manifest = {'tasks': tasks, 'devices': [0], 'python': str(executable)}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(run, 'validate', lambda *a: (manifest,
                         {'reservations': {'toy/pretrain': 1, 'toy/finetune': 1}}))
    monkeypatch.setattr(run, 'available', lambda *a: {0: 1})
    monkeypatch.setattr('pretrain.train.read_config', lambda p: yaml.safe_load(Path(p).read_text()))
    assert run.run(path) == int(fail)
    state = json.loads((tmp_path / 'queue-state.json').read_text())
    assert state['status']['pretrain'] == ('failed' if fail else 'complete')
    assert state['status']['finetune'] == ('blocked' if fail else 'complete')


def test_process_setup_failure_drains_its_new_child(monkeypatch):
    owned = Mock(pid=123)
    monkeypatch.setattr(process.subprocess, 'Popen', lambda *a, **k: owned)
    monkeypatch.setattr(process, 'open_handle', Mock(side_effect=OSError('descriptor limit')))
    with pytest.raises(OSError, match='descriptor limit'):
        process.start(['training'])
    owned.terminate.assert_called_once()
    owned.wait.assert_called_once()


def test_impossible_reservations_fail_before_dispatch_instead_of_waiting_forever():
    profile = {'hardware': [{'gpu': '0', 'total_mib': '100'}],
               'reservations': {'pretrain': 95}}
    with pytest.raises(ValueError, match='cannot fit'):
        capacity.feasible(profile, {'devices': [0], 'headroom_mib': 10})


def test_inference_benchmark_uses_declared_compute_policy_and_reports_batch_throughput(monkeypatch):
    from contextlib import contextmanager, nullcontext
    import torch
    from tools.queue.benchmark import inference

    active = []
    config = {'precision': 'fp32', 'compute': {'matmul': 'tf32'}}

    @contextmanager
    def policy(settings, device):
        assert settings == config
        active.append(True)
        yield {'matmul': 'tf32'}
        active.pop()

    class Model:
        def eval(self):
            return self

        def __call__(self, eeg, ids):
            assert active and not torch.is_grad_enabled()

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 8.0

    monkeypatch.setattr('pretrain.compute.configure_compute', policy)
    monkeypatch.setattr('pretrain.compute.autocast_context', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    result = inference(Model(), torch.zeros(64, 14, 256), torch.zeros(64, 14),
                       config, warmup=2, iterations=3)
    assert not active
    assert result['compute'] == {'matmul': 'tf32'}
    assert result['batches']['1']['median_ms'] == 8
    assert result['batches']['64']['samples_per_second'] == 8000
