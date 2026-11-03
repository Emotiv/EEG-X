"""Size comparisons preserve the study and dispatch only ready, memory-admitted tasks."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from tools.queue import prepare, run, capacity, process


def test_size_queue_preserves_complete_study_and_uses_only_its_own_checkpoints(tmp_path, monkeypatch):
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


def test_admission_credits_only_usage_observed_on_both_sides_of_free_memory_snapshot(monkeypatch):
    snapshots = iter([{('GPU-a', 10): 60}, {('GPU-a', 10): 40}])
    monkeypatch.setattr(capacity, 'process_memory', lambda: next(snapshots))
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: '0, GPU-a, 100\n')
    owner = Mock(pid=10)
    free = run.available({'devices': [0], 'headroom_mib': 10},
                         {'reservations': {'pretrain': 80}},
                         {1: (owner, {'stage': 'pretrain'}, 0, None)})
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
        tasks.append({'id': name, 'stage': name, 'domain': 'in_domain', 'depends': parent,
                      'output': settings['output'], 'configs': {'0': str(config)}})
    manifest = {'tasks': tasks, 'devices': [0], 'python': str(executable)}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(run, 'validate', lambda *a: (manifest, {'reservations': {'pretrain': 1, 'finetune': 1}}))
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
