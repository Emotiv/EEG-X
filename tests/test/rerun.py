"""Fresh queue identity, dependency and immutable configuration checks."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.rerun.prepare import build_tasks
from scripts.rerun.launch import validate
from pretrain.pipeline.checkpoint import software_contract
from pretrain.train import read_config

ROOT = Path(__file__).resolve().parents[2]


def generated(tmp_path):
    root, data = tmp_path / 'rerun', tmp_path / 'data'
    (root / 'configs').mkdir(parents=True)
    data.mkdir()
    configs = {p.stem: read_config(p) for p in sorted((ROOT / 'configs').glob('*.yaml'))}
    for settings in configs.values():
        (data / Path(settings['dataset']).name).touch()
    manifest = build_tasks(configs, root, 7, data)
    manifest.update(source_root=str(ROOT), software_contract=software_contract(), runtime_hashes={},
                    devices=[0, 1], slots_per_gpu=7)
    return root, manifest


def test_fresh_queue_has_all_tasks_and_only_new_checkpoint_dependencies(tmp_path):
    root, manifest = generated(tmp_path)
    validate(manifest, root)
    tasks = manifest['tasks']
    assert Counter((t['domain'], t['stage']) for t in tasks) == {
        ('in_domain', 'pretrain'): 20, ('in_domain', 'finetune'): 20,
        ('cross_domain', 'pretrain'): 1, ('cross_domain', 'linear_probe'): 20,
        ('cross_domain', 'finetune'): 20,
    }
    assert Counter(t['gpu'] for t in tasks) == {0: 41, 1: 40}
    for task in tasks:
        config = read_config(task['config'])
        assert config['resume'] is None
        assert config['wandb']['group'] is None
        assert config['wandb']['project'] == 'eegx'
        assert config['wandb']['enabled']
        assert config['purpose'] == 'paper'
        if task['stage'] == 'pretrain':
            assert config['microbatch'] == 512 and config['logical_batch'] == 4096
        if task['domain'] == 'cross_domain' and task['stage'] != 'pretrain':
            assert task['depends'] == ['cross_domain--tuab--pretrain--s241']
        if task['domain'] == 'in_domain' and task['stage'] == 'finetune':
            assert task['depends'][0].endswith(f'--s{config["seed"]}')


def test_modified_generated_config_is_rejected_before_any_task_runs(tmp_path):
    root, manifest = generated(tmp_path)
    config = Path(manifest['tasks'][0]['config'])
    value = yaml.safe_load(config.read_text())
    value['checkpoint'] = '/old-study/best.pth'
    config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='configuration changed'):
        validate(manifest, root)


def test_unknown_dependency_is_rejected_before_any_task_runs(tmp_path):
    root, manifest = generated(tmp_path)
    manifest['tasks'][0]['depends'] = ['missing-task']
    with pytest.raises(ValueError, match='unknown dependency'):
        validate(manifest, root)


def test_slot_drain_waits_for_recorded_training_child(tmp_path):
    import concurrent.futures
    import hashlib
    import json
    import subprocess
    import sys
    from scripts.paper.process import process_start
    from scripts.rerun.launch import drain_training

    child = subprocess.Popen([sys.executable, '-u', '-c',
        "import signal,sys\n"
        "def stop(*_):\n print('TERM',flush=True); input(); sys.exit(2)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "print('READY',flush=True)\n"
        "signal.pause()\n"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        output = str(tmp_path / 'output')
        digest = hashlib.sha256(output.encode()).hexdigest()
        (tmp_path / (digest + '.json')).write_text(json.dumps({
            'output': output, 'pid': child.pid, 'process_start': process_start(child.pid)}))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            done = executor.submit(drain_training, {'claims': str(tmp_path)}, {'output': output})
            assert child.stdout.readline().strip() == 'TERM'
            assert not done.done()
            child.stdin.write('release\n')
            child.stdin.flush()
            done.result(timeout=10)
        assert child.wait(timeout=10) == 2
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdin.close()
        child.stdout.close()


def test_frozen_revision_uses_snapshot_receipt_and_rejects_changed_contract(tmp_path, monkeypatch):
    import json
    from pretrain.pipeline import checkpoint

    monkeypatch.setattr(checkpoint, '__file__', str(tmp_path / 'pretrain/pipeline/checkpoint.py'))
    monkeypatch.setattr(checkpoint, 'software_contract', lambda: {'source': 'frozen'})
    receipt = tmp_path / 'source-revision.json'
    receipt.write_text(json.dumps({'commit': 'snapshot-commit', 'dirty': False,
                                   'software_contract': {'source': 'frozen'}}))
    assert checkpoint.source_revision() == {'commit': 'snapshot-commit', 'dirty': False}
    monkeypatch.setattr(checkpoint, 'software_contract', lambda: {'source': 'changed'})
    with pytest.raises(ValueError, match='Frozen source revision'):
        checkpoint.source_revision()


def test_memory_admission_reserves_startup_and_future_peak():
    from scripts.rerun.memory import remaining_mib

    # The first worker has not allocated yet; the second can grow back to peak.
    assert remaining_mib(20000, 6000, [(4000, 0), (12000, 3000)]) == 1000
    # Foreign allocations reduce global free memory directly.
    assert remaining_mib(15000, 6000, [(4000, 0), (12000, 3000)]) == -4000
    # Workers above their reservation do not create extra capacity.
    assert remaining_mib(20000, 6000, [(4000, 5000)]) == 14000


def test_memory_claim_rejects_pid_reuse_and_accounts_for_device(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts.paper import process
    from scripts.rerun.memory import task_usage

    output = str(tmp_path / 'run')
    task, manifest = {'output': output}, {'claims': str(tmp_path)}
    digest = hashlib.sha256(output.encode()).hexdigest()
    claim = tmp_path / (digest + '.json')
    memory = {('gpu0', 123): 900, ('gpu1', 123): 300}
    assert task_usage(manifest, task, 'gpu0', memory) == 0
    claim.write_text(json.dumps({'output': output, 'pid': 123, 'process_start': 'old'}))
    monkeypatch.setattr(process, 'process_start', lambda pid: 'new')
    assert task_usage(manifest, task, 'gpu0', memory) == 0
    monkeypatch.setattr(process, 'process_start', lambda pid: 'old')
    assert task_usage(manifest, task, 'gpu0', memory) == 900
    assert task_usage(manifest, task, 'gpu1', memory) == 300


def test_memory_queue_rejects_changed_reservation(tmp_path):
    root, manifest = generated(tmp_path)
    cases = {}
    for task in manifest['tasks']:
        config = read_config(task['config'])
        key = config['stage'] + '/' + Path(config['dataset']).stem
        cases[key] = {'config': {k: config[k] for k in
            ('depth', 'width', 'heads', 'microbatch', 'logical_batch', 'precision')},
            'reservation_mib': 5000}
        task['reservation_mib'] = 5000
    manifest.update(admission='memory', memory_profile={
        'software_contract': manifest['software_contract'], 'cases': cases})
    validate(manifest, root)
    manifest['tasks'][0]['reservation_mib'] = 1
    with pytest.raises(ValueError, match='memory reservation'):
        validate(manifest, root)


def test_ready_job_can_use_other_gpu_when_preferred_device_is_full():
    from scripts.rerun.memory import choose_placement

    task = {'id': 'one', 'gpu': 0, 'config': 'gpu0.yaml', 'reservation_mib': 5000,
            'placements': [{'gpu': 1, 'config': 'gpu1.yaml'}]}
    chosen = choose_placement(task, Counter({0: 7}), 81, {0: 4000, 1: 20000})
    assert chosen['gpu'] == 1 and chosen['config'] == 'gpu1.yaml'
    assert task['gpu'] == 0  # The immutable manifest remains unchanged.
    assert choose_placement(task, Counter(), 81, {0: 4000, 1: 4000}) is None


def test_memory_snapshot_does_not_double_credit_allocator_release(tmp_path, monkeypatch):
    from scripts.rerun import memory
    from pretrain import gpu

    readings = iter([{('gpu0', 123): 9000}, {('gpu0', 123): 2000}])
    monkeypatch.setattr(memory, 'process_memory', lambda: next(readings))
    monkeypatch.setattr(gpu, 'gpu_snapshot', lambda index: {'uuid': 'gpu0', 'free_mib': 15000})
    monkeypatch.setattr(memory, 'task_usage', lambda manifest, task, uuid, values: values[uuid, 123])
    manifest = {'devices': [0], 'headroom_mib': 6000}
    live = {1: (None, {'gpu': 0, 'reservation_mib': 12000}, None)}
    assert memory.available_mib(manifest, live) == {0: -1000}


def test_restart_uses_recorded_placement_even_when_other_gpu_has_more_space():
    from scripts.rerun.memory import choose_placement

    task = {'id': 'one', 'gpu': 0, 'config': 'gpu0.yaml', 'reservation_mib': 5000,
            'placements': [{'gpu': 1, 'config': 'gpu1.yaml'}]}
    pinned = {'gpu': 1, 'config': 'gpu1.yaml'}
    assert choose_placement(task, Counter(), 81, {0: 40000, 1: 10000}, pinned)['gpu'] == 1
    assert choose_placement(task, Counter(), 81, {0: 40000, 1: 1000}, pinned) is None
    with pytest.raises(ValueError, match='not a reviewed configuration'):
        choose_placement(task, Counter(), 81, {0: 40000, 1: 10000},
                         {'gpu': 1, 'config': 'unreviewed.yaml'})
