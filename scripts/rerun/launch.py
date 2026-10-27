"""Run a frozen dependency queue with measured per-GPU capacity and owned subprocesses."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate(manifest, root):
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.train import read_config

    source = Path(manifest['source_root'])
    if source.resolve() != ROOT.resolve() or software_contract() != manifest['software_contract']:
        raise ValueError('Frozen training source or environment changed')
    for relative, digest in manifest['runtime_hashes'].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen queue runtime changed: {relative}')
    identities, outputs = set(), set()
    records = {task['id']: task for task in manifest['tasks']}
    if len(records) != 81 or len(manifest['tasks']) != 81:
        raise ValueError('Expected exactly 81 paper tasks')
    for task in manifest['tasks']:
        baseline = read_config(task['config'])
        for placement in task.get('placements', []):
            candidate = read_config(placement['config'])
            if ({k: v for k, v in candidate.items() if k not in {'device', 'budget'}} !=
                    {k: v for k, v in baseline.items() if k not in {'device', 'budget'}}):
                raise ValueError('GPU placement must preserve the complete scientific configuration')
    placements = manifest['tasks'] + [dict(task, **placement)
        for task in manifest['tasks'] for placement in task.get('placements', [])]
    for task in placements:
        path = Path(task['config'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['config_hashes'][str(path)]:
            raise ValueError(f'Generated configuration changed: {path.name}')
        config = read_config(path)
        if manifest.get('admission') == 'memory':
            key = config['stage'] + '/' + Path(config['dataset']).stem
            case = manifest['memory_profile']['cases'][key]
            signature = {k: config[k] for k in
                         ('depth', 'width', 'heads', 'microbatch', 'logical_batch', 'precision')}
            if (case['config'] != signature or task['reservation_mib'] != case['reservation_mib']
                    or task['reservation_mib'] <= 0
                    or manifest['memory_profile']['software_contract'] != manifest['software_contract']):
                raise ValueError('Task memory reservation does not match measured configuration')
        output = Path(task['output'])
        if (not path.resolve().is_relative_to(root / 'configs')
                or not output.resolve().is_relative_to(root / 'runs')
                or str(output) != config['output'] or task['gpu'] not in manifest['devices']
                or config['device'] != f'cuda:{task["gpu"]}'
                or config['concurrent_gpu_runs'] != manifest['slots_per_gpu']
                or config['budget'] != str(root / f'usage/gpu{task["gpu"]}')
                or config['resume'] is not None
                or config['wandb']['project'] != 'eegx'):
            raise ValueError('Task paths, device, concurrency or fresh-run identity are inconsistent')
        if any(d not in records for d in task['depends']):
            raise ValueError('Queue contains an unknown dependency')
        if task['stage'] == 'pretrain':
            if task['depends'] or config.get('checkpoint'):
                raise ValueError('Fresh pretraining cannot depend on an existing checkpoint')
        else:
            if len(task['depends']) != 1:
                raise ValueError('Downstream tasks require one selected pretraining checkpoint')
            parent = records[task['depends'][0]]
            expected = str(Path(parent['output']) / 'training/best.pth')
            if config['checkpoint'] != expected or parent['stage'] != 'pretrain':
                raise ValueError('Downstream checkpoint must belong to its declared dependency')
        identities.add(task['id'])
        outputs.add(task['output'])
    if len(identities) != 81 or len(outputs) != 81:
        raise ValueError('Task identities and output directories must be unique')
    reachable = set()
    while True:
        ready = {t['id'] for t in manifest['tasks'] if set(t['depends']) <= reachable}
        added = ready - reachable
        if not added:
            break
        reachable |= added
    if reachable != identities:
        raise ValueError('Queue dependencies contain a cycle')


def drain_training(manifest, task):
    """Retain the slot until the wrapper's recorded training child exits."""
    from scripts.paper.process import open_matching_process, signal_process

    identity = hashlib.sha256(str(Path(task['output']).resolve()).encode()).hexdigest()
    receipt = Path(manifest['claims']) / (identity + '.json')
    if not receipt.exists():
        # The child establishes parent-death protection before importing training.
        return
    owner = json.loads(receipt.read_text())
    if owner['output'] != task['output']:
        raise ValueError('Claim receipt does not match its queue task')
    handle = open_matching_process(owner['pid'], owner['process_start'])
    if handle is None:
        return
    try:
        signal_process(handle, signal.SIGTERM)
        select.select([handle], [], [])
    finally:
        os.close(handle)


def run(path, *, validate_only=False):
    from scripts.paper.process import open_process, signal_process

    path = Path(path).resolve()
    root = path.parent
    manifest = json.loads(path.read_text())
    validate(manifest, root)
    if validate_only:
        print(json.dumps({'status': 'validated', 'tasks': 81,
                          'slots_per_gpu': manifest['slots_per_gpu']}), flush=True)
        return 0
    from pretrain.gpu import gpu_snapshot
    from pretrain.metadata import atomic_json

    # The lock belongs to this launcher for its lifetime. A second launcher cannot own this queue.
    with (root / 'launch.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for gpu in manifest['devices']:
            state = gpu_snapshot(gpu)
            needed = manifest['slots_per_gpu'] * manifest['reservation_mib'] + manifest['headroom_mib']
            if manifest.get('admission') == 'memory':
                continue
            if state['other_compute_workload'] or state['free_mib'] < needed:
                raise RuntimeError(f'GPU {gpu} no longer has its tested capacity; preserve unrelated work')
        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        status = {t['id']: 'pending' for t in manifest['tasks']}
        state_path = root / 'queue-state.json'
        pinned = json.loads(state_path.read_text()).get('placements', {}) if state_path.exists() else {}
        live = {}
        started = time.time()
        (root / 'logs').mkdir(exist_ok=True)

        def save():
            atomic_json(root / 'queue-state.json', {
                'pid': os.getpid(), 'started_utc': started, 'updated_utc': time.time(),
                'stop_requested': stopped, 'placements': pinned, 'status': status, 'counts': dict(Counter(status.values())),
                'running': {t['id']: {'pid': p.pid, 'gpu': t['gpu']}
                            for p, t, log in live.values()},
            })

        save()
        try:
            while True:
                if stopped:
                    for fd in live:
                        signal_process(fd, signal.SIGTERM)
                else:
                    active = Counter(t['gpu'] for p, t, log in live.values())
                    memory = None
                    if manifest.get('admission') == 'memory':
                        from scripts.rerun.memory import available_mib
                        memory = available_mib(manifest, live)
                    # Shared source first, followed by the other pretraining dependencies.
                    ordered = sorted(manifest['tasks'], key=lambda t: (
                        t['stage'] != 'pretrain', t['domain'] != 'cross_domain', t['id']))
                    for task in ordered:
                        if status[task['id']] != 'pending':
                            continue
                        dependencies = [status[d] for d in task['depends']]
                        if any(s in {'failed', 'blocked'} for s in dependencies):
                            status[task['id']] = 'blocked'
                            continue
                        if any(s != 'complete' for s in dependencies):
                            continue
                        from scripts.rerun.memory import choose_placement
                        task = choose_placement(task, active, manifest['slots_per_gpu'], memory,
                                                pinned.get(task['id']))
                        if task is None:
                            continue
                        pinned[task['id']] = {'gpu': task['gpu'], 'config': task['config']}
                        # Commit placement before spawning. A restarted launcher
                        # must check completed outputs with the same YAML.
                        save()
                        log = (root / 'logs' / f'{task["id"]}.log').open('a')
                        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                               'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
                               'PYTHONDONTWRITEBYTECODE': '1'}
                        try:
                            process = subprocess.Popen([
                                manifest['python'], str(ROOT / 'scripts/paper/task.py'),
                                '--config', task['config'], '--manifest', str(path),
                                '--guardian', str(os.getpid())], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            try:
                                fd = open_process(process.pid)
                            except BaseException:
                                process.terminate()
                                process.wait()
                                raise
                        except BaseException:
                            log.close()
                            raise
                        live[fd] = process, task, log
                        status[task['id']] = 'running'
                        active[task['gpu']] += 1
                        if memory is not None:
                            memory[task['gpu']] -= task['reservation_mib']
                        print(f'START GPU {task["gpu"]} {task["id"]}', flush=True)
                    save()
                if not live:
                    if stopped or all(s != 'pending' for s in status.values()):
                        break
                    if manifest.get('admission') != 'memory':
                        raise RuntimeError('Queue has pending tasks with no runnable dependencies')
                    # External memory owners may finish later. Keep pending work
                    # queued without allocating or changing another process.
                    select.select([], [], [], 1)
                    continue
                for fd in select.select(list(live), [], [], 1)[0]:
                    process, task, log = live.pop(fd)
                    code = process.wait()
                    drain_training(manifest, task)
                    log.close()
                    os.close(fd)
                    status[task['id']] = 'complete' if code == 0 else 'failed'
                    print(f'{status[task["id"]].upper()} {task["id"]} exit={code}', flush=True)
                save()
        finally:
            for fd in live:
                signal_process(fd, signal.SIGTERM)
            for fd, (process, task, log) in list(live.items()):
                process.wait()
                drain_training(manifest, task)
                status[task['id']] = 'stopped'
                log.close()
                os.close(fd)
            live.clear()
            save()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        code = int(stopped or any(s != 'complete' for s in status.values()))
        atomic_json(root / 'finished.json', {'exit_code': code, 'counts': dict(Counter(status.values())),
                                           'stopped': stopped, 'updated_utc': time.time()})
        return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, validate_only=args.validate_only))
