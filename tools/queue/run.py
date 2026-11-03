"""Execute a frozen size queue with dependencies and measured GPU memory admission."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.queue.prepare import digest  # noqa: E402
from tools.queue.process import start, protect_parent  # noqa: E402


def validate(path):
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.train import read_config
    from tools.queue.capacity import hardware, feasible, memory_contract
    from pretrain.pipeline.dataset import DatasetReference, read_metadata

    root = Path(path).resolve().parent
    manifest = json.loads(Path(path).read_text())
    if Path(manifest['source']).resolve() != ROOT or software_contract() != manifest['software']:
        raise ValueError('Frozen training source or software changed')
    for relative, expected in manifest['runtime_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Frozen queue runtime changed')
    for config, expected in manifest['hashes'].items():
        if digest(config) != expected:
            raise ValueError('Generated configuration changed')
    capacity = json.loads((root / 'capacity.json').read_text())
    if (capacity['software'] != manifest['software'] or capacity['hardware'] != hardware()
            or digest(root / 'capacity.json') != manifest['capacity_sha256']):
        raise ValueError('Capacity evidence changed')
    feasible(capacity, manifest)
    tasks = {t['id']: t for t in manifest['tasks']}
    if len(tasks) != manifest['study']['expected_runs']:
        raise ValueError('Study matrix contains duplicate or missing tasks')
    metadata = {}
    for task in tasks.values():
        settings = [read_config(config) for config in task['configs'].values()]
        if len(settings) != len(manifest['devices']):
            raise ValueError('Each task needs one reviewed configuration per GPU')
        for gpu, config in zip(task['configs'], settings):
            if (config['size'] != manifest['size'] or config['resume'] is not None
                    or config['output'] != task['output']
                    or config['stage'] != task['stage'] or config['wandb']['domain'] != task['domain']
                    or config['device'] != f'cuda:{gpu}'
                    or config['budget'] != str(root / f'usage/gpu{gpu}')
                    or not Path(config['output']).is_relative_to(root / 'runs')
                    or config['wandb']['project'] != 'eegx' or config['wandb'].get('group') is not None):
                raise ValueError('Task size, output or tracking identity changed')
        def normalize(c):
            return {k: v for k, v in c.items() if k not in {'device', 'budget'}}
        if any(normalize(c) != normalize(settings[0]) for c in settings):
            raise ValueError('GPU configurations changed the scientific recipe')
        config = settings[0]
        key = (config['dataset'], config['split_seed'], config.get('pretraining_scope', 'subject_holdout'))
        if key not in metadata:
            metadata[key] = read_metadata(DatasetReference(*key))
        view = metadata[key]
        name = task['dataset'].replace('-', '_')
        release = manifest['data_release']
        if (view['release_id'] != release['datasets'][name]['release_id']
                or config['split_seed'] != release['protocol']['datasets'][name]['split_seed']):
            raise ValueError('Data release or fixed split changed')
        records = [r for r in capacity['records'] if r['stage'] == task['stage']
                   and memory_contract(r['config']) == memory_contract(config)]
        if (not records or view['window_samples'] > max(r['samples'] for r in records)
                or len(view['channels']) > max(r['channels'] for r in records)
                or view['window_samples'] * len(view['channels']) >
                max(r['samples'] * r['channels'] for r in records)):
            raise ValueError('Task geometry or recipe exceeds measured capacity coverage')
        parent = task['depends']
        if parent:
            if parent not in tasks or tasks[parent]['stage'] != 'pretrain':
                raise ValueError('Missing pretraining dependency')
            expected = str(Path(tasks[parent]['output']) / 'training/best.pth')
            if settings[0]['checkpoint'] != expected:
                raise ValueError('Downstream checkpoint belongs to another queue')
        elif task['stage'] != 'pretrain':
            raise ValueError('Downstream task requires a pretraining dependency')
    return manifest, capacity


def available(manifest, capacity, live):
    from tools.queue.capacity import process_memory

    before = process_memory()
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    states = {int(gpu): (uuid.strip(), int(free))
              for gpu, uuid, free in (row.split(',') for row in output.splitlines())}
    after = process_memory()
    result = {}
    for gpu in manifest['devices']:
        uuid, free = states[gpu]
        outstanding = sum(max(0, capacity['reservations'][task['stage']] -
            min(before.get((uuid, process.pid), 0), after.get((uuid, process.pid), 0)))
            for process, task, device, log in live.values() if device == gpu)
        result[gpu] = free - manifest['headroom_mib'] - outstanding
    return result


def complete(task):
    path = Path(task['output']) / 'training/result.json'
    return path.exists() and json.loads(path.read_text()).get('status') == 'complete'


def run(path):
    from pretrain.metadata import atomic_json
    from pretrain.train import read_config

    path = Path(path).resolve()
    root = path.parent
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, capacity = validate(path)
        status = {}
        for task in manifest['tasks']:
            if Path(task['output']).exists() and not complete(task):
                raise RuntimeError('Interrupted output exists; preserve checkpoints and prepare a resume')
            status[task['id']] = 'complete' if complete(task) else 'pending'
        live = {}
        stopped = False

        def stop(*_):
            nonlocal stopped
            stopped = True

        def save():
            atomic_json(root / 'queue-state.json', {
                'pid': os.getpid(), 'counts': dict(Counter(status.values())), 'status': status,
                'running': {task['id']: {'pid': process.pid, 'gpu': gpu}
                            for process, task, gpu, log in live.values()}})

        previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        (root / 'logs').mkdir(exist_ok=True)
        try:
            while True:
                if stopped:
                    for process, _, _, _ in live.values():
                        process.terminate()
                else:
                    free = available(manifest, capacity, live)
                    ordered = sorted(manifest['tasks'], key=lambda t: (
                        t['domain'] != 'cross_domain', t['stage'] != 'pretrain', t['id']))
                    for task in ordered:
                        if status[task['id']] != 'pending':
                            continue
                        parent = status.get(task['depends'])
                        if parent in {'failed', 'blocked'}:
                            status[task['id']] = 'blocked'
                            continue
                        if task['depends'] and parent != 'complete':
                            continue
                        reservation = capacity['reservations'][task['stage']]
                        choices = [gpu for gpu in manifest['devices'] if free[gpu] >= reservation]
                        if not choices:
                            continue
                        gpu = max(choices, key=free.get)
                        config = task['configs'][str(gpu)]
                        read_config(config)
                        log = (root / 'logs' / f'{task["id"]}.log').open('a')
                        try:
                            process, fd = start([
                                manifest['python'], str(Path(__file__).resolve()), '--child', config,
                                '--parent', str(os.getpid())], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '2',
                                    'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                                    'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                        except BaseException:
                            log.close()
                            raise
                        live[fd] = process, task, gpu, log
                        status[task['id']] = 'running'
                        free[gpu] -= reservation
                        print(f'START GPU {gpu} {task["id"]}', flush=True)
                save()
                if not live:
                    if stopped or all(s != 'pending' for s in status.values()):
                        break
                    select.select([], [], [], 1)
                    continue
                for fd in select.select(list(live), [], [], 1)[0]:
                    process, task, gpu, log = live.pop(fd)
                    code = process.wait()
                    status[task['id']] = 'complete' if code == 0 and complete(task) else 'failed'
                    print(f'{status[task["id"]].upper()} {task["id"]} exit={code}', flush=True)
                    log.close()
                    os.close(fd)
            save()
        finally:
            for process, _, _, _ in live.values():
                process.terminate()
            for fd, (process, task, gpu, log) in live.items():
                process.wait()
                status[task['id']] = 'stopped'
                log.close()
                os.close(fd)
            live.clear()
            save()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        failed = stopped or any(s != 'complete' for s in status.values())
        atomic_json(root / 'finished.json', {'exit_code': int(failed),
                                            'counts': dict(Counter(status.values()))})
        return int(failed)


def child(config, parent):
    protect_parent(parent)
    from pretrain.train import run_config

    run_config(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--child')
    parser.add_argument('--parent', type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(args.child, args.parent)
    elif args.validate_only:
        validate(args.manifest)
        print('Queue validated', flush=True)
    else:
        raise SystemExit(run(args.manifest))
