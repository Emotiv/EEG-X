"""Freeze reviewed source and prepare a fresh dependency queue without starting jobs."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_tasks(configs, root, slots, data_root, *, flexible=False):
    from pretrain.train import read_config

    manifest = {'tasks': [], 'config_hashes': {}}
    for name, template in configs.items():
        for seed in template.get('seeds', [template.get('seed')]):
            settings = copy.deepcopy(template)
            settings.pop('seeds', None)
            settings['seed'] = seed
            dataset = data_root / Path(template['dataset']).name
            if not dataset.is_file():
                raise FileNotFoundError(dataset)
            domain = settings['wandb']['domain']
            task = dataset.stem
            stage = settings['stage']
            identity = f'{domain}--{task}--{stage}--s{seed}'
            gpu = len(manifest['tasks']) % 2
            output = root / 'runs' / domain / task / stage / f'seed-{seed}'
            settings.update(dataset=str(dataset), output=str(output), device=f'cuda:{gpu}',
                            budget=str(root / f'usage/gpu{gpu}'), resume=None,
                            shared_gpu=True, concurrent_gpu_runs=slots)
            settings['wandb'].update(project='eegx', entity='quoc-emotiv', group=None,
                                     name=f'EEG-X--{domain}--{task}--{stage}--seed-{seed}')
            dependencies = []
            if stage != 'pretrain':
                parent_task, parent_seed = ('tuab', 241) if domain == 'cross_domain' else (task, seed)
                dependency = f'{domain}--{parent_task}--pretrain--s{parent_seed}'
                dependencies.append(dependency)
                settings['checkpoint'] = str(root / 'runs' / domain / parent_task / 'pretrain' /
                                             f'seed-{parent_seed}' / 'training/best.pth')
            path = root / 'configs' / f'{identity}.yaml'
            path.write_text(yaml.safe_dump(settings, sort_keys=False))
            # Validate the actual generated paths, stages and YAML schema.
            read_config(path)
            manifest['tasks'].append({'id': identity, 'gpu': gpu, 'stage': stage, 'domain': domain,
                                      'config': str(path), 'output': str(output),
                                      'depends': dependencies})
            manifest['config_hashes'][str(path)] = sha256(path)
            if flexible:
                other = 1 - gpu
                alternate = copy.deepcopy(settings)
                alternate.update(device=f'cuda:{other}', budget=str(root / f'usage/gpu{other}'))
                alternate_path = path.with_name(path.stem + f'--gpu{other}.yaml')
                alternate_path.write_text(yaml.safe_dump(alternate, sort_keys=False))
                read_config(alternate_path)
                manifest['config_hashes'][str(alternate_path)] = sha256(alternate_path)
                manifest['tasks'][-1]['placements'] = [
                    {'gpu': other, 'config': str(alternate_path)}]
    tasks = manifest['tasks']
    identities = {t['id'] for t in tasks}
    outputs = {t['output'] for t in tasks}
    if len(tasks) != 81 or len(identities) != 81 or len(outputs) != 81:
        raise ValueError('Fresh paper study must contain exactly 81 unique tasks and outputs')
    if any(d not in identities for t in tasks for d in t['depends']):
        raise ValueError('Queue contains an unknown dependency')
    return manifest


def prepare(root, slots, data_root, capacity_paths, memory_profile=None):
    from pretrain.pipeline.budget import Budget
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.train import read_config

    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    if type(slots) is not int or slots < 1:
        raise ValueError('Concurrency must be positive')
    capacities = [json.loads(Path(p).read_text()) for p in capacity_paths]
    if ({c['gpu'] for c in capacities} != {0, 1}
            or any(c['status'] != 'passed' or (not memory_profile and c['slots'] != slots)
                   for c in capacities)):
        raise ValueError('Both GPUs require successful capacity checks at the selected concurrency')
    contexts = [(c['resident']['used_mib'] - c['before']['used_mib']) / c['slots']
                - max(w['peak_reserved_mib'] for w in c['workers']) for c in capacities]
    reservation = math.ceil(max(w['peak_reserved_mib'] for c in capacities
                               for w in c['concurrent']) + max(contexts) + 512)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir()
    source = root / 'source'
    source.mkdir()
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    retained = [p for p in files if p and (p.startswith(('pretrain/', 'scripts/paper/',
                    'scripts/rerun/', 'configs/')) or p in
                    {'Main.py', 'CONTEXT.md', 'uv.lock', 'pyproject.toml', 'docs/spec.md', 'docs/task.md'})]
    for relative in retained:
        original, target = ROOT / relative, source / relative
        if original.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit and review all source changes before freezing the queue')
    manifest = {'source_root': str(source), 'python': str(ROOT / '.venv/bin/python'),
                'source_commit': revision, 'software_contract': software_contract(),
                'slots_per_gpu': slots, 'devices': [0, 1], 'reservation_mib': reservation,
                'headroom_mib': 2048, 'claims': str(root / '.claims'),
                'budget': str(root / 'usage/gpu0'), 'tasks': [], 'config_hashes': {}}
    configs = {p.stem: read_config(p) for p in sorted((ROOT / 'configs').glob('*.yaml'))}
    manifest.update(build_tasks(configs, root, slots, data_root, flexible=bool(memory_profile)))
    if memory_profile:
        profile = json.loads(Path(memory_profile).read_text())
        if profile['software_contract'] != manifest['software_contract']:
            raise ValueError('Memory profile belongs to different training software')
        manifest.update(admission='memory', headroom_mib=6144, memory_profile=profile)
        for task in manifest['tasks']:
            config = read_config(task['config'])
            key = config['stage'] + '/' + Path(config['dataset']).stem
            case = profile['cases'][key]
            signature = {k: config[k] for k in
                         ('depth', 'width', 'heads', 'microbatch', 'logical_batch', 'precision')}
            if case['config'] != signature or case['reservation_mib'] <= 0:
                raise ValueError(f'Memory profile configuration mismatch: {key}')
            task['reservation_mib'] = case['reservation_mib']
        (root / 'memory-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    # Meters are independent per physical GPU and have an explicit null time limit.
    for gpu in manifest['devices']:
        Budget(root / f'usage/gpu{gpu}', gpu_hours=None).initialize()
    runtime = source / 'scripts/rerun/launch.py'
    task_runtime = source / 'scripts/paper/task.py'
    provenance = {'commit': revision, 'dirty': False,
                  'software_contract': manifest['software_contract']}
    (source / 'source-revision.json').write_text(json.dumps(provenance, indent=2) + '\n')
    manifest['runtime_hashes'] = {
        str(p.relative_to(source)): sha256(p)
        for p in [runtime, task_runtime, source / 'scripts/paper/process.py',
                  source / 'scripts/paper/child.py', source / 'scripts/rerun/memory.py',
                  source / 'source-revision.json']}
    (root / 'capacity.json').write_text(json.dumps(capacities, indent=2) + '\n')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'source-revision.json').write_text(json.dumps(
        {'commit': revision, 'dirty': False, 'software_contract': manifest['software_contract']},
        indent=2) + '\n')
    import shlex
    command = ' '.join(map(shlex.quote, [manifest['python'], str(runtime), '--manifest',
                                        str(root / 'manifest.json')]))
    (root / 'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + command + '\n')
    print(json.dumps({'manifest': str(root / 'manifest.json'), 'tasks': len(manifest['tasks']),
                      'slots_per_gpu': slots, 'source_commit': revision}))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--slots', type=int, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--capacity', type=Path, nargs=2, required=True)
    parser.add_argument('--memory-profile', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.slots, args.data_root, args.capacity, args.memory_profile)
