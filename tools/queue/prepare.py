"""Expand the existing YAML study for one model size and freeze its source."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_study(study, datasets=None, domains=None):
    """Select an explicit subset while retaining the complete source dependencies."""
    selected = copy.deepcopy(study)
    allowed_domains = {'in_domain', 'cross_domain'}
    domains = allowed_domains if domains is None else set(domains)
    available = set(study['in_domain']) | set(study['cross_domain'])
    datasets = available if datasets is None else set(datasets)
    if not domains or not domains <= allowed_domains or not datasets or not datasets <= available:
        raise ValueError('Unknown or empty dataset/domain selection')
    for domain in allowed_domains:
        selected[domain] = [name for name in study[domain] if name in datasets and domain in domains]
    selected['source'] = study['source'] if selected['cross_domain'] else None
    selected['expected_runs'] = len(study['training_seeds']) * 2 * (
        len(selected['in_domain']) + len(selected['cross_domain'])) + int(selected['source'] is not None)
    if not selected['expected_runs']:
        raise ValueError('Dataset/domain selection contains no experiments')
    return selected


def prepare(root, *, width=512, depth=8, datasets=None, domains=None, devices=None):
    from pretrain.pipeline.budget import Budget
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.models.factory import resolve_model_config
    from pretrain.train import read_config

    size = resolve_model_config({"width": width, "depth": depth})["size"]
    study = select_study(json.loads((ROOT / 'data/study.json').read_text()), datasets, domains)
    devices = [0, 1] if devices is None else list(devices)
    if not devices or len(set(devices)) != len(devices) or any(g not in (0, 1) for g in devices):
        raise ValueError('Choose distinct physical GPUs from 0 and 1')
    root = Path(root).resolve()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit and review changes before freezing the study")
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    source = root / "source"
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    retained = [name for name in tracked if name and (
        name.startswith(('pretrain/', 'tools/queue/', 'docs/', 'configs/'))
        or name in {'Main.py', 'uv.lock', 'pyproject.toml', 'CONTEXT.md'})]
    for name in retained:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    contract = software_contract()
    (source / 'source-revision.json').write_text(json.dumps(
        {"commit": revision, "dirty": False, "software_contract": contract}, indent=2) + '\n')
    manifest = {"source": str(source), "commit": revision, "software": contract,
                "python": sys.executable, "size": size, "headroom_mib": 6144,
                "slots_per_gpu": study['expected_runs'], "devices": devices,
                "tasks": [], "hashes": {}, "study": study,
                "data_release": json.loads((ROOT / 'data/manifest.json').read_text())}
    for path in sorted((ROOT / 'configs').glob('*.yaml')):
        original = read_config(path)
        domain = original['wandb']['domain']
        task = Path(original['dataset']).stem
        is_source = (domain == 'cross_domain' and original['stage'] == 'pretrain'
                     and study['source'] and task == study['source']['task'])
        if task not in study[domain] and not is_source:
            continue
        for seed in original.get('seeds', [original.get('seed')]):
            config = copy.deepcopy(original)
            config.pop('seeds', None)
            config.update(width=width, depth=depth, size=size, seed=seed, resume=None,
                          shared_gpu=True, concurrent_gpu_runs=study['expected_runs'])
            domain, stage = config['wandb']['domain'], config['stage']
            task = Path(config['dataset']).stem
            identity = f'{domain}--{task}--{stage}--s{seed}'
            output = root / 'runs' / domain / task / stage / f'seed-{seed}'
            config['output'] = str(output)
            config['wandb'].update(project='eegx', group=None,
                name=f'EEG-X--{size}--{domain}--{task}--{stage}--seed-{seed}')
            dependency = None
            if stage != 'pretrain':
                parent, parent_seed = ('tuab', 241) if domain == 'cross_domain' else (task, seed)
                dependency = f'{domain}--{parent}--pretrain--s{parent_seed}'
                config['checkpoint'] = str(root / 'runs' / domain / parent / 'pretrain' /
                                           f'seed-{parent_seed}' / 'training/best.pth')
            choices = {}
            for gpu in manifest['devices']:
                config.update(device=f'cuda:{gpu}', budget=str(root / f'usage/gpu{gpu}'))
                target = root / 'configs' / f'{identity}--gpu{gpu}.yaml'
                target.write_text(yaml.safe_dump(config, sort_keys=False))
                read_config(target)
                choices[str(gpu)] = str(target)
                manifest['hashes'][str(target)] = digest(target)
            manifest['tasks'].append({"id": identity, "stage": stage, "domain": domain,
                "dataset": task, "output": str(output), "depends": dependency, "configs": choices})
    expected = {(domain, task, stage, seed)
        for domain, stages in (("in_domain", ("pretrain", "finetune")),
                               ("cross_domain", ("linear_probe", "finetune")))
        for task in study[domain] for stage in stages for seed in study['training_seeds']}
    if study['source']:
        expected.add(('cross_domain', study['source']['task'], 'pretrain', study['source']['seed']))
    actual = {(t['domain'], t['dataset'], t['stage'],
               read_config(next(iter(t['configs'].values())))['seed']) for t in manifest['tasks']}
    if actual != expected or len(manifest['tasks']) != study['expected_runs']:
        raise ValueError("Generated configurations differ from the complete study matrix")
    identities = {t['id'] for t in manifest['tasks']}
    if any(t['depends'] and t['depends'] not in identities for t in manifest['tasks']):
        raise ValueError("Missing pretraining dependency")
    manifest['runtime_hashes'] = {str(p.relative_to(source)): digest(p)
                                 for p in (source / 'tools/queue').glob('*.py')}
    for gpu in manifest['devices']:
        Budget(root / f'usage/gpu{gpu}', gpu_hours=None).initialize()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({"manifest": str(root / 'manifest.json'), "size": size,
                      "tasks": len(manifest['tasks'])}), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--domains', nargs='+', choices=['in_domain', 'cross_domain'])
    parser.add_argument('--devices', nargs='+', type=int, choices=[0, 1])
    args = parser.parse_args()
    prepare(args.root, width=args.width, depth=args.depth, datasets=args.datasets,
            domains=args.domains, devices=args.devices)
