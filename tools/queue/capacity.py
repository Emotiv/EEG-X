"""Measure real training and full-encoder memory before admitting larger models."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def memory_contract(settings):
    keys = ('size', 'width', 'depth', 'heads', 'dropout', 'microbatch', 'logical_batch',
            'precision', 'compile', 'compute', 'dictionary_backend', 'normalization_backend')
    result = {key: settings.get(key) for key in keys}
    if settings['stage'] == 'pretrain':
        result['objective'] = {k: v for k, v in settings['objective'].items() if k != 'seed'}
    return result


def feasible(capacity, manifest):
    usable = max(int(row['total_mib']) for row in capacity['hardware']
                 if int(row['gpu']) in manifest['devices']) - manifest['headroom_mib']
    if any(value <= 0 or value > usable for value in capacity['reservations'].values()):
        raise ValueError('Measured memory reservation cannot fit on the configured GPUs')


def reservation(capacity, task):
    return capacity['reservations'][task['dataset'] + '/' + task['stage']]


def hardware():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    return [dict(zip(('gpu', 'uuid', 'name', 'total_mib'),
                     [part.strip() for part in row.split(',')])) for row in output.splitlines()]


def process_memory():
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    return {(uuid.strip(), int(pid)): int(used)
            for uuid, pid, used in (row.split(',') for row in output.splitlines())}


def worker(path, gpu, output):
    from pretrain.gpu import available_gpu, gpu_snapshot

    state = gpu_snapshot(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpu(gpu, allow_busy=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torch.utils.data import default_collate
    from pretrain.compute import configure_compute, autocast_context
    from pretrain.models.factory import Classifier, create_model, masks, attention_backends
    from pretrain.pipeline.checkpoint import set_seed, software_contract
    from pretrain.pipeline.dataset import DatasetReference, Windows
    from pretrain.pipeline.downstream import optimizer_groups
    from pretrain.pipeline.training import optimizer_for
    from pretrain.train import read_config, resolve_run_config

    torch.set_num_threads(2)
    settings = read_config(path)
    view = Windows(DatasetReference(settings['dataset'], settings['split_seed'],
                   settings.get('pretraining_scope', 'subject_holdout')), 'train')
    config = resolve_run_config(settings, view.metadata)
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    limit = max(0, state['free_mib'] - 6144)
    if limit < 4096:
        raise RuntimeError('Insufficient free GPU memory for the capacity check')
    torch.cuda.set_per_process_memory_fraction(min(0.9, limit / total))
    set_seed(config['seed'])
    batch = default_collate([view[i] for i in range(min(config['microbatch'], len(view)))])
    eeg, ids, labels = batch['eeg'].cuda(), batch['channel_ids'].cuda(), batch['label'].cuda()
    reconstruction = create_model(config).cuda()
    parameters = reconstruction.parameter_counts()
    stage = settings['stage']
    if stage == 'pretrain':
        model = reconstruction
        optimizer = optimizer_for(model, config['optimizer'])
    elif stage == 'finetune':
        model = Classifier(reconstruction.encoder, len(view.metadata['classes'])).cuda()
        groups, _ = optimizer_groups(model)
        for group in groups:
            group['lr'] = config['optimizer']['lr'] * group['lr_scale']
            group['weight_decay'] = config['optimizer']['weight_decay'] if group['weight_decay'] else 0
        optimizer = torch.optim.AdamW(groups, **config['optimizer'])
    else:
        model = reconstruction.encoder.requires_grad_(False)
        optimizer = None
    del reconstruction
    times = []
    with configure_compute(config, torch.device('cuda:0')):
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.train(stage != 'linear_probe')
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch.device('cuda:0'), config['precision']):
                if stage == 'pretrain':
                    mask = masks(batch['window_id'], batch['channel_ids'], samples=config['input_samples'],
                                 seed=config['seed']).cuda()
                    loss = model(eeg, ids, mask=mask)['loss']
                elif stage == 'finetune':
                    loss = torch.nn.functional.cross_entropy(model(eeg, ids), labels)
                else:
                    with torch.inference_mode():
                        model.forward_emotiv_features(eeg, ids, input_window_normalization='none')
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'],
                                                error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        if stage == 'pretrain':
            model.eval()
            with torch.inference_mode(), autocast_context(torch.device('cuda:0'), config['precision']):
                model.encoder.forward_emotiv_features(eeg, ids, input_window_normalization='none')
    torch.cuda.synchronize()
    reserved = torch.cuda.max_memory_reserved() / 2**20
    used = process_memory().get((state['uuid'], os.getpid()), 0)
    result = {'config': settings, 'software': software_contract(), 'gpu': gpu,
              'gpu_uuid': state['uuid'], 'pretraining_parameter_counts': parameters,
              'measured_model': type(model).__name__,
              'measured_model_parameters': {
                  'total': sum(p.numel() for p in model.parameters()),
                  'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)},
              'stage': stage, 'channels': len(view.metadata['channels']),
              'samples': view.metadata['window_samples'], 'classes': len(view.metadata['classes']),
              'peak_reserved_mib': reserved,
              'reservation_mib': math.ceil(max(reserved * 1.2 + 1024, used + 1024)),
              'seconds_per_update': times, 'attention': attention_backends(model),
              'allocator_limit_mib': min(0.9 * total, limit)}
    if stage == 'finetune':
        from tools.queue.benchmark import inference

        result['inference'] = inference(model, eeg, ids, config)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('stage', 'peak_reserved_mib', 'reservation_mib',
                                            'seconds_per_update')}), flush=True)


def profile(path, measurement_gpu=None):
    from pretrain.pipeline.checkpoint import software_contract

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    devices = hardware()
    if measurement_gpu is not None and (
            str(measurement_gpu) not in {d['gpu'] for d in devices}
            or len({d['name'] for d in devices}) != 1):
        raise ValueError('A shared measurement GPU requires matching physical GPU models')
    root = path.parent / 'capacity'
    root.mkdir(exist_ok=True)
    records = []
    cases = {(t['dataset'], t['stage']) for t in manifest['tasks']}
    for dataset, stage in sorted(cases):
        task = next(t for t in manifest['tasks'] if (t['dataset'], t['stage']) == (dataset, stage))
        gpu = (manifest['devices'][len(records) % len(manifest['devices'])]
               if measurement_gpu is None else measurement_gpu)
        config_path = task['configs'].get(str(gpu), next(iter(task['configs'].values())))
        target = root / f'{dataset}--{stage}.json'
        if not target.exists():
            subprocess.run([manifest['python'], str(Path(__file__).resolve()), '--worker',
                            config_path, '--gpu', str(gpu), '--output', str(target),
                            '--parent', str(os.getpid())],
                           check=True, env={**os.environ, 'OMP_NUM_THREADS': '2',
                                            'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
        record = json.loads(target.read_text())
        from pretrain.train import read_config

        if (memory_contract(record['config']) != memory_contract(read_config(config_path))
                or (measurement_gpu is not None and record['gpu'] != measurement_gpu)):
            raise ValueError('Cached capacity measurement belongs to another configuration')
        records.append(record)
    if any(r['software'] != manifest['software'] for r in records) or software_contract() != manifest['software']:
        raise ValueError('Training source changed during capacity measurement')
    result = {'software': manifest['software'], 'hardware': hardware(), 'records': records,
              'reservations': {Path(r['config']['dataset']).stem + '/' + r['stage']: r['reservation_mib']
                               for r in records}}
    feasible(result, manifest)
    (path.parent / 'capacity.json').write_text(json.dumps(result, indent=2) + '\n')
    from tools.queue.prepare import digest

    manifest['capacity_sha256'] = digest(path.parent / 'capacity.json')
    path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest')
    parser.add_argument('--worker')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--output')
    parser.add_argument('--parent', type=int)
    args = parser.parse_args()
    if args.worker:
        from tools.queue.process import protect_parent

        protect_parent(args.parent)
        worker(args.worker, args.gpu, args.output)
    else:
        profile(args.manifest, args.gpu)
