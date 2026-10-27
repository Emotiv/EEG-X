"""Measure real TUAB training residency and stress a requested number of GPU slots."""

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(args):
    from pretrain.gpu import available_gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpu(args.gpu, allow_busy=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from pretrain.pipeline import training
    from pretrain.pipeline.benchmark import UpdateTimer, BenchmarkComplete
    from pretrain.pipeline.budget import Budget
    from pretrain.pipeline.dataset import DatasetReference
    from pretrain.pipeline.data import Windows
    from pretrain.train import read_config, RUN_KEYS

    torch.set_num_threads(2)
    settings = read_config(args.config)
    dataset = DatasetReference(settings['dataset'], 241, settings.get('pretraining_scope', 'subject_holdout'))
    samples = Windows(dataset, 'train', verify_signals=False).metadata['window_samples']
    supplied = {k: v for k, v in settings.items() if k not in RUN_KEYS}
    if args.stage == 'finetune':
        supplied['optimizer'] = {'lr': 1e-3, 'betas': [0.9, 0.999], 'eps': 1e-8,
                                 'weight_decay': 0.01, **supplied.get('optimizer', {})}
        supplied['scheduler'] = {'minimum_lr': 1e-6, **supplied.get('scheduler', {})}
        supplied.setdefault('gradient_clip_norm', 1.0)
    config = training.resolve_training_config({
        **supplied, 'input_samples': samples, 'seed': 241,
        'purpose': 'engineering_' + args.stage,
    })
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.output / 'usage', gpu_hours=None)
    budget.initialize()
    with budget.stop_on_signals(), budget.lease():
        for phase, count, resume in [('warm', 1, None),
                                     ('concurrent', 2, args.output / 'warm/last.pth')]:
            timer = UpdateTimer(budget, updates=count, warmup=0, max_seconds=600,
                                synchronize=torch.cuda.synchronize)
            try:
                if args.stage == 'pretrain':
                    training.run(config, dataset, args.output / phase, device='cuda:0',
                                 budget=timer, resume=resume)
                else:
                    from pretrain.pipeline import finetuning
                    finetuning.run(config, dataset, args.checkpoint, args.output / phase,
                                   device='cuda:0', budget=timer, resume=resume)
            except BenchmarkComplete:
                pass
            torch.cuda.synchronize()
            state = torch.load(args.output / phase / 'last.pth', map_location='cpu',
                               weights_only=True)['state']
            assert state['global_step'] == (1 if phase == 'warm' else 3)
            result = {'phase': phase, 'pid': os.getpid(), 'gpu': args.gpu,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                      'update_seconds': timer.times, 'global_step': state['global_step']}
            (args.output / f'{phase}.json').write_text(json.dumps(result, indent=2) + '\n')
            print('CAPACITY ' + json.dumps(result), flush=True)
            if phase == 'warm' and sys.stdin.readline().strip() != 'GO':
                raise RuntimeError('Capacity coordinator stopped before concurrent test')


def measure(args):
    from pretrain.gpu import gpu_snapshot

    args.output.mkdir(parents=True, exist_ok=False)
    before = gpu_snapshot(args.gpu)
    processes, logs = [], []
    result = {'gpu': args.gpu, 'slots': args.slots, 'before': before, 'workers': [],
              'status': 'running'}
    path = args.output / 'result.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    try:
        for slot in range(args.slots):
            log = (args.output / f'worker-{slot}.log').open('w')
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--worker', '--gpu', str(args.gpu),
                 '--output', str(args.output / f'worker-{slot}'), '--config', str(args.config),
                 '--stage', args.stage] + (['--checkpoint', str(args.checkpoint)] if args.checkpoint else []),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env={**os.environ, 'OMP_NUM_THREADS': '2',
                                         'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'},
            )
            processes.append(process)
            deadline = time.monotonic() + 600
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Capacity worker did not report residency')
                if not select.select([process.stdout], [], [], 1)[0]:
                    continue
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f'Capacity worker exited early: {process.wait()}')
                log.write(line)
                log.flush()
                if line.startswith('CAPACITY '):
                    item = json.loads(line[len('CAPACITY '):])
                    result['workers'].append(item)
                    print(f'GPU {args.gpu}: resident worker {slot+1}/{args.slots}, '
                          f'reserved {item["peak_reserved_mib"]:.0f} MiB', flush=True)
                    break
        result['resident'] = gpu_snapshot(args.gpu)
        for process in processes:
            process.stdin.write('GO\n')
            process.stdin.flush()
        for process, log in zip(processes, logs):
            remaining, _ = process.communicate(timeout=900)
            log.write(remaining)
            log.flush()
            if process.returncode:
                raise RuntimeError(f'Concurrent capacity worker failed: {process.returncode}')
        result['concurrent'] = [json.loads((args.output / f'worker-{i}/concurrent.json').read_text())
                                for i in range(args.slots)]
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        result['after'] = gpu_snapshot(args.gpu)
        path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--slots', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/tuab.yaml')
    parser.add_argument('--stage', choices=['pretrain','finetune'], default='pretrain')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    args.config = args.config.resolve()
    if args.stage == 'finetune' and not args.checkpoint:
        parser.error('Finetuning capacity checks require --checkpoint')
    (worker if args.worker else measure)(args)
