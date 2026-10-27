"""Reserve measured peak device memory before a training child allocates it."""

import hashlib
import json
from pathlib import Path
import subprocess


def remaining_mib(free_mib, headroom_mib, reservations):
    """Global free memory already excludes current usage, including foreign owners."""
    return free_mib - headroom_mib - sum(
        max(0, reserved - used) for reserved, used in reservations)


def process_memory():
    result = subprocess.run([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    values = {}
    for row in result.stdout.splitlines():
        uuid, pid, used = (part.strip() for part in row.split(','))
        values[uuid, int(pid)] = int(used)
    return values


def task_usage(manifest, task, uuid, memory):
    from scripts.paper.process import process_start

    identity = hashlib.sha256(str(Path(task['output']).resolve()).encode()).hexdigest()
    path = Path(manifest['claims']) / (identity + '.json')
    if not path.exists():
        return 0
    owner = json.loads(path.read_text())
    if owner['output'] != task['output']:
        raise ValueError('Memory claim does not match its task')
    if owner['process_start'] != process_start(owner['pid']):
        return 0
    return memory.get((uuid, owner['pid']), 0)


def available_mib(manifest, live):
    from pretrain.gpu import gpu_snapshot

    # Bracket the free-memory snapshot. Credit only usage seen on both sides,
    # so an allocator release cannot credit the same memory twice.
    before = process_memory()
    states = {gpu: gpu_snapshot(gpu) for gpu in manifest['devices']}
    after = process_memory()
    memory = {key: min(used, after.get(key, 0)) for key, used in before.items()}
    available = {}
    for gpu in manifest['devices']:
        state = states[gpu]
        reservations = [(task['reservation_mib'], task_usage(manifest, task, state['uuid'], memory))
                        for process, task, log in live.values() if task['gpu'] == gpu]
        available[gpu] = remaining_mib(state['free_mib'], manifest['headroom_mib'], reservations)
    return available


def choose_placement(task, active, slots, memory, pinned=None):
    """Use either reviewed device configuration when its peak reservation fits."""
    choices = [task] + [dict(task, **placement) for placement in task.get('placements', [])]
    if pinned is not None:
        choices = [candidate for candidate in choices
                   if {'gpu': candidate['gpu'], 'config': candidate['config']} == pinned]
        if not choices:
            raise ValueError('Saved GPU placement is not a reviewed configuration')
    choices = [candidate for candidate in choices
               if active[candidate['gpu']] < slots
               and (memory is None or memory[candidate['gpu']] >= task['reservation_mib'])]
    if not choices:
        return None
    return max(choices, key=lambda t: memory[t['gpu']]) if memory is not None else choices[0]
