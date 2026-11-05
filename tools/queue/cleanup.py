"""Remove cloud model copies from explicitly selected legacy queues after local completion."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import threading


def sweep(api, manifests, done, empty, record):
    """Clean only completed, locally backed-up models in the approved project."""
    deleted = 0
    finished = True
    for manifest_path in manifests:
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        state = json.loads((path.parent / 'queue-state.json').read_text())
        finished = finished and (path.parent / 'finished.json').exists()
        for task in manifest['tasks']:
            identity = str(path) + ':' + task['id']
            if identity in done or state['status'].get(task['id']) != 'complete':
                continue
            output = Path(task.get('completed_output', task['output']))
            checkpoint = output / 'training/best.pth'
            receipt_path = output / 'wandb.json'
            if (not checkpoint.is_file() or checkpoint.stat().st_size == 0
                    or not receipt_path.is_file()):
                raise RuntimeError(f'Local checkpoint or tracking receipt missing: {task["id"]}')
            receipt = json.loads(receipt_path.read_text())
            if receipt['entity'] != 'quoc-emotiv' or receipt['project'] != 'eegx':
                raise ValueError('Cleanup is restricted to quoc-emotiv/eegx')
            if any(receipt[key] != task[key] for key in ('domain', 'stage')):
                raise ValueError('Tracking receipt belongs to another task')
            name = f'quoc-emotiv/eegx/model-{receipt["id"]}'
            pending = False
            artifacts = list(api.artifacts('model', name, per_page=100))
            for artifact in artifacts:
                if artifact.type != 'model' or artifact.is_link:
                    raise ValueError('Refusing to delete a foreign or non-model artifact')
            count = 0
            for artifact in artifacts:
                if artifact.state.upper() in {'DELETED', 'PENDING_DELETION'}:
                    continue
                if artifact.state.upper() != 'COMMITTED':
                    pending = True
                    continue
                item = {'id': artifact.id, 'name': name + ':' + artifact.version,
                        'bytes': artifact.size, 'local_checkpoint': str(checkpoint)}
                artifact.delete(delete_aliases=True)
                record(item)
                count += 1
                deleted += 1
            if not pending and (count or identity in empty):
                done.add(identity)
            elif not pending:
                # Confirm an empty collection on a later sweep to allow server indexing.
                empty.add(identity)
            else:
                empty.discard(identity)
            finished = finished and identity in done
    # A stopped/failed queue can have unfinished tasks, which must stay untouched.
    return deleted, finished


def watch(manifests, root, *, once=False, interval=60):
    import wandb

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'state.json'
        saved = json.loads(state_path.read_text()) if state_path.exists() else {}
        paths = [str(Path(p).resolve()) for p in manifests]
        if saved and saved['manifests'] != paths:
            raise ValueError('Cleaner state belongs to different queues')
        done = set(saved.get('done', []))
        empty = set()
        api = wandb.Api(timeout=45)

        def record(item):
            with (root / 'deleted.jsonl').open('a') as log:
                log.write(json.dumps({**item, 'time': datetime.now(timezone.utc).isoformat()}) + '\n')

        while True:
            try:
                deleted, finished = sweep(api, paths, done, empty, record)
                value = {'manifests': paths, 'done': sorted(done), 'queues_finished': finished,
                         'last_sweep': datetime.now(timezone.utc).isoformat()}
                temporary = state_path.with_suffix('.partial')
                temporary.write_text(json.dumps(value, indent=2) + '\n')
                temporary.replace(state_path)
                print(json.dumps({'deleted': deleted, 'completed_tasks_checked': len(done),
                                  'queues_finished': finished}), flush=True)
                if once or finished:
                    return
            except Exception as error:
                print(json.dumps({'error': type(error).__name__, 'message': str(error)[:240]}), flush=True)
                if once:
                    raise
            threading.Event().wait(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    watch(args.manifests, args.root, once=args.once)
