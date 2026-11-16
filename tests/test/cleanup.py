"""Cloud cleanup must preserve running tasks and all local checkpoints."""

import json
from types import SimpleNamespace

import pytest

from tools.queue.cleanup import sweep


@pytest.fixture
def queue(tmp_path):
    output = tmp_path / 'runs' / 'task'
    (output / 'training').mkdir(parents=True)
    (output / 'training/best.pth').write_bytes(b'local checkpoint')
    receipt = {'entity': 'quoc-emotiv', 'project': 'eegx', 'id': 'run123',
               'domain': 'in_domain', 'stage': 'pretrain'}
    (output / 'wandb.json').write_text(json.dumps(receipt))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'tasks': [{'id': 'task', 'output': str(output),
                                              'domain': 'in_domain', 'stage': 'pretrain'}]}))
    (tmp_path / 'queue-state.json').write_text(json.dumps({'status': {'task': 'complete'}}))
    (tmp_path / 'finished.json').write_text('{}')
    return manifest, output, receipt


def fake_api(artifacts):
    calls = []
    def versions(*args, **kwargs):
        calls.append((args, kwargs))
        return artifacts
    return SimpleNamespace(artifacts=versions), calls


def model(*, state='COMMITTED', kind='model', linked=False):
    deletions = []
    artifact = SimpleNamespace(id='artifact123', version='v0', size=100, state=state,
                               type=kind, is_link=linked,
                               delete=lambda **kwargs: deletions.append(kwargs))
    return artifact, deletions


def test_completed_model_deleted_once_with_local_backup_and_audit(queue):
    manifest, output, _ = queue
    artifact, deletions = model()
    api, calls = fake_api([artifact])
    done, records = set(), []
    assert sweep(api, [manifest], done, set(), records.append) == (1, True)
    assert calls == [(('model', 'quoc-emotiv/eegx/model-run123'), {'per_page': 100})]
    assert deletions == [{'delete_aliases': True}]
    assert records[0]['id'] == artifact.id
    assert records[0]['local_checkpoint'] == str(output / 'training/best.pth')
    assert (output / 'training/best.pth').read_bytes() == b'local checkpoint'
    assert sweep(api, [manifest], done, set(), records.append) == (0, True)
    assert len(deletions) == 1


def test_running_task_never_queries_cloud(queue):
    manifest, _, _ = queue
    (manifest.parent / 'queue-state.json').write_text(json.dumps({'status': {'task': 'running'}}))
    (manifest.parent / 'finished.json').unlink()
    api, calls = fake_api([])
    assert sweep(api, [manifest], set(), set(), list().append) == (0, False)
    assert calls == []


@pytest.mark.parametrize('problem', ['missing', 'empty', 'foreign_project', 'foreign_task'])
def test_invalid_local_evidence_prevents_cloud_access(queue, problem):
    manifest, output, receipt = queue
    if problem == 'missing':
        (output / 'training/best.pth').unlink()
    elif problem == 'empty':
        (output / 'training/best.pth').write_bytes(b'')
    else:
        receipt['project' if problem == 'foreign_project' else 'stage'] = 'other'
        (output / 'wandb.json').write_text(json.dumps(receipt))
    api, calls = fake_api([])
    with pytest.raises((RuntimeError, ValueError)):
        sweep(api, [manifest], set(), set(), list().append)
    assert calls == []


@pytest.mark.parametrize('kind,linked', [('dataset', False), ('model', True)])
def test_collection_validated_before_any_deletion(queue, kind, linked):
    manifest, _, _ = queue
    valid, deletions = model()
    foreign, _ = model(kind=kind, linked=linked)
    api, _ = fake_api([valid, foreign])
    with pytest.raises(ValueError):
        sweep(api, [manifest], set(), set(), list().append)
    assert deletions == []


def test_pending_upload_keeps_cleaner_alive_after_queue_finishes(queue):
    manifest, _, _ = queue
    pending, deletions = model(state='PENDING')
    api, _ = fake_api([pending])
    done = set()
    assert sweep(api, [manifest], done, set(), list().append) == (0, False)
    assert not done
    assert not deletions


def test_empty_collection_requires_two_sweeps(queue):
    manifest, _, _ = queue
    api, _ = fake_api([])
    done, empty = set(), set()
    assert sweep(api, [manifest], done, empty, list().append) == (0, False)
    assert sweep(api, [manifest], done, empty, list().append) == (0, True)
