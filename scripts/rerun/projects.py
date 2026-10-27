"""Archive the existing EEG-X project by renaming it, then create its private successor."""

import argparse
import json
from pathlib import Path


def archive(entity, project, reference, receipt):
    import wandb
    from wandb.apis._generated import CREATE_PROJECT_GQL

    api = wandb.Api(timeout=30)
    names = {p.name for p in api.projects(entity, per_page=100)}
    if reference in names:
        raise ValueError(f'Reference project already exists; inspect before retrying: {reference}')
    if project not in names:
        raise ValueError(f'Project does not exist: {project}')
    runs = list(api.runs(f'{entity}/{project}', per_page=100))
    if any(r.state == 'running' for r in runs):
        raise RuntimeError('The source project has running jobs; preserve them before renaming')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError('Archive receipt already exists; do not overwrite reference evidence')
    saved = {'entity': entity, 'old_project': project, 'reference_project': reference,
             'status': 'snapshot_saved', 'runs': [
                 {'id': r.id, 'name': r.name, 'state': r.state, 'config': dict(r.config),
                  'summary': dict(r.summary), 'reference_url': f'https://wandb.ai/{entity}/{reference}/runs/{r.id}'}
                 for r in runs]}
    receipt.write_text(json.dumps(saved, indent=2, default=str) + '\n')
    api._service_api.execute_graphql(
        'mutation($input:RenameProjectInput!){renameProject(input:$input){__typename}}',
        {'input': {'entityName': entity, 'oldProjectName': project, 'newProjectName': reference}},
    )
    archived = list(wandb.Api(timeout=30).runs(f'{entity}/{reference}', per_page=100))
    if {r.id for r in archived} != {r.id for r in runs}:
        raise RuntimeError('Renamed project run IDs do not match the saved inventory')
    saved['status'] = 'reference_verified'
    receipt.write_text(json.dumps(saved, indent=2, default=str) + '\n')
    api._service_api.execute_graphql(CREATE_PROJECT_GQL,
        {'input': {'name': project, 'entityName': entity, 'access': 'PRIVATE'}})
    fresh = wandb.Api(timeout=30)
    access = fresh._service_api.execute_graphql(
        'query($n:String!,$e:String!){project(name:$n,entityName:$e){name access}}',
        {'n': project, 'e': entity})
    if isinstance(access, str):
        access = json.loads(access)
    if access['project']['access'] != 'PRIVATE' or list(fresh.runs(f'{entity}/{project}')):
        raise RuntimeError('New project must be private and empty before training')
    saved['status'] = 'complete'
    receipt.write_text(json.dumps(saved, indent=2, default=str) + '\n')
    print(json.dumps({'status': saved['status'], 'reference_runs': len(archived),
                      'reference': f'https://wandb.ai/{entity}/{reference}',
                      'fresh': f'https://wandb.ai/{entity}/{project}', 'access': 'PRIVATE'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', default='quoc-emotiv')
    parser.add_argument('--project', default='eegx')
    parser.add_argument('--reference', default='eegx-bs256')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    archive(args.entity, args.project, args.reference, args.receipt)
