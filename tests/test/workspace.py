"""Check native chart bindings against the observed Neuralbench metric contract."""

import json

from pretrain.workspace import CLASSIFICATION_METRICS, configure, unresolved_bindings


def original_spec():
    return {
        "section": {
            "panelBankConfig": {"sections": [], "settings": {}},
            "runSets": [
                {
                    "search": {"query": "deleted-test-date"},
                    "grouping": [{"section": "config", "name": "task_name.value"}],
                    "selections": {"root": 0, "tree": ["deleted-run"]},
                    "filters": {},
                }
            ],
        }
    }


def test_test_loss_bar_resolves_the_existing_summary_key():
    spec = original_spec()
    spec["section"]["panelBankConfig"]["sections"] = [
        {
            "name": "test",
            "panels": [
                {
                    "viewType": "Bar Chart",
                    "config": {
                        "chartTitle": "test/loss",
                        "metrics": ["summary_metrics.test/loss"],
                    },
                }
            ],
        }
    ]
    assert unresolved_bindings(spec, set(), {"test/loss"}) == [
        {
            "section": "test",
            "panel": "test/loss",
            "metric": "summary_metrics.test/loss",
        },
        {"section": "test", "panel": "missing final-test chart", "metric": "test/loss"},
    ]
    fixed = configure(spec, "finetune", domain="in_domain")
    test = next(
        item for item in fixed["section"]["panelBankConfig"]["sections"] if item["name"] == "test"
    )
    loss = next(item for item in test["panels"] if item["config"]["chartTitle"] == "test/loss")
    assert loss["viewType"] == "Bar Chart"
    assert loss["config"]["metrics"] == ["test/loss"]
    assert "summary_metrics." not in json.dumps(fixed)


def test_both_finetuning_domains_use_identical_neuralbench_panels():
    left = configure(original_spec(), "finetune", domain="in_domain")
    right = configure(original_spec(), "finetune", domain="cross_domain")
    assert left["section"]["panelBankConfig"] == right["section"]["panelBankConfig"]
    groups = left["section"]["panelBankConfig"]["sections"]
    assert [item["name"] for item in groups] == [
        "Charts",
        "test",
        "train",
        "trainer",
        "val",
        "System",
        "Hidden Panels",
    ]
    assert {item["config"]["chartTitle"] for item in groups[0]["panels"]} >= {
        "lr-AdamW",
        "epoch",
        "best_epoch",
    }
    for spec in (left, right):
        runset = spec["section"]["runSets"][0]
        assert runset["search"]["query"] == ""
        assert runset["selections"]["root"] == 1
        assert runset["grouping"] == original_spec()["section"]["runSets"][0]["grouping"]


def test_classifier_panels_bind_only_to_logged_stores():
    spec = configure(original_spec(), "finetune")
    history = {
        "epoch",
        "trainer/global_step",
        "lr-AdamW",
        "train/loss_step",
        "train/loss_epoch",
        "train/lr_min",
        "train/lr_max",
        "val/confusion_matrix",
        "test/confusion_matrix",
    }
    history.update("val/" + key for key in CLASSIFICATION_METRICS)
    summary = {
        "best_epoch",
        "n_total_params",
        "n_trainable_params",
        "runtime/peak_allocated_gib",
        "runtime/wall_seconds",
    }
    summary.update("test/" + key for key in CLASSIFICATION_METRICS)
    summary.update("selected/val/" + key for key in CLASSIFICATION_METRICS)
    assert unresolved_bindings(spec, history, summary) == []


def test_pretraining_keeps_diagnostic_trends_separate_from_final_test():
    inside = configure(original_spec(), "pretrain", domain="in_domain")
    outside = configure(original_spec(), "pretrain", domain="cross_domain")
    names = [item["name"] for item in inside["section"]["panelBankConfig"]["sections"]]
    assert "linear_probe/val" in names and "linear_probe/test" in names
    assert "test" not in names
    assert not any(
        item["name"].startswith("linear_probe/")
        for item in outside["section"]["panelBankConfig"]["sections"]
    )
    charts = inside["section"]["panelBankConfig"]["sections"][0]
    titles = {panel["config"]["chartTitle"] for panel in charts["panels"]}
    assert "lr-StableAdamW" in titles and "lr-AdamW" not in titles
    validation = next(group for group in inside["section"]["panelBankConfig"]["sections"]
                      if group["name"] == "val")
    assert any(panel["config"]["metrics"] == ["val/soft_dict"]
               for panel in validation["panels"])
    assert "summary_metrics." not in json.dumps(inside)


def test_selected_checkpoint_validation_values_have_explicit_summary_bars():
    for stage in ("finetune", "linear_probe"):
        spec = configure(original_spec(), stage)
        val = next(
            group
            for group in spec["section"]["panelBankConfig"]["sections"]
            if group["name"] == "val"
        )
        selected = [
            item
            for item in val["panels"]
            if item["config"]["chartTitle"].startswith("selected/val/")
        ]
        assert {key for item in selected for key in item["config"]["metrics"]} == {
            "selected/val/" + key for key in CLASSIFICATION_METRICS
        }
        assert all(item["viewType"] == "Bar Chart" for item in selected)
    pretrain = configure(original_spec(), "pretrain", domain="cross_domain")
    val = next(
        group
        for group in pretrain["section"]["panelBankConfig"]["sections"]
        if group["name"] == "val"
    )
    assert any(
        item["viewType"] == "Bar Chart" and item["config"]["metrics"] == ["selected/validation_mse"]
        for item in val["panels"]
    )


def test_neuralbench_root_charts_and_complete_embedded_probe_trends():
    for stage in ("pretrain", "finetune", "linear_probe"):
        spec = configure(original_spec(), stage)
        groups = spec["section"]["panelBankConfig"]["sections"]
        assert len(groups[0]["panels"]) == 3
    metrics = (*CLASSIFICATION_METRICS, "f1_score_binary")
    spec = configure(original_spec(), "pretrain", domain="in_domain", probe_metrics=metrics)
    groups = spec["section"]["panelBankConfig"]["sections"]
    for role in ("val", "test"):
        group = next(g for g in groups if g["name"] == "linear_probe/" + role)
        trends = [p for p in group["panels"] if p["viewType"] == "Run History Line Plot"]
        assert len(trends) == 11
        assert {p["config"]["metrics"][0] for p in trends} == {
            "linear_probe/" + role + "/" + key for key in metrics}
        assert all(p["config"]["xAxis"] == "epoch" for p in trends)
        assert any(p["viewType"] == "Media Browser" for p in group["panels"])


def test_overview_automatic_sections_follow_real_logged_metrics():
    spec = configure(original_spec(), "finetune", overview=True)
    groups = {g["name"]: g for g in spec["section"]["panelBankConfig"]["sections"]}
    assert groups["Charts"]["isPanelsAuto"] is True
    assert groups["Charts"]["panels"] == []
    assert groups["trainer"]["isPanelsAuto"] is True
    for key in ("train", "val", "System"):
        assert groups[key]["isPanelsAuto"]
    assert groups["linear_probe/val"]["isPanelsAuto"] is True
    assert groups["linear_probe/val"]["panels"] == []
    assert groups["linear_probe/test"]["isPanelsAuto"] is True
    assert groups["linear_probe/test"]["panels"] == []


def test_shared_overview_does_not_force_classifier_panels_onto_pretraining_runs():
    spec = configure(original_spec(), 'finetune', overview=True)
    groups = {g['name']: g for g in spec['section']['panelBankConfig']['sections']}
    for name in ('test', 'linear_probe/val', 'linear_probe/test'):
        assert groups[name]['isPanelsAuto'] is True
        assert groups[name]['panels'] == []
    assert spec['section']['panelBankConfig']['settings']['autoOrganizePrefix'] == 2


def test_test_panel_repair_preserves_user_grouping_and_run_selection():
    import copy
    from pretrain.workspace import configure_test_panels

    spec = original_spec()
    spec['section']['workspaceSettings'] = {'shouldAutoGeneratePanels': True, 'linePlot': {}}
    bank = spec['section']['panelBankConfig']
    bank['sections'] = [{'name': 'test', 'isPanelsAuto': True, 'panels': []},
                        {'name': 'System', 'isPanelsAuto': True, 'panels': []}]
    bank['panelConfigOverrides'] = {'test/loss': {'old': True}, 'train/loss_step': {'keep': True}}
    snapshot = copy.deepcopy(spec)
    keys = ['test/' + key for key in CLASSIFICATION_METRICS] + ['test/f1_score_binary']
    fixed = configure_test_panels(spec, keys)
    assert spec == snapshot
    assert fixed['section']['runSets'] == snapshot['section']['runSets']
    assert fixed['section']['panelBankConfig']['sections'][1] == bank['sections'][1]
    assert fixed['section']['workspaceSettings']['shouldAutoGeneratePanels'] is True
    assert fixed['section'].get('settings') == snapshot['section'].get('settings')
    test = fixed['section']['panelBankConfig']['sections'][0]
    assert test['isPanelsAuto'] is False
    assert len(test['panels']) == 12
    assert len({(p['layout']['x'], p['layout']['y']) for p in test['panels']}) == 12
    assert [p['config']['metrics'][0] for p in test['panels'][:-1]] == keys
    assert fixed['section']['panelBankConfig']['panelConfigOverrides'] == {'train/loss_step': {'keep': True}}


def test_binding_audit_detects_missing_scalar_panels_even_when_image_is_valid():
    spec = original_spec()
    spec['section']['panelBankConfig']['sections'] = [{
        'name': 'test', 'isPanelsAuto': True,
        'panels': [{'viewType': 'Media Browser',
                    'config': {'chartTitle': 'test/confusion_matrix',
                               'mediaKeys': ['test/confusion_matrix']}}]}]
    scalar_keys = {'test/' + key for key in CLASSIFICATION_METRICS} | {'test/f1_score_binary'}
    missing = unresolved_bindings(spec, {'test/confusion_matrix'}, scalar_keys)
    assert scalar_keys <= {item['metric'] for item in missing}


def test_automatic_test_panels_are_valid_when_final_scores_have_history():
    spec = original_spec()
    spec['section']['panelBankConfig']['sections'] = [
        {'name': 'test', 'isPanelsAuto': True, 'panels': []}]
    final_keys = {'test/' + key for key in CLASSIFICATION_METRICS} | {'test/f1_score_binary'}
    assert unresolved_bindings(spec, final_keys, final_keys) == []


def test_probe_audit_catches_metrics_hidden_by_balanced_accuracy_only_views():
    from pretrain.workspace import line, section

    spec = original_spec()
    spec['section']['panelBankConfig']['sections'] = [
        section('linear_probe/' + role, [line('linear_probe/' + role + '/bal_acc')])
        for role in ('val', 'test')]
    recorded = {f'linear_probe/{role}/{key}' for role in ('val', 'test')
                for key in (*CLASSIFICATION_METRICS, 'f1_score_binary')}
    missing = unresolved_bindings(spec, recorded, recorded)
    assert len(missing) == 20
    assert all(item['panel'] == 'missing probe trend' for item in missing)
    spec['section']['panelBankConfig']['sections'] = [
        section('linear_probe', automatic=True)]
    assert unresolved_bindings(spec, recorded, recorded) == []


def test_binary_classifier_views_include_every_applicable_validation_metric():
    metrics = (*CLASSIFICATION_METRICS, "f1_score_binary")
    for stage in ("finetune", "linear_probe"):
        spec = configure(original_spec(), stage, classifier_metrics=metrics)
        groups = {g['name']: g for g in spec['section']['panelBankConfig']['sections']}
        assert len(groups['test']['panels']) == 12
        assert len(groups['val']['panels']) == 23
        curves = {p['config']['metrics'][0] for p in groups['val']['panels']
                  if p['viewType'] == 'Run History Line Plot'}
        assert curves == {'val/' + key for key in metrics}
        bars = {p['config']['metrics'][0] for p in groups['val']['panels']
                if p['viewType'] == 'Bar Chart'}
        assert bars == {'selected/val/' + key for key in metrics}


def test_publish_command_does_not_reinsert_classifier_bars_into_shared_overview(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    import wandb
    from pretrain import workspace

    node = {'id': 'view-id', 'name': 'personal', 'displayName': 'Personal',
            'spec': json.dumps(original_spec())}
    scores = {'test/' + key: 0.5 for key in (*CLASSIFICATION_METRICS, 'f1_score_binary')}
    run = SimpleNamespace(config={'stage': 'finetune', 'domain': 'in_domain'}, summary=scores)
    api = SimpleNamespace(runs=lambda *args, **kwargs: [run])
    targets = tmp_path / 'targets.json'
    targets.write_text(json.dumps([{'name': 'personal', 'stage': 'finetune', 'overview': True}]))
    saved = []
    monkeypatch.setattr(wandb, 'Api', lambda **kwargs: api)
    monkeypatch.setattr(workspace, 'fetch_views', lambda *args: [node])
    monkeypatch.setattr(workspace, 'save_view', lambda *args: saved.append(args[-1]))
    monkeypatch.setattr(sys, 'argv', ['workspace', '--entity', 'entity', '--project', 'project',
                                   '--targets', str(targets), '--backup', str(tmp_path / 'backup.json')])
    workspace.main()
    test = next(g for g in saved[0]['section']['panelBankConfig']['sections'] if g['name'] == 'test')
    assert test['isPanelsAuto'] is True
    assert test['panels'] == []
