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
        }
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
