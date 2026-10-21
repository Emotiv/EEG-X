import csv
import json

import pytest

from pretrain.pipeline.reporting import report
from pretrain.pipeline.study import create_plan
from pretrain.pipeline.tables import write_tables


def test_blocked_study_never_fills_paper_tables_with_old_scores(tmp_path):
    create_plan({}, tmp_path / "study")
    result = report(tmp_path / "study")
    assert result["counts"] == {"blocked": 330}
    with (tmp_path / "study" / "paper_tables.csv").open() as handle:
        assert list(csv.DictReader(handle)) == []
    saved = json.loads((tmp_path / "study" / "results.json").read_text())
    assert not saved["five_seed_summaries"]
    assert all("test" not in row for row in saved["rows"])


def test_paper_table_preserves_metric_units_and_seed_membership(tmp_path):
    summary = {
        "task": "dreamer_arousal",
        "size": "d8w64",
        "objective": "soft_hydra",
        "mode": "finetune",
        "seeds": [241, 242, 243, 244, 245],
        "metrics": {"balanced_accuracy": {"mean": 0.75, "sample_sd": 0.025}},
    }
    write_tables(tmp_path, [summary])
    with (tmp_path / "paper_tables.csv").open() as handle:
        (row,) = list(csv.DictReader(handle))
    assert float(row["mean"]) == pytest.approx(0.75)
    assert float(row["sample_sd"]) == pytest.approx(0.025)
    assert row["seeds"] == "241,242,243,244,245"
    assert r"balanced\_accuracy" in (tmp_path / "paper_tables.tex").read_text()
