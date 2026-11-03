"""Exercise the public study launcher without starting GPU workloads."""

import json
import os
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]


def launch(tmp_path, fail_config=""):
    executable = tmp_path / "fake python"
    executable.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$3" >> "$EEGX_TEST_CALLS"\n'
        'if [ "${3##*/}" = "$EEGX_TEST_FAIL_CONFIG" ]; then exit 7; fi\n'
    )
    executable.chmod(0o755)
    calls = tmp_path / "calls.txt"
    result = subprocess.run(
        ["bash", str(ROOT / "run_paper.sh")],
        cwd=tmp_path,
        env={**os.environ, "EEGX_PYTHON": str(executable),
             "EEGX_CONFIG_DIR": str(ROOT / "configs"),
             "EEGX_TEST_CALLS": str(calls), "EEGX_TEST_FAIL_CONFIG": fail_config},
        capture_output=True, text=True, timeout=10,
    )
    return result, [Path(line) for line in calls.read_text().splitlines()]


def test_launcher_covers_the_study_and_runs_checkpoint_dependencies_first(tmp_path):
    result, paths = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(paths) == len(set(paths))
    assert set(paths) == set((ROOT / "configs").glob("*.yaml"))
    selected_checkpoints = set()
    runs = 0
    for path in paths:
        config = yaml.safe_load(path.read_text())
        seeds = config.get("seeds", [config.get("seed")])
        runs += len(seeds)
        if config["stage"] == "pretrain":
            output = (path.parent / config["output"]).resolve()
            if "seeds" in config:
                selected_checkpoints.update(
                    output / f"seed-{seed}" / "training/best.pth" for seed in seeds
                )
            else:
                selected_checkpoints.add(output / "training/best.pth")
        else:
            for seed in seeds:
                checkpoint = config["checkpoint"].replace("{seed}", str(seed))
                assert (path.parent / checkpoint).resolve() in selected_checkpoints
    study = json.loads((ROOT / "data/study.json").read_text())
    assert runs == study["expected_runs"]


def test_launcher_stops_before_downstream_work_when_a_stage_fails(tmp_path):
    result, paths = launch(tmp_path, fail_config="stew.yaml")
    assert result.returncode == 7
    assert [p.stem for p in paths] == ["eegx", "dreamer-arousal-finetune", "stew"]
