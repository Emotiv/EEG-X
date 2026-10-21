"""Measured results and explicit missing cells; published scores never fill gaps."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pretrain.channel_identity import sha256_file
from pretrain.pipeline.metrics import classification
from pretrain.pipeline.data import Windows
from pretrain.pipeline.study import latest, read_plan, dataset_for
from pretrain.pipeline.tables import write_tables
from pretrain.metadata import atomic_json, fingerprint


def report(root):
    root = Path(root)
    plan = read_plan(root)
    rows, grouped = [], defaultdict(list)
    for job in plan["jobs"]:
        task = plan["datasets"][job["task"]]
        state = latest(root, job["id"])
        if task["status"] == "blocked":
            state = {"status": "blocked", "reason": task["reason"]}
        if state is None and job["depends_on"]:
            source = next(item["task"] for item in plan["jobs"] if item["id"] == job["depends_on"])
            if plan["datasets"][source]["status"] == "blocked":
                state = {"status": "blocked", "reason": "Pretraining source is blocked"}
        row = {
            "id": job["id"],
            "mode": job["mode"],
            "task": job["task"],
            "size": job["config"]["size"],
            "seed": job["config"]["seed"],
            "objective": job["config"]["objective"]["name"]
            if job["config"].get("objective") else "raw",
            **(state or {"status": "pending"}),
        }
        if row["status"] == "complete" and job["mode"] != "pretrain":
            output = Path(row["output"])
            result = json.loads((output / "result.json").read_text())
            selection = json.loads((output / "selection.json").read_text())
            if result["status"] != "complete" or selection != result["selection"]:
                raise ValueError("Completed result differs from its saved selection")
            if selection["status"] != "locked_before_test":
                raise ValueError("Paper results require a saved selection lock")
            provenance = selection["contract"]
            if any(provenance[key] != value for key, value in plan["software"].items()):
                raise ValueError("Result software differs from the frozen study")
            if provenance["dataset"] != dataset_for(plan, job["task"], row["seed"])["fingerprint"]:
                raise ValueError("Result split differs from the frozen study")
            if any(provenance["config"].get(key) != value for key, value in job["config"].items()):
                raise ValueError("Result configuration differs from the frozen study")
            prediction_path = output / "test_predictions.npz"
            with np.load(prediction_path, allow_pickle=False) as prediction:
                manifest = dataset_for(plan, job["task"], row["seed"])["manifest"]
                test = Windows(manifest, "test", verify_signals=False)
                expected_ids = [
                    fingerprint(
                        [
                            test.metadata["fingerprint"],
                            test.metadata["records"][int(test.index[i]["record"])]["recording"],
                            int(test.index[i]["start"]),
                        ]
                    )
                    for i in test.rows
                ]
                if prediction["window_ids"].tolist() != expected_ids or not np.array_equal(
                    prediction["labels"], test.index["label"][test.rows]
                ):
                    raise ValueError("Predictions do not cover the frozen test membership")
                row["test"] = classification(prediction["labels"], prediction["probabilities"])
            if row["test"] != result["test"]:
                raise ValueError("Reported metrics differ from saved test predictions")
            artifact = output / ("probe.joblib" if job["mode"] == "frozen" else "best.pth")
            expected = selection["probe_sha256" if job["mode"] == "frozen" else "checkpoint_sha256"]
            if sha256_file(artifact) != expected:
                raise ValueError("Selected artifact changed after evaluation")
            row["evidence"] = {
                path.name: sha256_file(path)
                for path in (
                    output / "selection.json",
                    prediction_path,
                    output / "run.json",
                    artifact,
                )
            }
            grouped[(row["task"], row["size"], row["objective"], row["mode"])].append(row)
        rows.append(row)
    summaries = []
    for key, values in sorted(grouped.items()):
        if {value["seed"] for value in values} != {241, 242, 243, 244, 245}:
            continue
        summaries.append(
            {
                "task": key[0],
                "size": key[1],
                "objective": key[2],
                "mode": key[3],
                "seeds": sorted(value["seed"] for value in values),
                "metrics": {
                    metric: {
                        "mean": float(np.mean([v["test"][metric] for v in values])),
                        "sample_sd": float(np.std([v["test"][metric] for v in values], ddof=1)),
                    }
                    for metric in values[0]["test"]
                },
            }
        )
    budget_path = Path(plan["budget_root"]) / "budget.json"
    budget = (
        json.loads(budget_path.read_text())
        if budget_path.exists()
        else {
            "used_seconds": plan.get("setup_seconds", 300),
            "setup_seconds": plan.get("setup_seconds", 300),
            "limit_seconds": plan["gpu_hours"] * 3600,
        }
    )
    result = {
        "study": plan["fingerprint"],
        "rows": rows,
        "five_seed_summaries": summaries,
        "counts": dict(Counter(row["status"] for row in rows)),
        "budget": budget,
        "auxiliary": {
            "status": "pending_main_matrix",
            "teacher_dependent_ablations": "not_applicable",
        },
    }
    atomic_json(root / "results.json", result)
    write_tables(root, summaries)
    lines = ["# EEG-X study results", "", "| Status | Runs |", "|---|---:|"]
    lines += [f"| {status} | {count} |" for status, count in sorted(result["counts"].items())]
    lines += [
        "",
        "Only measured runs count as complete. Blocked tasks retain their data evidence.",
        "",
        "| GPU hours used | GPU hour limit |",
        "|---:|---:|",
        f"| {budget['used_seconds'] / 3600:.4f} | {budget['limit_seconds'] / 3600:g} |",
        "",
        "The shared ledger includes all reserved experiment time.",
        "",
    ]
    if summaries:
        lines += [
            "| Task | Size | Objective | Mode | Metric | Mean | Sample SD |",
            "|---|---|---|---|---|---:|---:|",
        ]
        for summary in summaries:
            for metric, value in summary["metrics"].items():
                lines.append(
                    f"| {summary['task']} | {summary['size']} | {summary['objective']} | {summary['mode']} | {metric} | {value['mean']:.6f} | {value['sample_sd']:.6f} |"
                )
        lines += [
            "",
            "Each summary requires the full accepted seed bank. Individual results remain in the registry.",
            "",
        ]
    (root / "results.md").write_text("\n".join(lines))
    return {"counts": result["counts"], "five_seed_groups": len(summaries), "budget": budget}
