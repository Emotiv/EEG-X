"""Write two disjoint fixed seed queues for the current paper study; launch nothing."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

import yaml


def prepare(root=None):
    repository = Path.cwd()
    old = Path(json.loads(Path(".runs/current-paper-launch.json").read_text())["root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(root) if root is not None else repository / ".runs" / f"paper-expansion-{stamp}"
    if (root / "handoff.json").exists() or (root / "guardian.json").exists():
        raise RuntimeError("Cannot rewrite a queue that has started")
    root.mkdir(exist_ok=True)
    for directory in ("configs", "logs", "runtime"):
        (root / directory).mkdir(exist_ok=True)
    source = old / "source"
    budget = repository / ".runs/paper-gpu1-v1"
    meter = budget / "meters/gpu0"
    manifest = {
        "source_launch": str(old),
        "source_root": str(source),
        "python": str(repository / ".venv/bin/python"),
        "budget": str(budget),
        "meter": str(meter),
        "claims": str(repository / ".runs/paper-v1/.claims"),
        "runtime_root": str(root / "runtime"),
        "cohorts": {"0": [244, 245], "1": [241, 242, 243]},
        "tasks": [],
        "software_contract": json.loads((old / "software-contract.json").read_text()),
    }
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in (source / "configs").glob("*.yaml")}
    for name, settings in sorted(configs.items()):
        for seed in settings.get("seeds", [settings.get("seed")]):
            gpu = 0 if seed in (244, 245) or name == "tuab-source" else 1
            task = copy.deepcopy(settings)
            task.pop("seeds", None)
            task["seed"] = seed
            if name != "tuab-source":
                task["output"] = str(Path(settings["output"]) / f"seed-{seed}")
            task["device"] = f"cuda:{gpu}"
            task["budget"] = str(meter if gpu == 0 else budget)
            for field in ("checkpoint", "resume"):
                if task.get(field):
                    task[field] = task[field].replace("{seed}", str(seed))
            path = root / "configs" / f"{name}--seed-{seed}.yaml"
            path.write_text(yaml.safe_dump(task, sort_keys=False))
            manifest["tasks"].append(
                {
                    "name": name,
                    "seed": seed,
                    "gpu": gpu,
                    "output": task["output"],
                    "config": str(path),
                    "stage": task["stage"],
                    "domain": task["wandb"]["domain"],
                }
            )
    if len(manifest["tasks"]) != 81 or len({t["output"] for t in manifest["tasks"]}) != 81:
        raise ValueError("Expected exactly eighty-one distinct paper outputs")
    for gpu, seeds in ((0, [244, 245]), (1, [241, 242, 243])):
        log_dir = root / "logs" / f"gpu{gpu}"
        log_dir.mkdir(exist_ok=True)
        q = shlex.quote
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {q(str(source))}",
            "export PYTHONUNBUFFERED=1",
            f"python_bin={q(manifest['python'])}",
            f"task_script={q(str(root / 'runtime/task.py'))}",
            f"manifest={q(str(root / 'manifest.json'))}",
            f"config_dir={q(str(root / 'configs'))}",
            f"log_dir={q(str(log_dir))}",
            "run() {",
            '    "$python_bin" "$task_script" --config "$config_dir/$1--seed-$2.yaml" --manifest "$manifest" --guardian "$EEGX_GUARDIAN"',
            "}",
            "in_domain() {",
        ]
        lines += [f'    run "$1" {seed}' for seed in seeds]
        lines += [f'    run "$2" {seed}' for seed in seeds]
        lines += ["}", "cross_domain() {"]
        lines += [f'    run "$1-linear-probe" {seed}' for seed in seeds]
        lines += [f'    run "$1-finetune" {seed}' for seed in seeds]
        lines += [
            "}",
            "wait_all() {",
            "    local failed=0 pid",
            '    for pid in "$@"; do wait "$pid" || failed=1; done',
            '    return "$failed"',
            "}",
        ]
        for number, (pre, fine, task) in enumerate(
            (
                ("eegx", "dreamer-arousal-finetune", "dreamer-arousal"),
                ("stew", "stew-finetune", "stew"),
                ("tuab", "tuab-finetune", "tuab"),
                ("tuev-binary", "tuev-binary-finetune", "tuev-binary"),
            ),
            1,
        ):
            lines += [f'in_domain {pre} {fine} >"$log_dir/{task}.log" 2>&1 &', f"p{number}=$!"]
        lines += ['wait_all "$p1" "$p2" "$p3" "$p4"']
        if gpu == 0:
            lines += [
                'run tuab-source 241 >"$log_dir/tuab-source.log" 2>&1',
                f"touch {q(str(root / 'source-complete'))}",
                'flock -u "$EEGX_SOURCE_FD"',
            ]
        else:
            result = str(Path(configs["tuab-source"]["output"]) / "training/result.json")
            check = "import json,sys,pathlib; assert pathlib.Path(sys.argv[2]).is_file() and json.load(open(sys.argv[1]))['status']=='complete', 'Shared TUAB source did not finalize successfully'"
            lines += [
                f"exec 8>{q(str(root / 'source.lock'))}",
                "flock -s 8",
                f'"$python_bin" -c {q(check)} {q(result)} {q(str(root / "source-complete"))}',
                "flock -u 8",
                "exec 8>&-",
            ]
        for number, target in enumerate(
            ("crowdsourced", "bciciv-2a", "kalunga", "physionet-mi"), 1
        ):
            lines += [f'cross_domain {target} >"$log_dir/{target}.log" 2>&1 &', f"p{number}=$!"]
        lines += [
            'wait_all "$p1" "$p2" "$p3" "$p4"',
            f'printf "GPU {gpu} assigned tasks complete.\\n"',
        ]
        (root / f"queue-gpu{gpu}.sh").write_text("\n".join(lines) + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    Path(".runs/pending-paper-queues.json").write_text(
        json.dumps({"manifest": str(root / "manifest.json")}, indent=2) + "\n"
    )
    return root


if __name__ == "__main__":
    print(prepare())
