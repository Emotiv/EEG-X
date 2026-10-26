"""Train one EEG-X configuration directly from YAML."""

from contextlib import nullcontext
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


RUN_KEYS = {
    "dataset",
    "output",
    "device",
    "budget",
    "resume",
    "wandb",
    "stage",
    "checkpoint",
    "seeds",
    "purpose",
    "shared_gpu",
    "concurrent_gpu_runs",
}
TRAINING_KEYS = {
    "model",
    "size",
    "depth",
    "width",
    "heads",
    "dropout",
    "seed",
    "epochs",
    "warmup_epochs",
    "precision",
    "logical_batch",
    "microbatch",
    "workers",
    "probe_interval",
    "diagnostic_test_probes",
    "early_stopping",
    "objective",
    "compile",
    "deterministic",
    "compute",
    "optimizer",
    "scheduler",
    "gradient_clip_norm",
    "pretraining_scope",
    "linear_probe",
}


def read_config(path):
    """Resolve paths beside the YAML file and reject misspelled settings."""
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("Training YAML must contain a mapping")
    unknown = settings.keys() - RUN_KEYS - TRAINING_KEYS
    if unknown:
        raise ValueError(f"Unknown training settings: {', '.join(sorted(map(str, unknown)))}")
    if "seeds" in settings:
        seeds = settings["seeds"]
        if "seed" in settings:
            raise ValueError("Use seed for one run or seeds for a list, not both")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise ValueError("seeds must be a nonempty list of nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        for name in ("resume",):
            value = settings.get(name)
            if len(seeds) > 1 and value and "{seed}" not in str(value):
                raise ValueError(f"Multiple seeds require {{seed}} in the {name} path")
    for name in ("dataset", "output"):
        if not isinstance(settings.get(name), str) or not settings[name].strip():
            raise ValueError(f"Training YAML requires {name}")
    for name in ("dataset", "output", "budget", "resume", "checkpoint"):
        value = settings.get(name)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a path or null")
            settings[name] = str((path.parent / value).resolve())
    device = settings.setdefault("device", "cuda:1")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("device must be cpu or cuda:<physical GPU index>")
    if device != "cpu" and not settings.get("budget"):
        raise ValueError("CUDA training requires the existing shared budget path")
    stage = settings.setdefault("stage", "pretrain")
    if stage not in {"pretrain", "finetune", "linear_probe"}:
        raise ValueError("stage must be pretrain, finetune or linear_probe")
    if stage != "pretrain" and not settings.get("checkpoint"):
        raise ValueError("finetune and linear_probe require a pretraining checkpoint")
    if stage == "pretrain" and settings.get("checkpoint"):
        raise ValueError("Pretraining starts from scratch; use resume for an interrupted run")
    if stage == "linear_probe" and settings.get("resume"):
        raise ValueError("Linear probing starts a fresh classifier fit")
    if settings.setdefault("purpose", "paper") not in {"paper", "engineering"}:
        raise ValueError("purpose must be paper or engineering")
    if type(settings.setdefault("shared_gpu", False)) is not bool:
        raise ValueError("shared_gpu must be boolean")
    if settings["shared_gpu"] and device == "cpu":
        raise ValueError("shared_gpu requires a CUDA device")
    concurrency = settings.setdefault("concurrent_gpu_runs", 1)
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrent_gpu_runs must be a positive integer")
    if concurrency > 1 and not settings["shared_gpu"]:
        raise ValueError("concurrent_gpu_runs greater than one requires shared_gpu")
    scope = settings.get("pretraining_scope", "subject_holdout")
    if scope not in {"subject_holdout", "tuab_source"}:
        raise ValueError("pretraining_scope must be subject_holdout or tuab_source")
    if scope == "tuab_source":
        if stage != "pretrain" or "seeds" in settings:
            raise ValueError("Shared TUAB pretraining uses stage pretrain and one scalar seed")
        settings.setdefault("wandb", {}).setdefault("domain", "cross_domain")
    from pretrain.tracking import validate_settings

    validate_settings(settings.get("wandb", {}))
    return settings


def run_config(path):
    """Prepare one seed split, then call the existing training loop in this process."""
    settings = read_config(path)
    if "seeds" in settings:
        return run_seeds(settings)
    device = settings["device"]
    if device != "cpu":
        from pretrain.gpu import available_gpu, gpu_snapshot

        physical_gpu = int(device.split(":")[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu(
            physical_gpu, allow_busy=settings["shared_gpu"]
        )
        initial_gpu = gpu_snapshot(physical_gpu)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        device = "cuda:0"

    from pretrain.pipeline.budget import Budget, NoGpuBudget
    from pretrain.pipeline.data import Windows
    from pretrain.pipeline.splits import validate_prepared_input, validate_run_dataset, select_split
    from pretrain.pipeline.training import resolve_training_config, run
    from pretrain.tracking import track

    source = Windows(settings["dataset"], "train")
    validate_prepared_input(source.metadata)
    stage = settings["stage"]
    defaults = (
        {
            "epochs": 20,
            "warmup_epochs": 5,
            "logical_batch": 64,
            "gradient_clip_norm": 1.0,
            "early_stopping": {"patience": 5, "min_delta": 0.0},
        }
        if stage != "pretrain"
        else {}
    )
    supplied = {key: value for key, value in settings.items() if key not in RUN_KEYS}
    if stage != "pretrain":
        supplied["optimizer"] = {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            **supplied.get("optimizer", {}),
        }
        supplied["scheduler"] = {"minimum_lr": 1e-6, **supplied.get("scheduler", {})}
    config = resolve_training_config(
        {
            **defaults,
            **supplied,
            "input_samples": source.metadata["window_samples"],
            "purpose": f"{settings['purpose']}_{stage}",
        }
    )
    if device != "cpu":
        config["gpu_policy"] = {
            "physical_gpu": physical_gpu,
            "shared": settings["shared_gpu"],
            "concurrent_runs": settings["concurrent_gpu_runs"],
        }
    if device == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU training requires precision: fp32")
    root = Path(settings["output"])
    if root.exists():
        raise FileExistsError(f"Output already exists; choose a fresh output directory: {root}")
    if settings.get("resume") and not Path(settings["resume"]).is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {settings['resume']}")
    if settings.get("checkpoint") and not Path(settings["checkpoint"]).is_file():
        raise FileNotFoundError(f"Pretraining checkpoint does not exist: {settings['checkpoint']}")
    if settings.get("checkpoint"):
        import copy
        import torch
        from pretrain.metadata import fingerprint
        from pretrain.pipeline.downstream import validate_pretraining_model

        payload = torch.load(settings["checkpoint"], map_location="cpu", weights_only=True)
        pretrained = payload["config"]
        validate_pretraining_model(payload, config, target_task=source.metadata["task"])
        recipe = copy.deepcopy(pretrained)
        recipe.pop("seed", None)
        if recipe.get("objective"):
            recipe["objective"].pop("seed", None)
        config["pretraining_recipe_id"] = fingerprint(recipe)[:12]
        receipt = Path(settings["checkpoint"]).parent.parent / "wandb.json"
        if receipt.exists():
            import json

            config["source_run_id"] = json.loads(receipt.read_text())["id"]
        config["source_task"] = payload["contract"]["dataset_identity"]["task"]
    budget = NoGpuBudget()
    if device != "cpu":
        ledger = Path(settings["budget"])
        if not (ledger / "budget.json").is_file():
            raise FileNotFoundError(
                f"Shared budget ledger does not exist: {ledger / 'budget.json'}"
            )
        budget = Budget(ledger, setup_seconds=300, max_concurrent=settings["concurrent_gpu_runs"])
        budget.initialize()

    root.mkdir(parents=True, exist_ok=False)
    dataset = select_split(
        settings["dataset"],
        root,
        config["seed"],
        scope=config.get("pretraining_scope", "subject_holdout"),
    )
    train = Windows(dataset, "train", verify_signals=False)
    if settings.get("checkpoint"):
        from pretrain.pipeline.downstream import validate_pretraining_source

        validate_pretraining_source(payload, config, train.metadata, mode=stage)
    validate_run_dataset(config, train.metadata)
    config["logical_batch"] = min(config["logical_batch"], len(train))
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                **settings,
                **{key: value for key, value in config.items() if key in TRAINING_KEYS},
            },
            sort_keys=False,
        )
    )
    signals = budget.stop_on_signals() if device != "cpu" else nullcontext()
    lease = budget.lease() if device != "cpu" else nullcontext()
    with (
        signals,
        lease,
        track(
            settings.get("wandb", {}),
            config,
            source.metadata,
            root,
            stage=stage,
            resume=settings.get("resume"),
        ) as tracker,
    ):
        from pretrain.gpu import record_runtime

        monitoring = (
            record_runtime(root, physical_gpu, initial_gpu, tracker)
            if device != "cpu"
            else nullcontext()
        )
        with monitoring:
            kwargs = {"device": device, "budget": budget, "tracker": tracker}
            if stage == "pretrain":
                return run(
                    config, dataset, root / "training", resume=settings.get("resume"), **kwargs
                )
            from pretrain.pipeline import finetuning, linear_probe

            if stage == "finetune":
                kwargs["resume"] = settings.get("resume")
            module = finetuning if stage == "finetune" else linear_probe
            return module.run(config, dataset, settings["checkpoint"], root / "training", **kwargs)


def run_seeds(settings):
    """Run the configured seeds sequentially in fresh local processes."""
    import copy

    root = Path(settings["output"])
    root.mkdir(parents=True, exist_ok=False)
    entrypoint = Path(__file__).resolve().parents[1] / "Main.py"
    outputs = {}
    for seed in settings["seeds"]:
        selected = copy.deepcopy(settings)
        selected.pop("seeds")
        selected["seed"] = seed
        selected["output"] = str(root / f"seed-{seed}")
        for name in ("checkpoint", "resume"):
            if selected.get(name):
                selected[name] = selected[name].replace("{seed}", str(seed))
        config = root / f"seed-{seed}.yaml"
        config.write_text(yaml.safe_dump(selected, sort_keys=False))
        print(f"Running seed {seed} locally", flush=True)
        subprocess.run([sys.executable, str(entrypoint), "--config", str(config)], check=True)
        outputs[seed] = selected["output"]
    return outputs
