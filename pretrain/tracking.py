"""One W&B run per EEG-X seed; grouping and aggregation stay in W&B."""

from contextlib import contextmanager
import json
from pathlib import Path

from pretrain.metadata import atomic_json, fingerprint


METRIC_NAMES = {"balanced_accuracy": "bal_acc"}


def metric_names(values):
    """Keep W&B names consistent with Neuralbench and local metric names explicit."""
    result = {}
    for key, value in values.items():
        prefix, separator, name = key.rpartition("/")
        result[prefix + separator + METRIC_NAMES.get(name, name)] = value
    return result


def neuralbench_metrics(values, optimizer_name="AdamW"):
    """Keep measured values and expose Neuralbench's public chart keys."""
    result = metric_names(values)
    if "train/lr" in result:
        result[f"lr-{optimizer_name}"] = result["train/lr"]
    if "val/raw_mse" in result:
        result["val/loss"] = result["val/raw_mse"]
    return result


def confusion_figure(counts):
    """Render class counts with the same image format used by Neuralbench."""
    from matplotlib.figure import Figure
    from sklearn.metrics import ConfusionMatrixDisplay

    figure = Figure(figsize=(6, 5), dpi=120, layout="constrained")
    display = ConfusionMatrixDisplay(counts, display_labels=[str(i) for i in range(len(counts))])
    display.plot(ax=figure.subplots(), cmap="Blues", colorbar=False, values_format="d")
    return figure


def validate_settings(settings):
    if not isinstance(settings, dict) or settings.keys() - {
        "enabled",
        "project",
        "entity",
        "group",
        "mode",
        "name",
        "domain",
    }:
        raise ValueError("wandb accepts enabled, project, entity, group, mode, name and domain")
    if type(settings.get("enabled", False)) is not bool:
        raise ValueError("wandb.enabled must be a boolean")
    if settings.get("mode", "online") not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline or disabled")
    if settings.get("domain") is not None and settings["domain"] not in {
        "in_domain",
        "cross_domain",
    }:
        raise ValueError("wandb.domain must be in_domain or cross_domain")
    for key in ("project", "entity", "group", "name"):
        if (
            key in settings
            and settings[key] is not None
            and (not isinstance(settings[key], str) or not settings[key].strip())
        ):
            raise ValueError(f"wandb.{key} must be a nonempty string")


class Tracker:
    def __init__(self, run, *, optimizer_name="AdamW"):
        self.run = run
        self.optimizer_name = optimizer_name
        self.epoch = None
        self.global_step = 0

    def log(self, values):
        self.epoch = values.get("epoch", self.epoch)
        self.global_step = values.get("trainer/global_step", self.global_step)
        self.run.log(neuralbench_metrics(values, self.optimizer_name))

    def summary(self, values):
        self.run.summary.update(metric_names(values))
        if "best_epoch" in values:
            progress = {"trainer/global_step": self.global_step, "best_epoch": values["best_epoch"]}
            if self.epoch is not None:
                progress["epoch"] = self.epoch
            self.run.log(progress)

    def parameters(self, model, *, classifier_parameters=0):
        values = {
            "n_total_params": sum(p.numel() for p in model.parameters()) + classifier_parameters,
            "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
            + classifier_parameters,
        }
        self.run.config.update(values)
        self.summary(values)

    def checkpoint(self, path, *, metadata):
        """Upload the selected checkpoint with its validation-only selection record."""
        import wandb

        path = Path(path)
        name = f"model-{self.run.id}"
        artifact = wandb.Artifact(
            name,
            type="model",
            metadata={
                "model_name": "EEG-X",
                "task_name": self.run.config.get("task_name"),
                "stage": self.run.config.get("stage"),
                "domain": self.run.config.get("domain"),
                "seed": self.run.config.get("seed"),
                "source_run_id": self.run.config.get("source_run_id"),
                **metadata,
            },
        )
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=[path.stem])
        self.summary({"selected/model_artifact": f"{name}:{path.stem}"})

    def confusion(self, labels, probabilities, *, prefix="test", epoch=None):
        import numpy as np
        import wandb
        from sklearn.metrics import confusion_matrix

        counts = confusion_matrix(
            np.asarray(labels),
            np.asarray(probabilities).argmax(-1),
            labels=np.arange(probabilities.shape[1]),
        )
        figure = confusion_figure(counts)
        try:
            values = {f"{prefix}/confusion_matrix": wandb.Image(figure)}
        finally:
            figure.clear()
        if epoch is not None:
            values["epoch"] = epoch
        self.run.log(values)


@contextmanager
def track(settings, config, metadata, output, *, stage, resume=None):
    """Keep logging settings outside the model's resume contract."""
    validate_settings(settings)
    if not settings.get("enabled", False) or settings.get("mode") == "disabled":
        yield None
        return
    import copy
    import wandb

    from pretrain.pipeline.selection import selection_metadata

    selection = selection_metadata(stage)

    recipe = copy.deepcopy(config)
    for key in ("seed", "purpose", "pretraining_checkpoint_sha256", "source_run_id"):
        recipe.pop(key, None)
    if recipe.get("objective"):
        recipe["objective"].pop("seed", None)
    recipe_id = fingerprint(
        {"config": recipe, "data": metadata["fingerprint"], "stage": stage, "selection": selection}
    )[:12]
    group = settings.get("group")
    domain = settings.get("domain") or config.get("domain", "in_domain")
    evaluation_mode = stage
    identity = {
        "project": settings.get("project", "eegx"),
        "entity": settings.get("entity"),
        "group": group,
        "stage": stage,
        "domain": domain,
        "seed": config["seed"],
        "recipe_id": recipe_id,
    }
    previous = None
    if resume:
        receipt = Path(resume).parent.parent / "wandb.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError(
                    "W&B resume requires the same project, domain, group, seed and recipe"
                )
    run = wandb.init(
        project=identity["project"],
        entity=identity["entity"],
        group=group,
        job_type=stage,
        name=(
            settings.get("name") or f"{metadata['task']}-{evaluation_mode}-seed-{config['seed']}"
        ).replace("{seed}", str(config["seed"])),
        mode=settings.get("mode", "online"),
        settings=wandb.Settings(console="wrap", x_disable_stats=False),
        dir=str(output),
        id=previous["id"] if previous else None,
        resume="must" if previous and settings.get("mode", "online") == "online" else None,
        config={
            **config,
            "model_name": "EEG-X",
            "task_name": metadata["task"],
            "stage": stage,
            "domain": domain,
            "evaluation_mode": evaluation_mode,
            "recipe_id": recipe_id,
            "selection_metric": selection["metric"],
            "selection_mode": selection["mode"],
            "optimizer_name": "StableAdamW" if stage == "pretrain" else "AdamW",
        },
    )
    atomic_json(Path(output) / "wandb.json", {**identity, "id": run.id})
    run.define_metric("*", step_metric="epoch")
    run.define_metric("trainer/global_step", step_metric="_step")
    run.define_metric("epoch", step_metric="trainer/global_step")
    run.define_metric("lr-AdamW", step_metric="trainer/global_step")
    run.define_metric("lr-StableAdamW", step_metric="trainer/global_step")
    logger = Tracker(run, optimizer_name="StableAdamW" if stage == "pretrain" else "AdamW")
    logger.summary(
        {
            "run_status": "running",
            "selection/metric": selection["metric"],
            "selection/mode": selection["mode"],
        }
    )
    try:
        yield logger
    except BaseException as error:
        from pretrain.pipeline.budget import BudgetStop

        logger.summary(
            {
                "run_status": "stopped"
                if isinstance(error, (BudgetStop, KeyboardInterrupt))
                else "failed"
            }
        )
        run.finish(exit_code=1)
        raise
    else:
        logger.summary({"run_status": "complete"})
        run.finish()
