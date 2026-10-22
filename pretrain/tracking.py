"""One W&B run per EEG-X seed; grouping and aggregation stay in W&B."""

from contextlib import contextmanager
import json
from pathlib import Path

from pretrain.metadata import atomic_json, fingerprint


def validate_settings(settings):
    if not isinstance(settings, dict) or settings.keys() - {
        "enabled",
        "project",
        "entity",
        "group",
        "mode",
    }:
        raise ValueError("wandb accepts enabled, project, entity, group and mode")
    if type(settings.get("enabled", False)) is not bool:
        raise ValueError("wandb.enabled must be a boolean")
    if settings.get("mode", "online") not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline or disabled")
    for key in ("project", "entity", "group"):
        if (
            key in settings
            and settings[key] is not None
            and (not isinstance(settings[key], str) or not settings[key].strip())
        ):
            raise ValueError(f"wandb.{key} must be a nonempty string")


class Tracker:
    def __init__(self, run):
        self.run = run

    def log(self, values):
        self.run.log(values)

    def summary(self, values):
        self.run.summary.update(values)

    def confusion(self, labels, probabilities):
        import numpy as np
        import wandb

        self.run.log(
            {
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=np.asarray(labels).tolist(),
                    preds=np.asarray(probabilities).argmax(-1).tolist(),
                    class_names=[str(index) for index in range(probabilities.shape[1])],
                )
            }
        )


@contextmanager
def track(settings, config, metadata, output, *, stage, resume=None):
    """Keep logging settings outside the model's resume contract."""
    validate_settings(settings)
    if not settings.get("enabled", False) or settings.get("mode") == "disabled":
        yield None
        return
    import copy
    import wandb

    recipe = copy.deepcopy(config)
    for key in ("seed", "purpose", "pretraining_checkpoint_sha256"):
        recipe.pop(key, None)
    if recipe.get("objective"):
        recipe["objective"].pop("seed", None)
    recipe_id = fingerprint({"config": recipe, "data": metadata["fingerprint"], "stage": stage})[
        :12
    ]
    group = settings.get("group") or f"EEG-X/{metadata['task']}/{stage}/{recipe_id}"
    identity = {
        "project": settings.get("project", "eeg-x"),
        "entity": settings.get("entity"),
        "group": group,
        "stage": stage,
        "seed": config["seed"],
        "recipe_id": recipe_id,
    }
    previous = None
    if resume:
        receipt = Path(resume).parent.parent / "wandb.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError("W&B resume requires the same project, group, seed and recipe")
    run = wandb.init(
        project=identity["project"],
        entity=identity["entity"],
        group=group,
        job_type=stage,
        name=f"{metadata['task']}-{stage}-seed-{config['seed']}",
        mode=settings.get("mode", "online"),
        dir=str(output),
        id=previous["id"] if previous else None,
        resume="must" if previous and settings.get("mode", "online") == "online" else None,
        config={
            **config,
            "model_name": "EEG-X",
            "task_name": metadata["task"],
            "stage": stage,
            "recipe_id": recipe_id,
        },
    )
    atomic_json(Path(output) / "wandb.json", {**identity, "id": run.id})
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    logger = Tracker(run)
    logger.summary({"run_status": "running"})
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
