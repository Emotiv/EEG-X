"""Check frozen resume and W&B identities on CPU before admitting GPU work."""

import copy
import json
from pathlib import Path
import sys
import tempfile


def check(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    sys.path.insert(0, manifest["source_root"])
    import torch
    from pretrain.train import read_config, RUN_KEYS
    from pretrain.pipeline.training import resolve_training_config
    from pretrain.pipeline.splits import select_split
    from pretrain.pipeline.data import Windows
    from pretrain.pipeline.checkpoint import software_contract, dataset_identity
    from pretrain.pipeline.selection import selection_metadata
    from pretrain.metadata import fingerprint

    receipts = []
    for task in manifest["tasks"]:
        if not task.get("resume_from"):
            continue
        settings = read_config(task["config"])
        payload = torch.load(settings["resume"], map_location="cpu", weights_only=True)
        source = Windows(settings["dataset"], "train")
        with tempfile.TemporaryDirectory() as directory:
            reference = select_split(
                settings["dataset"], Path(directory) / "split", settings["seed"]
            )
            train = Windows(reference, "train", verify_signals=False)
            config = resolve_training_config(
                {
                    **{key: value for key, value in settings.items() if key not in RUN_KEYS},
                    "input_samples": source.metadata["window_samples"],
                    "purpose": "paper_pretrain",
                }
            )
            config["gpu_policy"] = {
                "physical_gpu": int(settings["device"].split(":")[1]),
                "shared": settings["shared_gpu"],
                "concurrent_runs": settings["concurrent_gpu_runs"],
            }
            config["logical_batch"] = min(config["logical_batch"], len(train))
            previous = payload["contract"]
            if config != previous["config"]:
                raise ValueError(f"Resume configuration changed: {task['name']}")
            if any(previous.get(key) != value for key, value in software_contract().items()):
                raise ValueError("Frozen software or environment differs from the checkpoint")
            if previous["dataset"] != train.metadata["fingerprint"] or previous[
                "dataset_identity"
            ] != dataset_identity(train.metadata):
                raise ValueError("Resume dataset or saved split changed")
            recipe = copy.deepcopy(config)
            for key in ("seed", "purpose", "pretraining_checkpoint_sha256", "source_run_id"):
                recipe.pop(key, None)
            if recipe.get("objective"):
                recipe["objective"].pop("seed", None)
            # Match frozen track(): its seed-independent recipe uses the source
            # H5 metadata, while checkpoint contracts use the selected seed bank.
            recipe_id = fingerprint(
                {
                    "config": recipe,
                    "data": source.metadata["fingerprint"],
                    "stage": "pretrain",
                    "selection": selection_metadata("pretrain"),
                }
            )[:12]
            old = json.loads((Path(settings["resume"]).parent.parent / "wandb.json").read_text())
            if recipe_id != old["recipe_id"]:
                raise ValueError(f"W&B resume recipe changed: {task['name']}")
            receipts.append(
                {
                    "task": task["name"],
                    "seed": settings["seed"],
                    "completed_epochs": payload["state"]["epoch"],
                    "cursor": payload["state"]["cursor"],
                    "wandb_id": old["id"],
                    "recipe_id": recipe_id,
                    "output": settings["output"],
                }
            )
    return receipts


if __name__ == "__main__":
    result = check(sys.argv[1])
    text = json.dumps(result, indent=2) + "\n"
    (Path(sys.argv[1]).parent / "resume-contract-preflight.json").write_text(text)
    print(text)
