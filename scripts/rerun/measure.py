"""Measure real-window GPU peaks for every task before memory-based admission."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(config_path, gpu, output):
    from pretrain.gpu import available_gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu(gpu, allow_busy=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.utils.data import default_collate
    from pretrain.compute import configure_compute, autocast_context
    from pretrain.models.factory import Classifier, create_model, masks
    from pretrain.pipeline.checkpoint import set_seed
    from pretrain.pipeline.dataset import DatasetReference, Windows
    from pretrain.pipeline.downstream import optimizer_groups
    from pretrain.pipeline.training import optimizer_for
    from pretrain.train import read_config, resolve_run_config
    from scripts.rerun.memory import process_memory, workload_contract

    torch.set_num_threads(2)
    settings = read_config(config_path)
    view = Windows(DatasetReference(settings["dataset"], settings["split_seed"],
                   settings.get("pretraining_scope", "subject_holdout")), "train")
    config = resolve_run_config(settings, view.metadata)
    set_seed(config["seed"])
    batch = default_collate([view[i] for i in range(min(config["microbatch"], len(view)))])
    eeg, ids = batch["eeg"].cuda(), batch["channel_ids"].cuda()
    labels = batch["label"].cuda()
    reconstruction = create_model(config).cuda()
    stage = settings["stage"]
    if stage == "pretrain":
        model = reconstruction
        optimizer = optimizer_for(model, config["optimizer"])
    elif stage == "finetune":
        model = Classifier(reconstruction.encoder, len(view.metadata["classes"])).cuda()
        groups, _ = optimizer_groups(model)
        optimizer = torch.optim.AdamW(groups, lr=1e-3)
    else:
        model = reconstruction.encoder
        model.requires_grad_(False)
        head = torch.nn.Linear(config["width"], len(view.metadata["classes"])).cuda()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    with configure_compute(config, torch.device("cuda:0")):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model.train(stage != "linear_probe")
            with autocast_context(torch.device("cuda:0"), config["precision"]):
                if stage == "pretrain":
                    mask = masks(batch["window_id"], batch["channel_ids"],
                                 samples=config["input_samples"], seed=config["seed"]).cuda()
                    result = model(eeg, ids, mask=mask)
                    loss = result["loss"]
                elif stage == "finetune":
                    loss = torch.nn.functional.cross_entropy(model(eeg, ids), labels)
                else:
                    with torch.no_grad():
                        features = model.forward_emotiv_features(
                            eeg, ids, input_window_normalization="none")
                    loss = torch.nn.functional.cross_entropy(head(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        if stage == "pretrain" and config["probe_interval"]:
            model.eval()
            with torch.inference_mode(), autocast_context(torch.device("cuda:0"), config["precision"]):
                model.encoder.forward_emotiv_features(eeg, ids, input_window_normalization="none")
    torch.cuda.synchronize()
    reserved = torch.cuda.max_memory_reserved() / 2**20
    used = max((mib for (uuid, pid), mib in process_memory().items() if pid == os.getpid()), default=0)
    result = {
        "gpu": gpu, "gpu_name": torch.cuda.get_device_name(),
        "workload": workload_contract(settings, view.metadata),
        "peak_reserved_mib": reserved,
        "reservation_mib": int(max(reserved + 1024, used + 512)),
        "measurement": "three training updates; embedded encoder feature extraction when enabled",
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


def profile(output, gpus):
    from pretrain.pipeline.checkpoint import software_contract
    from pretrain.train import read_config
    from pretrain.metadata import atomic_json
    from scripts.rerun.memory import hardware, memory_key

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    contract = software_contract()
    cases = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = read_config(path)
        key = memory_key(config)
        gpu = gpus[len(cases) % len(gpus)]
        receipt = output.parent / (key.replace("/", "--") + ".json")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path),
                        "--gpus", str(gpu), "--output", str(receipt)], check=True,
                       env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        measured = json.loads(receipt.read_text())
        if key not in cases or measured["reservation_mib"] > cases[key]["reservation_mib"]:
            cases[key] = measured
        print(f"PROFILE {key}: {cases[key]['reservation_mib']} MiB", flush=True)
    if software_contract() != contract:
        raise ValueError("Training source changed during memory measurement; measure again")
    atomic_json(output, {"software_contract": contract, "hardware": hardware(),
                         "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.gpus[0], args.output)
    else:
        profile(args.output, args.gpus)
