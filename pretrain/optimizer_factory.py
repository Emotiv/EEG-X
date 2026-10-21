"""Pretraining optimizer construction and parameter-routing policy."""

from __future__ import annotations

import torch


_MUON_EXCLUDED_NAME_PARTS = (
    "embed",
    "embedding",
    "token",
    "head",
    "pred",
    "pos",
    "enc_channel_emd",
    "dec_channel_emd",
)


def _should_use_muon_for_param(
    name: str,
    parameter: torch.nn.Parameter,
) -> bool:
    if not parameter.requires_grad or parameter.ndim != 2:
        return False
    lowered_name = name.lower()
    return not any(marker in lowered_name for marker in _MUON_EXCLUDED_NAME_PARTS)


def build_muon_param_groups(model: torch.nn.Module, args) -> list[dict]:
    """Route matrix weights to Muon and embeddings/heads to AdamW."""
    adamw_params = []
    muon_params_by_shape: dict[tuple[int, ...], list[torch.nn.Parameter]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _should_use_muon_for_param(name, parameter):
            muon_params_by_shape.setdefault(
                tuple(parameter.shape),
                [],
            ).append(parameter)
        else:
            adamw_params.append(parameter)

    if not muon_params_by_shape:
        raise ValueError(
            f"Optimizer {args.optimizer!r} did not find any eligible 2D matrix parameters for Muon."
        )

    param_groups = [
        {
            "kind": "adamw",
            "params": adamw_params,
            "lr": args.lr,
            "betas": (args.adamw_beta1, args.adamw_beta2),
            "eps": args.adamw_eps,
            "weight_decay": args.weight_decay,
        }
    ]
    for shape in sorted(muon_params_by_shape):
        param_groups.append(
            {
                "kind": "muon",
                "params": muon_params_by_shape[shape],
                "lr": args.lr,
                "momentum": args.muon_momentum,
                "ns_steps": args.muon_ns_steps,
                "beta2": args.muon_beta2,
                "weight_decay": args.weight_decay,
            }
        )
    return param_groups


def build_optimizer(
    model: torch.nn.Module,
    args,
) -> torch.optim.Optimizer:
    """Build an optimizer, honoring an optional model-owned AdamW grouping."""
    if args.optimizer == "stable_adamw":
        from pretrain.optimizers.stable_adamw import StableAdamW

        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return StableAdamW(
            parameters,
            lr=args.lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            weight_decay=args.weight_decay,
            eps=args.adamw_eps,
            return_norms=True,
        )

    if args.optimizer == "adamw":
        param_groups_builder = getattr(model, "build_adamw_param_groups", None)
        parameters = (
            param_groups_builder(args.weight_decay)
            if callable(param_groups_builder)
            else model.parameters()
        )
        return torch.optim.AdamW(
            parameters,
            lr=args.lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            weight_decay=args.weight_decay,
            eps=args.adamw_eps,
        )

    from pretrain.optim import DistMuonAdamW, MuonAdamW

    param_groups = build_muon_param_groups(model, args)
    if args.optimizer == "dist_muon_adamw" and not getattr(
        args,
        "distributed",
        False,
    ):
        raise ValueError(
            "dist_muon_adamw requires torch.distributed to be initialized via torchrun."
        )
    optimizer_type = DistMuonAdamW if args.optimizer == "dist_muon_adamw" else MuonAdamW
    optimizer = optimizer_type(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
