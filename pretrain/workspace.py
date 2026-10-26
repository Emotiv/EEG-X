"""Native W&B panel contracts shared by both evaluation domains."""

import argparse
import copy
import json
from pathlib import Path

from pretrain.metadata import fingerprint


CLASSIFICATION_METRICS = (
    "loss",
    "acc",
    "bal_acc",
    "auroc",
    "auprc",
    "f1_score_micro",
    "f1_score_macro",
    "weighted_f1",
    "precision_macro",
    "recall_macro",
)


def panel(kind, title, metrics=(), *, x="epoch", media=None):
    config = {"chartTitle": title}
    if kind == "Media Browser":
        config.update(mediaKeys=[media], mode="gallery", gallerySettings={"axis": "run"})
    else:
        if any(key.startswith("summary_metrics.") for key in metrics):
            raise ValueError("Workspace metric names must use actual logged keys")
        config.update(metrics=list(metrics), limit=100)
        if kind == "Run History Line Plot":
            config.update(xAxis=x, smoothingType="none", groupAgg="mean", groupArea="stddev")
        elif kind == "Bar Chart":
            # Project Bar Chart resolves raw keys against run summary. Report
            # SummaryMetric wrappers add an incompatible summary_metrics prefix.
            config.update(vertical=False, groupAgg="mean", groupArea="stddev")
    return {
        "__id__": fingerprint([kind, title, list(metrics), media])[:11],
        "viewType": kind,
        "config": config,
        "layout": {"x": 0, "y": 0, "w": 8, "h": 6},
    }


def line(key, *, title=None, x="epoch"):
    return panel("Run History Line Plot", title or key, [key], x=x)


def bar(key, *, title=None):
    return panel("Bar Chart", title or key, [key])


def media(key, *, title=None):
    return panel("Media Browser", title or key, media=key)


def section(name, panels=(), *, automatic=False):
    panels = list(panels)
    for number, item in enumerate(panels):
        item["layout"].update(x=(number % 3) * 8, y=(number // 3) * 6)
    result = {
        "__id__": fingerprint(name)[:11],
        "name": name,
        "isOpen": True,
        "type": "flow",
        "panels": panels,
        "isPanelsAuto": automatic,
        "localPanelSettings": {"useRunsTableGroupingInPanels": True},
    }
    if automatic:
        result["defaultName"] = name
    return result


def sections(stage, *, embedded_probes=False):
    if stage not in {"pretrain", "finetune", "linear_probe"}:
        raise ValueError("Unknown W&B stage")
    optimizer = "StableAdamW" if stage == "pretrain" else "AdamW"
    root = [
        line("lr-" + optimizer, x="trainer/global_step"),
        line("epoch", x="trainer/global_step"),
        bar("best_epoch"),
    ]
    root += [
        bar(key)
        for key in (
            "n_total_params",
            "n_trainable_params",
            "runtime/peak_allocated_gib",
            "runtime/wall_seconds",
        )
    ]
    train = [line("train/loss_step"), line("train/loss_epoch")]
    if stage == "pretrain":
        train += [line(key) for key in ("train/raw_mse", "train/soft_dict", "train/grad_norm")]
        validation = [
            line("val/loss"),
            line("val/raw_mse"),
            bar("selected/validation_mse", title="selected/val/raw_mse"),
        ]
    else:
        root += (
            [line(key) for key in ("train/lr_min", "train/lr_max")] if stage == "finetune" else []
        )
        validation = [line("val/" + key) for key in CLASSIFICATION_METRICS]
        validation.append(media("val/confusion_matrix"))
        validation += [bar("selected/val/" + key) for key in CLASSIFICATION_METRICS]
    result = [section("Charts", root)]
    if stage != "pretrain":
        result.append(
            section(
                "test",
                [bar("test/" + key) for key in CLASSIFICATION_METRICS]
                + [media("test/confusion_matrix")],
            )
        )
    result += [
        section("train", train),
        section("trainer", [line("trainer/global_step")]),
        section("val", validation),
    ]
    if embedded_probes:
        for role, prefix in (("val", "probe/validation"), ("test", "diagnostic/test")):
            trends = [
                line(prefix + "/" + key, title="linear_probe/" + role + "/" + key)
                for key in CLASSIFICATION_METRICS
            ]
            selected = [
                bar(
                    "selected/" + prefix + "/" + key,
                    title="selected/linear_probe/" + role + "/" + key,
                )
                for key in CLASSIFICATION_METRICS
            ]
            result.append(
                section(
                    "linear_probe/" + role,
                    trends
                    + [media(prefix + "/confusion_matrix")]
                    + selected
                    + [media("selected/" + prefix + "/confusion_matrix")],
                )
            )
    result.append(section("System", automatic=True))
    hidden = section("Hidden Panels")
    hidden["isOpen"] = False
    result.append(hidden)
    return result


def configure(spec, stage, *, domain=None, overview=False):
    """Replace the layout, preserving table columns and manual grouping choices."""
    spec = copy.deepcopy(spec)
    bank = spec["section"]["panelBankConfig"]
    desired = sections(stage, embedded_probes=stage == "pretrain" and domain != "cross_domain")
    if overview:
        desired[0]["panels"][0] = line("train/lr", title="learning_rate")
        val = next(item for item in desired if item["name"] == "val")
        val["panels"].append(bar("selected/validation_mse", title="selected/val/raw_mse"))
        desired[desired.index(val)] = section("val", val["panels"])
        # The project overview supports selecting either completed stage.
        probes = sections("pretrain", embedded_probes=True)
        desired[-2:-2] = [item for item in probes if item["name"].startswith("linear_probe/")]
    bank.update(sections=desired, state=1)
    bank.pop("panelConfigOverrides", None)
    bank.setdefault("settings", {}).update(showEmptySections=False, sortAlphabetically=False)
    for runset in spec["section"]["runSets"]:
        runset["search"] = {"query": ""}
        runset["selections"] = {"root": 1, "bounds": [], "tree": []}
        runset["expandedRowAddresses"] = []
        filters = []
        if not overview:
            filters.append(
                {
                    "key": {"section": "config", "name": "stage"},
                    "op": "=",
                    "value": stage,
                    "disabled": False,
                }
            )
            if domain:
                filters.append(
                    {
                        "key": {"section": "config", "name": "domain"},
                        "op": "=",
                        "value": domain,
                        "disabled": False,
                        "connector": "AND",
                    }
                )
        runset["filters"] = {"filterFormat": "filterV2", "filters": filters}
    return spec


def unresolved_bindings(spec, history_keys, summary_keys):
    missing = []
    for group in spec["section"]["panelBankConfig"]["sections"]:
        for item in group["panels"]:
            kind, config = item["viewType"], item["config"]
            known = summary_keys if kind == "Bar Chart" else history_keys
            keys = (
                config.get("mediaKeys", [])
                if kind == "Media Browser"
                else config.get("metrics", [])
            )
            for key in keys:
                if key not in known:
                    missing.append(
                        {"section": group["name"], "panel": config["chartTitle"], "metric": key}
                    )
    return missing


def graphql(api, query, variables):
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None:
        return service.execute_graphql(query, variables=variables)
    raise RuntimeError("This W&B workspace command requires the locked SDK GraphQL transport")


def fetch_views(api, entity, project):
    query = """query($entity:String,$project:String){
      project(name:$project,entityName:$entity){allViews(viewType:"project-view"){
        edges{node{id name displayName type spec}}}}}"""
    response = graphql(api, query, {"entity": entity, "project": project})
    return [edge["node"] for edge in response["project"]["allViews"]["edges"]]


def save_view(api, entity, project, node, spec):
    query = """mutation($id:ID,$entity:String,$project:String,$name:String,$display:String,$spec:String){
      upsertView(input:{id:$id,entityName:$entity,projectName:$project,name:$name,
        displayName:$display,type:"project-view",spec:$spec}){view{id name} inserted}}"""
    result = graphql(
        api,
        query,
        {
            "id": node["id"],
            "entity": entity,
            "project": project,
            "name": node["name"],
            "display": node["displayName"],
            "spec": json.dumps(spec),
        },
    )
    if result["upsertView"]["inserted"]:
        raise RuntimeError("Expected an existing workspace update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="eegx")
    parser.add_argument("--targets", required=True, help="Workspace names, stages and domains JSON")
    parser.add_argument("--backup", required=True)
    args = parser.parse_args()
    import wandb

    api = wandb.Api(timeout=60)
    nodes = {node["name"]: node for node in fetch_views(api, args.entity, args.project)}
    targets = json.loads(Path(args.targets).read_text())
    Path(args.backup).write_text(json.dumps(list(nodes.values()), indent=2) + "\n")
    for target in targets:
        node = nodes[target["name"]]
        spec = configure(
            json.loads(node["spec"]),
            target["stage"],
            domain=target.get("domain"),
            overview=target.get("overview", False),
        )
        node["displayName"] = target.get("display_name", node["displayName"])
        save_view(api, args.entity, args.project, node, spec)
        print(node["displayName"], flush=True)


if __name__ == "__main__":
    main()
