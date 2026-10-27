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


def final_test_section(metrics):
    """Explicit bars make summary-only final scores visible."""
    metrics = list(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or any(
        not isinstance(key, str) or not key.startswith("test/")
        or key == "test/confusion_matrix" for key in metrics
    ):
        raise ValueError("Final-test panels require unique scalar test metric keys")
    return section("test", [bar(key) for key in metrics] + [media("test/confusion_matrix")])


def configure_test_panels(spec, metrics):
    """Repair final-test panels while preserving run selections, filters and grouping."""
    spec = copy.deepcopy(spec)
    bank = spec["section"]["panelBankConfig"]
    replacement = final_test_section(metrics)
    groups = bank["sections"]
    index = next((i for i, group in enumerate(groups) if group["name"] == "test"), None)
    if index is None:
        groups.insert(1, replacement)
    else:
        groups[index] = replacement
    for setting in ("panelConfigOverrides", "panelPlacementOverrides"):
        overrides = bank.get(setting)
        if isinstance(overrides, dict):
            bank[setting] = {key: value for key, value in overrides.items()
                             if not key.startswith("test/")}
    return spec


def sections(stage, *, embedded_probes=False, probe_metrics=None, classifier_metrics=None):
    if stage not in {"pretrain", "finetune", "linear_probe"}:
        raise ValueError("Unknown W&B stage")
    optimizer = "StableAdamW" if stage == "pretrain" else "AdamW"
    classifier_metrics = CLASSIFICATION_METRICS if classifier_metrics is None else classifier_metrics
    root = [
        line("lr-" + optimizer, x="trainer/global_step"),
        line("epoch", x="trainer/global_step"),
        bar("best_epoch"),
    ]
    train = [line("train/loss_step"), line("train/loss_epoch")]
    if stage == "pretrain":
        train += [line(key) for key in ("train/raw_mse", "train/soft_dict", "train/grad_norm")]
        validation = [
            line("val/loss"),
            line("val/raw_mse"),
            line("val/soft_dict"),
            bar("selected/validation_mse", title="selected/val/raw_mse"),
        ]
    else:
        validation = [line("val/" + key) for key in classifier_metrics]
        validation.append(media("val/confusion_matrix"))
        validation += [bar("selected/val/" + key) for key in classifier_metrics]
    result = [section("Charts", root)]
    if stage != "pretrain":
        result.append(final_test_section("test/" + key for key in classifier_metrics))
    result += [
        section("train", train),
        section("trainer", [line("trainer/global_step")]),
        section("val", validation),
    ]
    if embedded_probes:
        for role, prefix in (("val", "linear_probe/val"), ("test", "linear_probe/test")):
            metrics = CLASSIFICATION_METRICS if probe_metrics is None else probe_metrics
            result.append(section("linear_probe/" + role,
                [line(prefix + "/" + key) for key in metrics]
                + [media(prefix + "/confusion_matrix")]))
    result.append(section("System", automatic=True))
    hidden = section("Hidden Panels")
    hidden["isOpen"] = False
    result.append(hidden)
    return result


def configure(spec, stage, *, domain=None, overview=False, probe_metrics=None, classifier_metrics=None):
    """Replace the layout, preserving table columns and manual grouping choices."""
    spec = copy.deepcopy(spec)
    bank = spec["section"]["panelBankConfig"]
    desired = sections(stage, embedded_probes=stage == "pretrain" and domain != "cross_domain",
                       probe_metrics=probe_metrics, classifier_metrics=classifier_metrics)
    if overview:
        # Like Neuralbench, automatic sections follow the metrics actually logged.
        # The shared overview must not force fields from another stage onto a run.
        hidden = section("Hidden Panels")
        hidden["isOpen"] = False
        desired = [section("Charts", automatic=True), section("test", automatic=True),
                   section("train", automatic=True), section("trainer", automatic=True),
                   section("val", automatic=True), section("linear_probe/val", automatic=True),
                   section("linear_probe/test", automatic=True), section("System", automatic=True),
                   hidden]
        # Native metric discovery adapts the shared layout to an individual run.
        # Last-prefix grouping keeps validation/test probe namespaces separate.
        bank.setdefault("settings", {})["autoOrganizePrefix"] = 2
        spec["section"]["workspaceSettings"] = {
            **(spec["section"].get("workspaceSettings") or {}), "shouldAutoGeneratePanels": True}
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
    test_bars = set()
    automatic_test = False
    probe_lines = set()
    automatic_probe_prefixes = set()
    for group in spec["section"]["panelBankConfig"]["sections"]:
        if group["name"] == "test" and group.get("isPanelsAuto"):
            automatic_test = True
        if group.get("isPanelsAuto") and group["name"] in {
            "linear_probe", "linear_probe/val", "linear_probe/test"
        }:
            automatic_probe_prefixes.add(group["name"] + "/")
        for item in group["panels"]:
            kind, config = item["viewType"], item["config"]
            if kind == "Run History Line Plot":
                probe_lines.update(key for key in config.get("metrics", [])
                                   if key.startswith(("linear_probe/val/", "linear_probe/test/")))
            if group["name"] == "test" and not group.get("isPanelsAuto") and kind == "Bar Chart":
                test_bars.update(config.get("metrics", []))
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
    expected_test = {"test/" + key for key in (*CLASSIFICATION_METRICS, "f1_score_binary")}
    visible = test_bars | (set(history_keys) if automatic_test else set())
    for key in sorted((expected_test & set(summary_keys)) - visible):
        missing.append({"section": "test", "panel": "missing final-test chart", "metric": key})
    expected_probe = {
        f"linear_probe/{role}/{key}"
        for role in ("val", "test")
        for key in (*CLASSIFICATION_METRICS, "f1_score_binary")
    } & set(history_keys)
    for key in sorted(expected_probe - probe_lines):
        if not any(key.startswith(prefix) for prefix in automatic_probe_prefixes):
            missing.append({"section": key.rsplit("/", 1)[0],
                            "panel": "missing probe trend", "metric": key})
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
    runs = None
    for target in targets:
        node = nodes[target["name"]]
        if runs is None:
            runs = list(api.runs(f"{args.entity}/{args.project}", per_page=100))
        matching = [run for run in runs if target.get("overview", False) or (
            run.config.get("stage") == target["stage"]
            and (not target.get("domain") or run.config.get("domain") == target["domain"])
        )]
        observed = {key for run in matching for key in run.summary.keys()}
        probe_metrics = [key for key in (*CLASSIFICATION_METRICS, "f1_score_binary")
                         if any(f"linear_probe/{role}/{key}" in observed for role in ("val", "test"))]
        classifier_metrics = [key for key in (*CLASSIFICATION_METRICS, "f1_score_binary")
                              if "val/" + key in observed or "test/" + key in observed]
        spec = configure(
            json.loads(node["spec"]),
            target["stage"],
            domain=target.get("domain"),
            overview=target.get("overview", False),
            probe_metrics=probe_metrics or None,
            classifier_metrics=classifier_metrics or None,
        )
        if target["stage"] != "pretrain" and not target.get("overview", False):
            candidates = ["test/" + key for key in (*CLASSIFICATION_METRICS, "f1_score_binary")]
            observed = {key for run in matching for key in run.summary.keys()}
            metrics = [key for key in candidates if key in observed]
            if metrics:
                spec = configure_test_panels(spec, metrics)
        node["displayName"] = target.get("display_name", node["displayName"])
        save_view(api, args.entity, args.project, node, spec)
        print(node["displayName"], flush=True)


if __name__ == "__main__":
    main()
