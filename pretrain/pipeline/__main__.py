"""Prepare, plan, execute and inspect the reproducible EEG-X paper-task study."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument(
        "--datasets", required=True, help="JSON task-to-manifest or blocked-reason mapping"
    )
    plan.add_argument("--study", required=True)
    plan.add_argument("--budget", required=True, help="Existing shared GPU ledger directory")
    plan.add_argument("--gpu", type=int, default=1, help="Physical NVIDIA GPU index")
    plan.add_argument("--microbatch", type=int, help="Override measured input-size defaults")
    run = sub.add_parser("run")
    run.add_argument("--study", required=True)
    run.add_argument("--wait-for-gpu", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--study", required=True)
    worker.add_argument("--job", required=True)
    worker.add_argument("--attempt", required=True)
    report = sub.add_parser("report")
    report.add_argument("--study", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        from pretrain.pipeline.study import create_plan

        result = create_plan(
            json.loads(Path(args.datasets).read_text()),
            args.study,
            physical_gpu=args.gpu,
            microbatch=args.microbatch,
            budget_root=args.budget,
        )
        print(json.dumps({"jobs": len(result["jobs"]), "datasets": result["datasets"]}))
    elif args.command == "run":
        from pretrain.pipeline.execution import execute

        execute(args.study, wait_for_gpu=args.wait_for_gpu)
    elif args.command == "export":
        from pretrain.pipeline.exports import export

        print(export(args.checkpoint, args.manifest, args.output))
    elif args.command == "worker":
        from pretrain.pipeline.execution import worker

        result = worker(args.study, args.job, args.attempt)
        raise SystemExit(0 if result["status"] == "complete" else 2)
    else:
        from pretrain.pipeline.reporting import report

        print(json.dumps(report(args.study)))


if __name__ == "__main__":
    main()
