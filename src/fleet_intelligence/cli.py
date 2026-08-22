from __future__ import annotations

import argparse
import json

import pandas as pd

from .anomaly import evaluate_anomaly_detection
from .mobility import evaluate_mobility_release
from .training import compare_models, train_baseline, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fleet-intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and evaluate the v0.1 baseline")
    train.add_argument("--data", required=True)
    train.add_argument("--report", default="artifacts/baseline_report.json")
    train.add_argument("--test-fraction", type=float, default=0.25)
    train.add_argument("--false-negative-cost", type=float, default=5.0)
    train.add_argument("--false-positive-cost", type=float, default=1.0)

    compare = subparsers.add_parser(
        "compare",
        help="Run the v0.2 baseline-vs-candidate promotion evaluation",
    )
    compare.add_argument("--data", required=True)
    compare.add_argument("--report", default="artifacts/model_comparison_v0.2.json")
    compare.add_argument("--artifact-dir", default="artifacts/models")
    compare.add_argument("--dev-fraction", type=float, default=0.20)
    compare.add_argument("--test-fraction", type=float, default=0.20)
    compare.add_argument("--false-negative-cost", type=float, default=5.0)
    compare.add_argument("--false-positive-cost", type=float, default=1.0)

    anomaly = subparsers.add_parser(
        "anomaly",
        help="Run the v0.3 telemetry anomaly and robustness benchmark",
    )
    anomaly.add_argument("--data", required=True)
    anomaly.add_argument("--report", default="artifacts/anomaly_report_v0.3.json")
    anomaly.add_argument("--artifact-dir", default="artifacts/anomaly-model")
    anomaly.add_argument("--dev-fraction", type=float, default=0.20)
    anomaly.add_argument("--test-fraction", type=float, default=0.20)
    anomaly.add_argument("--false-negative-cost", type=float, default=8.0)
    anomaly.add_argument("--false-positive-cost", type=float, default=1.0)

    mobility = subparsers.add_parser(
        "eta-routing",
        help="Run the v0.4 ETA prediction and capacitated routing benchmark",
    )
    mobility.add_argument("--trips", required=True)
    mobility.add_argument("--stops", required=True)
    mobility.add_argument("--report", default="artifacts/mobility_report_v0.4.json")
    mobility.add_argument("--artifact-dir", default="artifacts/mobility")
    mobility.add_argument("--dev-fraction", type=float, default=0.20)
    mobility.add_argument("--test-fraction", type=float, default=0.20)
    mobility.add_argument("--vehicle-count", type=int, default=3)
    mobility.add_argument("--vehicle-capacity", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "eta-routing":
        report = evaluate_mobility_release(
            pd.read_csv(args.trips),
            pd.read_csv(args.stops),
            dev_fraction=args.dev_fraction,
            test_fraction=args.test_fraction,
            vehicle_count=args.vehicle_count,
            vehicle_capacity=args.vehicle_capacity,
            artifact_dir=args.artifact_dir,
        )
    else:
        frame = pd.read_csv(args.data)
        if args.command == "train":
            _, report = train_baseline(
                frame,
                test_fraction=args.test_fraction,
                false_negative_cost=args.false_negative_cost,
                false_positive_cost=args.false_positive_cost,
            )
        elif args.command == "compare":
            report = compare_models(
                frame,
                dev_fraction=args.dev_fraction,
                test_fraction=args.test_fraction,
                false_negative_cost=args.false_negative_cost,
                false_positive_cost=args.false_positive_cost,
                artifact_dir=args.artifact_dir,
            )
        else:
            report = evaluate_anomaly_detection(
                frame,
                dev_fraction=args.dev_fraction,
                test_fraction=args.test_fraction,
                false_negative_cost=args.false_negative_cost,
                false_positive_cost=args.false_positive_cost,
                artifact_dir=args.artifact_dir,
            )

    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
