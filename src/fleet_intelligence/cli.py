from __future__ import annotations

import argparse
import json

import pandas as pd

from .training import train_baseline, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fleet-intelligence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and evaluate the v0.1 baseline")
    train.add_argument("--data", required=True)
    train.add_argument("--report", default="artifacts/baseline_report.json")
    train.add_argument("--test-fraction", type=float, default=0.25)
    train.add_argument("--false-negative-cost", type=float, default=5.0)
    train.add_argument("--false-positive-cost", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        frame = pd.read_csv(args.data)
        _, report = train_baseline(
            frame,
            test_fraction=args.test_fraction,
            false_negative_cost=args.false_negative_cost,
            false_positive_cost=args.false_positive_cost,
        )
        write_report(report, args.report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
