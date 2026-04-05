"""Command line interface for the driver drowsiness project."""

from __future__ import annotations

import argparse
import json

from .data import discover_frame_records, write_manifest
from .pipelines.hog_svm import HogSvmTrainer
from .pipelines.optical_flow import OpticalFlowBaseline, OpticalFlowConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Driver drowsiness detection toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Index a dataset tree and save a CSV manifest.",
    )
    manifest_parser.add_argument("--dataset-root", required=True)
    manifest_parser.add_argument("--output", required=True)

    hog_parser = subparsers.add_parser(
        "train-hog-svm",
        help="Train the HOG-SVM pipeline and persist the best model.",
    )
    hog_parser.add_argument("--dataset-root", required=True)
    hog_parser.add_argument("--predictor", required=True)
    hog_parser.add_argument("--artifacts-dir", required=True)

    flow_parser = subparsers.add_parser(
        "evaluate-optical-flow",
        help="Evaluate the optical-flow baseline on discovered sequences.",
    )
    flow_parser.add_argument("--dataset-root", required=True)
    flow_parser.add_argument("--predictor", required=True)
    flow_parser.add_argument("--output", required=True)
    flow_parser.add_argument("--max-sequences", type=int, default=None)
    flow_parser.add_argument(
        "--method",
        choices=("lucas_kanade", "horn_schunck"),
        default="lucas_kanade",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        records = discover_frame_records(args.dataset_root)
        output = write_manifest(records, args.output)
        print(f"Wrote {len(records)} records to {output}")
        return 0

    if args.command == "train-hog-svm":
        trainer = HogSvmTrainer(args.predictor)
        summary = trainer.train_from_dataset(args.dataset_root, args.artifacts_dir)
        print(json.dumps(summary.__dict__, indent=2))
        return 0

    if args.command == "evaluate-optical-flow":
        config = OpticalFlowConfig(method=args.method)
        baseline = OpticalFlowBaseline(args.predictor, config=config)
        output = baseline.evaluate_to_json(
            args.dataset_root,
            args.output,
            max_sequences=args.max_sequences,
        )
        print(f"Saved evaluation report to {output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
