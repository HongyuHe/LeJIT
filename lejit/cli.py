from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lejit.config import LeJITConfig
from lejit.pipeline import LeJITPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run LeJIT.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a LeJIT bundle.")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)

    generate = subparsers.add_parser("generate", help="Generate constrained rows.")
    generate.add_argument("--config", required=True)
    generate.add_argument("--model-bundle", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--n-samples", type=int)
    generate.add_argument("--device", default="cpu")

    complete = subparsers.add_parser("complete", help="Complete prefix prompts.")
    complete.add_argument("--config", required=True)
    complete.add_argument("--model-bundle", required=True)
    complete.add_argument("--prompts", required=True)
    complete.add_argument("--output", required=True)
    complete.add_argument("--device", default="cpu")
    complete.add_argument("--samples-per-prompt", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = LeJITConfig.from_toml(config_path)

    if args.command == "train":
        pipeline = LeJITPipeline.build_from_config(config, base_dir=config_path.parent)
        pipeline.train(args.output)
        return 0

    pipeline = LeJITPipeline.load(args.model_bundle, device=args.device)
    pipeline.config = config

    if args.command == "generate":
        frame = pipeline.generate(n_samples=args.n_samples, device=args.device)
        frame.to_csv(args.output, index=False)
        return 0

    prompts = pd.read_csv(args.prompts)
    frame = pipeline.complete(
        prompts=prompts,
        samples_per_prompt=args.samples_per_prompt,
        device=args.device,
    )
    frame.to_csv(args.output, index=False)
    return 0
