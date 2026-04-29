from __future__ import annotations

import argparse

from lejit.cli import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples-per-prompt", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extras: list[str] = []
    if args.samples_per_prompt is not None:
        extras = ["--samples-per-prompt", str(args.samples_per_prompt)]
    raise SystemExit(
        main(
            [
                "complete",
                "--config",
                "configs/cidds/complete.toml",
                "--model-bundle",
                args.model_bundle,
                "--prompts",
                args.prompts,
                "--output",
                args.output,
                "--device",
                args.device,
                *extras,
            ]
        )
    )
