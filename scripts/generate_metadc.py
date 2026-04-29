from __future__ import annotations

import argparse

from lejit.cli import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        main(
            [
                "generate",
                "--config",
                "configs/metadc/generate.toml",
                "--model-bundle",
                args.model_bundle,
                "--output",
                args.output,
                "--device",
                args.device,
                *(["--n-samples", str(args.n_samples)] if args.n_samples is not None else []),
            ]
        )
    )
