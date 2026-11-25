"""Explicit later dispatch CLI. Never invoked by the observational or test paths.

python -m experiments.coverage_quality.modal_job --tier l40s -- <gpu.py options>
Inputs/output must be paths on the pre-existing private research volume, /research.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .gpu import build_modal_app, parser


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tier", choices=("l4", "l40s"), default="l40s")
    cli.add_argument("arguments", nargs=argparse.REMAINDER)
    args = cli.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    options = parser().parse_args(arguments)
    for path in (
        options.dataset,
        options.out,
        options.ply,
        options.external_test_dataset,
        options.frame_meta,
    ):
        if path is not None and not Path(path).is_relative_to("/research"):
            raise ValueError("Modal inputs and output must be in /research")
    app, functions = build_modal_app()
    with app.run():
        functions[args.tier].remote(arguments)


if __name__ == "__main__":
    main()
