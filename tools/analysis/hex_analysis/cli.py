"""Run one registry module on synthetic bundle depth; stdout is an Entities JSON array."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from hex_analysis.fixture import FixturePlantIdentifier, FixtureSegmenter, load_fixture
from hex_analysis.registry import TOOLS, Registry


def analyze(
    module: str, bundle: Path, prompt: str, index: int, output: Path, *, backend: str = "fixture"
) -> list[dict[str, Any]]:
    frame, truth = load_fixture(bundle, index)
    registry = (
        Registry(FixtureSegmenter(truth), FixturePlantIdentifier())
        if backend == "fixture"
        else Registry()
    )
    output.mkdir(parents=True, exist_ok=True)
    number = 0

    def write_mask(frame_index: int, mask: npt.NDArray[np.bool_]) -> str:
        nonlocal number
        ref = f"analysis/{module}-mask-{frame_index}-{number}.png"
        number += 1
        path = output / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        return ref

    location = json.loads((bundle / "bundle/manifest.json").read_text())["location"]
    return registry.run(module, {"prompt": prompt}, [frame], location, write_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=[tool["name"] for tool in TOOLS])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prompt", default="tree")
    parser.add_argument("--frame", type=int, default=2)
    parser.add_argument("--backend", choices=["fixture", "real"], default="fixture")
    parser.add_argument("--output", type=Path, default=Path(os.getenv("TMPDIR", ".tmp-test")))
    args = parser.parse_args()
    bundle = args.bundle
    if not bundle.exists() and not bundle.is_absolute():
        bundle = Path(__file__).resolve().parents[3] / bundle
    entities = analyze(
        args.module, bundle, args.prompt, args.frame, args.output, backend=args.backend
    )
    sys.stdout.write(json.dumps(entities, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
