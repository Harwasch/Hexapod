"""The shipped SH degree is a measured choice (photo-reconstruct.yaml's `train` comment)."""

from pathlib import Path

import yaml

RECIPES = Path(__file__).resolve().parents[1] / "recipes"


def _params(recipe: str, stage: str) -> dict[str, object]:
    document = yaml.safe_load((RECIPES / recipe).read_text())
    return next(s for s in document["stages"] if s["id"] == stage)["params"]


def test_photo_reconstruct_ships_every_trained_band() -> None:
    assert _params("photo-reconstruct.yaml", "train")["ship_sh_degree"] == 3


def test_splat_ingest_ships_every_uploaded_band() -> None:
    assert _params("splat-ingest.yaml", "normalize")["ship_sh_degree"] == 3
