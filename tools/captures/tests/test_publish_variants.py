"""`publish_variants`: a fill layer staged as a variant, the scan's other variants kept."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

import attach_sidecars
import publish_variants as pv
import teacher_fill as tf
from splat_render import Camera, Splats

TRANSFORM = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]
ASSET = {"assetId": "11111111-2222-3333-4444-555555555555", "url": "https://x.test/t/tileset.json"}


def _current(variants: dict | None = None) -> dict:
    extras: dict = {"instances": {"uri": "instances.json"}, "inferredLayers": [{"uri": "a"}]}
    if variants is not None:
        extras["variants"] = variants
    return {"root": {"transform": TRANSFORM, "extras": extras}}


def _layer_archive(folder: Path, filler: str) -> Path:
    """A real inferred layer (teacher_fill.package_inferred) as fill.yml uploads it."""
    measured = folder / "scan" / "tileset.json"
    measured.parent.mkdir(parents=True)
    measured.write_text(json.dumps({"root": {"transform": TRANSFORM}}))
    n = 50
    rng = np.random.default_rng(0)
    splats = Splats(
        rng.normal(size=(n, 3)),
        np.tile([1.0, 0, 0, 0], (n, 1)),
        np.full((n, 3), 0.05),
        rng.random((n, 3)),
        np.full(n, 0.8),
    )
    cams = [Camera.look_at([0.0, -3.0, 1.0], [0.0, 0.0, 0.0])]
    layer = folder / "layer" / "inferred"
    tf.package_inferred(splats, np.full(n, 0.5), cams, measured, layer, filler)
    archive = folder / "inferred.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(layer, arcname="inferred")
    return archive


def test_register_keeps_every_other_entry() -> None:
    variants = {
        "objects": [{"name": "ground-first"}],
        "fill": [{"name": "vace-1-3b", "label": "old"}, {"name": "other"}],
    }
    out = pv.register(variants, "fill", {"name": "vace-1-3b", "label": "new"})
    out = pv.register(out, "fill", {"name": "cosmos-p2-2b"})
    assert out["objects"] == [{"name": "ground-first"}]
    assert [e["name"] for e in out["fill"]] == ["vace-1-3b", "other", "cosmos-p2-2b"]
    assert out["fill"][0]["label"] == "new"
    assert variants["fill"][0]["label"] == "old"  # the input is not changed
    with pytest.raises(ValueError):
        pv.register(out, "fill", {"label": "no name"})


def test_build_stages_the_layer_and_refresh_keeps_newer_variants(tmp_path: Path) -> None:
    fill = tmp_path / "fill"
    job = "gen-spool-wan2-1-vace-1-3b"
    (fill / job).mkdir(parents=True)
    archive = _layer_archive(tmp_path / "make", "wan2.1-vace-1.3b")
    (fill / job / "inferred.tar.gz").write_bytes(archive.read_bytes())
    out = tmp_path / "out" / "spool"
    current = _current({"objects": [{"name": "ground-first"}]})
    manifest = pv.build("spool", fill, out, [job], asset=ASSET, current=current)
    assert (out / "variants" / "fill" / "vace-1-3b" / "tileset.json").exists()
    assert all(f.startswith("variants/fill/vace-1-3b/") for f in manifest["files"])
    variants = manifest["extras"]["variants"]
    assert set(manifest["extras"]) == {"variants"}  # today's layers untouched
    assert variants["objects"] == [{"name": "ground-first"}]
    (entry,) = variants["fill"]
    assert entry["name"] == "vace-1-3b" and entry["label"] == "Wan2.1-VACE 1.3B"
    layer = entry["inferredLayers"][0]
    assert layer["uri"] == "variants/fill/vace-1-3b/tileset.json"
    assert layer["evidence"]["kind"] == "inferred" and layer["evidence"]["gaussians"] == 50
    # Another bake-off published in between: refresh keeps it, and moves basedOn.
    newer = _current(
        {
            "objects": [{"name": "ground-first"}],
            "skins": [{"name": "freeform"}],
            "fill": [{"name": "other"}],
        }
    )
    moved = {**ASSET, "url": "https://x.test/g2/tileset.json"}
    again = pv.refresh(out, asset=moved, current=newer)
    assert again["basedOn"] == moved["url"]
    assert [e["name"] for e in again["extras"]["variants"]["fill"]] == ["other", "vace-1-3b"]
    assert again["extras"]["variants"]["skins"] == [{"name": "freeform"}]
    assert attach_sidecars.read_manifest(out)["extras"] == again["extras"]


def test_a_layer_in_another_frame_or_a_held_out_run_is_refused(tmp_path: Path) -> None:
    fill = tmp_path / "fill"
    job = "gen-spool-inpaint-lama"
    (fill / job).mkdir(parents=True)
    archive = _layer_archive(tmp_path / "make", "inpaint-lama")
    (fill / job / "inferred.tar.gz").write_bytes(archive.read_bytes())
    other = {"root": {"transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}}
    with pytest.raises(SystemExit):
        pv.build("spool", fill, tmp_path / "o" / "spool", [job], asset=ASSET, current=other)
    with pytest.raises(SystemExit):
        pv.variant_for("holdout-spool-inpaint-lama", "spool")
    with pytest.raises(SystemExit):
        pv.variant_for("gen-spool-something-else", "spool")
