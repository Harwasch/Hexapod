"""`publish_variants`: fill layers staged as variants and registered, and variants withdrawn,
only through the shared attach (`register` / `withdraw`): the request never carries a
`variants` value of its own, and every other system's entry survives the attach's merge."""

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
OTHERS = {
    "objects": [{"name": "ground-first"}, {"name": "feature-fields"}],
    "skins": [{"name": "freeform"}],
}


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


def _fill(tmp_path: Path, jobs: list[str]) -> Path:
    fill = tmp_path / "fill"
    for k, job in enumerate(jobs):
        (fill / job).mkdir(parents=True)
        archive = _layer_archive(tmp_path / f"make{k}", job)
        (fill / job / "inferred.tar.gz").write_bytes(archive.read_bytes())
    return fill


class _Bucket:
    def __init__(self) -> None:
        self.uploaded: list[str] = []

    def list_objects_v2(self, **kw):
        return {}

    def delete_objects(self, **kw) -> None:
        pass

    def upload_file(self, path, bucket, key, ExtraArgs=None) -> None:
        self.uploaded.append(key)


def _attach(out: Path, live: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """`attach_sidecars.attach` on `out` against a stub API whose tileset is `live`: the body
    it posts."""
    sent: list[dict] = []
    monkeypatch.setattr(attach_sidecars, "resolve_asset", lambda asset_id, api=None: dict(ASSET))
    monkeypatch.setattr(attach_sidecars, "get_json", lambda url: live)

    def post(url, body, token):
        sent.append(json.loads(json.dumps(body)))
        return 200, {"generation": "g", "url": "u", "previousUrl": "p", "staged": body["files"]}

    monkeypatch.setattr(attach_sidecars, "post", post)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    attach_sidecars.attach(out, s3=_Bucket(), bucket="b", write_token="t")
    (body,) = sent
    return body


def test_v2_layers_register_through_the_shared_attach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = ["anchor-spool-anchor-refs", "anchor-spool-anchor-norefs", "anchor-spool-anchor-vace"]
    fill = _fill(tmp_path, jobs)
    out = tmp_path / "out" / "spool"
    round1 = [{"name": n} for n in ("vace-1-3b", "wan22-5b", "cosmos-p2-2b", "lama-baseline")]
    current = _current({**OTHERS, "fill": round1})
    manifest = pv.build("spool", fill, out, jobs, asset=ASSET, current=current)
    # The request carries no variants value of its own: only the entries to register.
    on_disk = json.loads((out / attach_sidecars.MANIFEST).read_text())
    assert "variants" not in on_disk["extras"] and on_disk["extras"] == {}
    names = [r["entry"]["name"] for r in on_disk["register"]]
    assert names == ["anchor-refs", "anchor-norefs", "anchor-vace"]
    entry = on_disk["register"][0]["entry"]
    assert (
        entry["look"]
        == "Set Inferred to Highlight; orbit to look down on the spool's top. Purple is generated."
    )
    assert entry["inferredLayers"][0]["uri"] == "variants/fill/anchor-refs/tileset.json"
    assert entry["inferredLayers"][0]["evidence"]["kind"] == "inferred"
    assert all(f.startswith("variants/fill/anchor-") for f in on_disk["files"])
    assert manifest["preflight"]["fill"][:4] == [e["name"] for e in round1]
    # Meanwhile another bake-off attached a skin and a fill variant: the attach keeps them.
    live = _current(
        {
            **OTHERS,
            "skins": [{"name": "freeform"}, {"name": "tetfem-stiff"}],
            "fill": [*round1, {"name": "someone-else"}],
        }
    )
    body = _attach(out, live, monkeypatch)
    variants = body["extras"]["variants"]
    assert variants["objects"] == OTHERS["objects"]
    assert variants["skins"] == [{"name": "freeform"}, {"name": "tetfem-stiff"}]
    assert [e["name"] for e in variants["fill"]] == [
        *(e["name"] for e in round1),
        "someone-else",
        "anchor-refs",
        "anchor-norefs",
        "anchor-vace",
    ]
    assert "inferredLayers" not in body["extras"]  # today's layers are not touched


def test_round1_variants_are_withdrawn_and_the_rest_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fill_now = [
        {"name": n}
        for n in ("vace-1-3b", "wan22-5b", "cosmos-p2-2b", "lama-baseline", "anchor-refs")
    ]
    current = _current({**OTHERS, "fill": fill_now})
    out = tmp_path / "out" / "spool"
    names = ["wan22-5b", "cosmos-p2-2b", "lama-baseline"]
    manifest = pv.withdraw("spool", out, names, asset=ASSET, current=current)
    on_disk = json.loads((out / attach_sidecars.MANIFEST).read_text())
    assert on_disk["files"] == [] and on_disk["extras"] == {}
    assert [w["name"] for w in on_disk["withdraw"]] == names
    assert manifest["preflight"]["fill"] == ["vace-1-3b", "anchor-refs"]
    body = _attach(out, current, monkeypatch)
    variants = body["extras"]["variants"]
    assert [e["name"] for e in variants["fill"]] == ["vace-1-3b", "anchor-refs"]
    assert variants["objects"] == OTHERS["objects"] and variants["skins"] == OTHERS["skins"]
    with pytest.raises(SystemExit):
        pv.withdraw("spool", tmp_path / "x", ["not-there"], asset=ASSET, current=current)


def test_preflight_refuses_a_merge_that_loses_anything_else() -> None:
    current = {**OTHERS, "fill": [{"name": "a"}]}
    _, summary = pv.preflight(current, [("with", "fill", {"name": "b"})])
    assert summary == {
        "objects": ["ground-first", "feature-fields"],
        "skins": ["freeform"],
        "fill": ["a", "b"],
    }
    with pytest.raises(SystemExit):  # no variants at all would be left: never send none
        pv.preflight({"fill": [{"name": "a"}]}, [("without", "fill", "a")])
    assert pv.preflight(current, [("without", "fill", "a")])[1].get("fill") is None


def test_a_layer_in_another_frame_or_a_held_out_run_is_refused(tmp_path: Path) -> None:
    fill = _fill(tmp_path, ["anchor-spool-anchor-refs"])
    other = {"root": {"transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}}
    with pytest.raises(SystemExit):
        pv.build(
            "spool",
            fill,
            tmp_path / "o" / "spool",
            ["anchor-spool-anchor-refs"],
            asset=ASSET,
            current=other,
        )
    for job in ("holdout-spool-inpaint-lama", "leaveout-spool-anchor-refs", "anchor-spool-nope"):
        with pytest.raises(SystemExit):
            pv.variant_for(job, "spool")
    assert pv.variant_for("gen-spool-wan2-1-vace-1-3b", "spool")[1]["name"] == "vace-1-3b"


def test_a_rebuilt_surface_publishes_what_it_supersedes(tmp_path: Path) -> None:
    archive = _layer_archive(tmp_path / "make", "anchor-spool-anchor-refs")
    layer = tmp_path / "layer"
    with tarfile.open(archive) as tar:
        tar.extractall(layer, filter="data")
    folder = layer / "inferred"
    (folder / "supersedes.json").write_text(json.dumps({"tiles": {"abc": [0, 3, 1, 2]}}))
    tileset = json.loads((folder / "tileset.json").read_text())
    tileset["root"]["extras"]["supersedes"] = {"uri": "supersedes.json", "superseded": 2}
    (folder / "tileset.json").write_text(json.dumps(tileset))
    job = tmp_path / "fill" / "anchor-spool-anchor-refs"
    job.mkdir(parents=True)
    with tarfile.open(job / "inferred.tar.gz", "w:gz") as tar:
        tar.add(folder, arcname="inferred")
    out = tmp_path / "out"
    jobs = ["anchor-spool-anchor-refs"]
    pv.build("spool", tmp_path / "fill", out, jobs, asset=ASSET, current=_current(OTHERS))
    on_disk = json.loads((out / attach_sidecars.MANIFEST).read_text())
    entry = on_disk["register"][0]["entry"]
    assert entry["supersedes"] == "variants/fill/anchor-refs/supersedes.json"
    assert "variants/fill/anchor-refs/supersedes.json" in on_disk["files"]


def test_a_variant_of_two_layers_and_the_solidity_variants(tmp_path: Path) -> None:
    fill = _fill(
        tmp_path,
        ["anchor-spool-anchor-refs", "anchor-spool-anchor-shape", "anchor-spool-anchor-refs-full"],
    )
    # The top's layer supersedes measured splats; the shape's does not.
    top = tmp_path / "top"
    with tarfile.open(fill / "anchor-spool-anchor-refs" / "inferred.tar.gz") as tar:
        tar.extractall(top, filter="data")
    folder = top / "inferred"
    (folder / "supersedes.json").write_text(json.dumps({"tiles": {}}))
    tileset = json.loads((folder / "tileset.json").read_text())
    tileset["root"]["extras"]["supersedes"] = {"uri": "supersedes.json", "superseded": 0}
    (folder / "tileset.json").write_text(json.dumps(tileset))
    with tarfile.open(fill / "anchor-spool-anchor-refs" / "inferred.tar.gz", "w:gz") as tar:
        tar.add(folder, arcname="inferred")
    out = tmp_path / "out"
    jobs = ["anchor-spool-anchor-refs+anchor-spool-anchor-shape", "anchor-spool-anchor-refs-full"]
    pv.build("spool", fill, out, jobs, asset=ASSET, current=_current(OTHERS))
    on_disk = json.loads((out / attach_sidecars.MANIFEST).read_text())
    both, solid = (r["entry"] for r in on_disk["register"])
    assert both["name"] == "refs-shape" and "under the spool's top flange" in both["look"]
    assert [layer["uri"] for layer in both["inferredLayers"]] == [
        "variants/fill/refs-shape/top/tileset.json",
        "variants/fill/refs-shape/shape/tileset.json",
    ]
    assert both["supersedes"] == "variants/fill/refs-shape/top/supersedes.json"
    assert {
        "variants/fill/refs-shape/top/supersedes.json",
        "variants/fill/refs-shape/shape/tileset.json",
    } <= set(on_disk["files"])
    assert solid["name"] == "anchor-refs-solid" and "supersedes" not in solid
    # A layer that only goes with another is not a variant of its own.
    with pytest.raises(SystemExit):
        pv.variant_for("anchor-spool-anchor-shape", "spool")
    with pytest.raises(SystemExit):
        pv.variant_for("anchor-spool-anchor-norefs+anchor-spool-anchor-shape", "spool")


def test_object_layers_register_with_where_to_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = ["objects-pumpkin-objects-trellis"]
    fill = _fill(tmp_path, jobs)
    out = tmp_path / "out" / "pumpkin"
    v2 = [{"name": n} for n in ("vace-1-3b", "anchor-refs", "anchor-norefs", "anchor-vace")]
    pv.build("pumpkin", fill, out, jobs, asset=ASSET, current=_current({**OTHERS, "fill": v2}))
    on_disk = json.loads((out / attach_sidecars.MANIFEST).read_text())
    assert "variants" not in on_disk["extras"]
    (item,) = on_disk["register"]
    assert item["entry"]["name"] == "objects-trellis"
    assert "under the pumpkins" in item["entry"]["look"]
    assert item["entry"]["inferredLayers"][0]["uri"] == "variants/fill/objects-trellis/tileset.json"
    body = _attach(out, _current({**OTHERS, "fill": v2}), monkeypatch)
    variants = body["extras"]["variants"]
    assert variants["objects"] == OTHERS["objects"] and variants["skins"] == OTHERS["skins"]
    assert [e["name"] for e in variants["fill"]] == [*(e["name"] for e in v2), "objects-trellis"]
