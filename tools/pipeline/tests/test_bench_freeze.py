import io
from pathlib import Path
from typing import Any

import pytest

from experiments.bench.freeze import ReadOnlyR2, freeze_scene, resolve_run, select_generation
from experiments.bench.rehearsal import synthetic_scene
from experiments.bench.scene import verify

ROOT = "runs/ec644db5-ceaf-4c01-bf89-b261623327c2/"
GEN = "p0123456789abcdef"


def objects(prefix: str) -> list[dict[str, Any]]:
    return [
        {"Key": prefix + name, "Size": 1, "ETag": '"tag"'}
        for name in (
            "normalize/frames/a.jpg",
            "normalize/frames_meta.json",
            "pose/poses/images.bin",
        )
    ]


@pytest.mark.parametrize("generation", ["root", GEN])
def test_generation_one_populated_candidate(generation: str) -> None:
    prefix = ROOT if generation == "root" else ROOT + GEN + "/"
    selected, members = select_generation([*objects(prefix), {"Key": ROOT + "train/x"}], ROOT)
    assert selected == prefix
    assert {name for _, name in members} == {"frames/a.jpg", "frames_meta.json", "poses/images.bin"}


def test_generation_ambiguity_includes_partially_populated_candidates() -> None:
    listing = objects(ROOT + GEN + "/") + objects(ROOT)[:1]
    with pytest.raises(ValueError, match="exactly one populated"):
        select_generation(listing, ROOT)
    assert select_generation(listing, ROOT, GEN)[0] == ROOT + GEN + "/"
    with pytest.raises(ValueError, match="missing required"):
        select_generation(listing, ROOT, "root")
    with pytest.raises(ValueError, match="exactly one populated"):
        select_generation([], ROOT)


def test_generation_rejects_traversal() -> None:
    listing = objects(ROOT)
    listing.append({"Key": ROOT + "normalize/frames/../../bad"})
    with pytest.raises(ValueError, match="unsafe"):
        select_generation(listing, ROOT)


class Store:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list", kwargs))
        listing = [
            {"Key": k, "Size": len(v), "ETag": '"fixed"'}
            for k, v in sorted(self.payloads.items())
            if k.startswith(kwargs["Prefix"])
        ]
        offset = int(kwargs.get("ContinuationToken", 0))
        return {
            "Contents": listing[offset : offset + 3],
            "IsTruncated": offset + 3 < len(listing),
            "NextContinuationToken": str(offset + 3),
        }

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["IfMatch"] == '"fixed"'
        self.calls.append(("get", kwargs))
        return {"Body": io.BytesIO(self.payloads[kwargs["Key"]])}


def test_readonly_wrapper_has_only_list_and_get_and_resolves_prefix() -> None:
    client = Store({ROOT + "normalize/frames/a": b"x"})
    store = ReadOnlyR2(client, "private")
    assert {n for n in dir(store) if not n.startswith("_")} == {"list", "get"}
    assert resolve_run(store, "ec644db5") == ROOT
    client.payloads[ROOT.replace("ceaf", "ffff") + "normalize/frames/a"] = b"x"
    with pytest.raises(ValueError, match="2 runs"):
        resolve_run(store, "ec644db5")
    with pytest.raises(ValueError, match="UUID"):
        resolve_run(store, "../bad")


def test_freeze_copies_only_allowed_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    synthetic_scene(source, "spool", "commit")
    payloads = {}
    for directory, remote in (("frames", "normalize/frames"), ("poses", "pose/poses")):
        for file in (source / directory).iterdir():
            payloads[ROOT + f"{remote}/{file.name}"] = file.read_bytes()
    payloads[ROOT + "normalize/frames_meta.json"] = (source / "frames_meta.json").read_bytes()
    payloads[ROOT + "train/trained.ply"] = b"never copied"
    client = Store(payloads)
    store = ReadOnlyR2(client, "private")
    target = tmp_path / "scenes"
    manifest = freeze_scene(store, target, "spool", "ec644db5", "bench-commit")
    assert verify(target / "spool")[0] == manifest
    assert manifest["benchCommit"] == "bench-commit"
    assert manifest["sourcePrefix"] == ROOT
    assert len(manifest["files"]) == 20
    assert not (target / "spool/trained.ply").exists()
    assert all(call[0] in {"list", "get"} for call in client.calls)
    with pytest.raises(ValueError, match="already frozen"):
        freeze_scene(store, target, "spool", "ec644db5", "commit")
    assert (
        freeze_scene(store, target, "spool", "ec644db5", "new", refreeze=True)["benchCommit"]
        == "new"
    )
    before = (target / "spool/manifest.json").read_bytes()
    del client.payloads[ROOT + "pose/poses/cameras.bin"]
    with pytest.raises(FileNotFoundError):
        freeze_scene(store, target, "spool", "ec644db5", "broken", refreeze=True)
    assert (target / "spool/manifest.json").read_bytes() == before
