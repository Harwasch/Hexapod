"""The worker's per-tile position checksum is the one the bindings were written with.

`instances.json`, `skin.json`, `plants.json` and `rig.json` key tiles by
`synthetic_tree.checksum_positions(rig_tiles.tile_positions(glb))` (tools/captures), and the
viewer refuses a tile whose checksum its binding does not list (`checksumPositions` in
packages/world). `app/worker/positions.py` transcribes that function so a republish can tell
whether a binding still holds for new tiles; if the two ever drifted, the worker would keep
objects on splats the viewer then refuses, or drop objects that still hold. These hold it to
the same vectors both other languages read, and to checksums tools/captures stamped.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from app.config import REPO_ROOT
from app.services import sidecars
from app.worker import positions
from tests.sidecar_fixtures import TREE_LOD, TREE_SH, fixture_scan, splat_glb

VECTORS = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source" / "checksum_vectors.json"


def test_the_checksum_agrees_with_the_vectors_both_other_languages_read() -> None:
    cases = json.loads(VECTORS.read_text())["cases"]
    assert cases
    for case in cases:
        array = np.frombuffer(bytes.fromhex(case["positionsHex"]), dtype="<f4").reshape(-1, 3)
        assert positions.checksum_positions(array) == case["checksum"], case["name"]


def test_every_fixture_tile_hashes_to_what_tools_captures_stamped_on_its_rig() -> None:
    rig = json.loads((TREE_LOD / "rig.json").read_text())
    _, tiles = fixture_scan(TREE_LOD)
    assert sorted({positions.tile_checksum(data) for data in tiles.values()}) == sorted(
        rig["tileChecksums"]
    )


def test_a_repack_with_harmonics_rewrites_every_tile_and_moves_no_position() -> None:
    """The case the byte fingerprint got wrong: another `--sh-degree`."""
    _, lod = fixture_scan(TREE_LOD)
    _, sh = fixture_scan(TREE_SH)
    assert set(lod) == set(sh)
    for uri in lod:
        assert lod[uri] != sh[uri], uri
        assert positions.tile_checksum(lod[uri]) == positions.tile_checksum(sh[uri]), uri


def test_versions_two_and_three_read_the_same_positions() -> None:
    points = [[0.5, -1.25, 3.0], [-2.0, 0.000244140625, 7.75]]
    two = splat_glb(points, version=2)
    three = splat_glb(points, version=3, fill=9, sh_degree=1)
    assert two != three
    assert positions.tile_checksum(two) == positions.tile_checksum(three)
    assert positions.tile_positions(two).tolist() == points
    moved = splat_glb([[0.5, -1.25, 3.0], [-2.0, 0.00048828125, 7.75]])
    assert positions.tile_checksum(moved) != positions.tile_checksum(two)


def test_a_position_of_zero_hashes_as_positive_zero() -> None:
    """Negative zero has its own bytes; SPZ's integers cannot hold it, nor can the viewer's
    un-bake, so a tile never hashes with it."""
    glb = splat_glb([[0.0, -0.0, 1.0]])
    assert positions.tile_checksum(glb) == positions.checksum_positions(
        np.array([[0.0, 0.0, 1.0]], dtype="<f4")
    )


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not a glb at all, but long enough",
        splat_glb([[1.0, 2.0, 3.0]])[:-40],
        b"glTF" + b"\0" * 30,
    ],
)
def test_bytes_that_are_not_a_splat_tile_are_refused(data: bytes) -> None:
    with pytest.raises(positions.TileFormatError):
        positions.tile_checksum(data)


def test_a_stream_that_does_not_inflate_is_refused() -> None:
    stream = bytearray(gzip.compress(bytes(range(256)) * 8, mtime=0))
    stream[20:30] = b"\xff" * 10
    with pytest.raises(positions.TileFormatError):
        positions.spz_positions(bytes(stream))


def test_a_binding_lists_the_checksums_every_one_of_its_files_lists() -> None:
    rig = sidecars.RIG
    both = sidecars.bound_checksums(
        rig,
        {
            "rig.json": json.dumps({"tileChecksums": ["a", "b", "c"]}).encode(),
            "plants.json": json.dumps({"tiles": {"b": [0, 1], "c": [0, 2], "d": []}}).encode(),
        },
    )
    assert both == frozenset({"b", "c"})
    assert sidecars.bound_checksums(
        sidecars.INSTANCES, {"instances.json": b'{"tiles": {"x": [1, 5]}}'}
    ) == frozenset({"x"})
    # Nothing to vouch with: no file, a single-object rig, a file that is not JSON.
    assert sidecars.bound_checksums(sidecars.INSTANCES, {}) is None
    assert sidecars.bound_checksums(rig, {"rig.json": b'{"canonicalChecksum": "a"}'}) is None
    assert sidecars.bound_checksums(sidecars.SKIN, {"skin.json": b"{"}) is None
