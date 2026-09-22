"""SPZ in: the format Scaniverse exports and the format this project already writes.

A0 #4 measured all of this before it was built, and the numbers are asserted here rather
than quoted: position error exactly 0 round-tripping the committed fixture (because
`synthetic_tree.py` snaps positions to the 1/4096 grid), alpha 0 and 255 clamped before
the logit, and `unpack -> pack` **not** byte-idempotent -- which is why no SPZ round trip
may sit inside the fixture byte-identity gate, and why this file generates its `.spz`
rather than committing one.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import gaussians
from captures_bridge import SplatFormatError, pack_spz, read_ply, sigmoid, unpack_spz
from conftest import FIXTURE_PLY


def _spz_of(path: Path) -> bytes:
    """The fixture packed to SPZ exactly as `splat_tiles.convert` packs it."""
    data = read_ply(path)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    sh0 = np.stack([data[f"f_dc_{i}"] for i in range(3)], axis=1)
    scales = np.stack([data[f"scale_{i}"] for i in range(3)], axis=1)
    quat_wxyz = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1)
    return bytes(pack_spz(xyz, sh0, data["opacity"], scales, quat_wxyz[:, [1, 2, 3, 0]]))


@pytest.fixture(scope="module")
def spz_bytes() -> bytes:
    return _spz_of(FIXTURE_PLY)


def test_an_spz_ingests_with_the_positions_it_was_packed_from(
    spz_bytes: bytes, tmp_path: Path
) -> None:
    source = read_ply(FIXTURE_PLY)
    path = tmp_path / "scan.spz"
    path.write_bytes(spz_bytes)

    splat = gaussians.read_splat(path)

    assert splat.source_format == "spz"
    assert splat.count == 12_000
    # Exactly zero, not approximately: 24-bit fixed point at 1/4096 is lossless for a
    # capture already on that grid, which is what makes this the cheap lane.
    original = np.stack([source["x"], source["y"], source["z"]], axis=1)
    assert float(np.abs(splat.xyz - original).max()) == 0.0


def test_the_spz_carries_the_fixtures_colours_and_rotations(spz_bytes: bytes) -> None:
    source = read_ply(FIXTURE_PLY)
    back = unpack_spz(spz_bytes)

    # Colour survives to a quantisation step: 1/(0.15 * 255) in SH DC units.
    assert float(np.abs(back["f_dc_0"] - source["f_dc_0"]).max()) < 0.03
    # The packed quaternion is w-positive and normalised; w is reconstructed, not stored.
    norm = np.linalg.norm(np.stack([back[f"rot_{i}"] for i in range(4)], axis=1), axis=1)
    assert float(np.abs(norm - 1.0).max()) < 0.02
    assert bool((back["rot_0"] >= 0.0).all())


def test_alpha_zero_and_two_hundred_and_fifty_five_clamp_instead_of_going_infinite() -> None:
    """The one genuine ambiguity A0 found: 0 and 255 are -inf and +inf as a logit."""
    count = 3
    logits = np.array([-40.0, 40.0, 0.0], dtype=np.float32)
    flat = np.zeros((count, 3), dtype=np.float32)
    rotation = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1))

    back = unpack_spz(pack_spz(flat, flat, logits, flat, rotation))

    assert bool(np.isfinite(back["opacity"]).all())
    # And the clamp is chosen so the bytes survive the trip back out again.
    assert np.round(sigmoid(back["opacity"]) * 255).tolist() == [0.0, 255.0, 128.0]


def test_unpack_then_pack_is_not_byte_identical(spz_bytes: bytes) -> None:
    """A0 #4, asserted rather than quoted -- and the reason for this file's docstring.

    One byte in 228,016 differs, at a rotation rounding boundary. It is a rounding tie,
    not a bug, and it is exactly why no SPZ round trip belongs inside
    `git diff --exit-code -- data/tiles/synthetic-tree`.
    """
    back = unpack_spz(spz_bytes)
    xyz = np.stack([back["x"], back["y"], back["z"]], axis=1)
    sh0 = np.stack([back[f"f_dc_{i}"] for i in range(3)], axis=1)
    scales = np.stack([back[f"scale_{i}"] for i in range(3)], axis=1)
    quat = np.stack([back["rot_1"], back["rot_2"], back["rot_3"], back["rot_0"]], axis=1)

    again = pack_spz(xyz, sh0, back["opacity"], scales, quat)

    first, second = gzip.decompress(spz_bytes), gzip.decompress(again)
    assert len(first) == len(second) == 16 + 19 * 12_000 == 228_016
    differing = int((np.frombuffer(first, np.uint8) != np.frombuffer(second, np.uint8)).sum())
    assert differing > 0, "if this ever becomes 0, say so -- but do not gate the fixture on it"
    # A0 measured exactly one. A handful would still be rounding ties; a thousand would be
    # a changed constant somewhere.
    assert differing <= 8


def test_a_file_that_is_not_an_spz_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "scan.spz"
    path.write_bytes(gzip.compress(b"\x00" * 64))

    with pytest.raises(SplatFormatError, match="not an SPZ file"):
        gaussians.read_splat(path)


def test_a_truncated_spz_says_how_many_bytes_it_needed(tmp_path: Path) -> None:
    raw = gzip.decompress(_spz_of(FIXTURE_PLY))
    path = tmp_path / "short.spz"
    path.write_bytes(gzip.compress(raw[: 16 + 19 * 100]))

    with pytest.raises(SplatFormatError, match="SPZ is truncated"):
        gaussians.read_splat(path)


def test_an_ingested_spz_writes_a_canonical_ply_that_reads_back(
    spz_bytes: bytes, tmp_path: Path
) -> None:
    """The lane's actual shape: `.spz` in, `canonical.ply` out, and `package` reads that."""
    (tmp_path / "scan.spz").write_bytes(spz_bytes)
    splat = gaussians.read_splat(tmp_path / "scan.spz")

    gaussians.write_ply(tmp_path / "canonical.ply", splat.columns)
    again = gaussians.read_splat(tmp_path / "canonical.ply")

    assert again.count == splat.count
    for name in gaussians.CANONICAL_PROPERTIES:
        assert again.columns[name].tolist() == splat.columns[name].tolist()
