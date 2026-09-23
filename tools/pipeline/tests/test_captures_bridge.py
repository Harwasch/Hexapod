"""The seam to tools/captures, and the proof that the stub honours the same contract.

`package: splat_tiles` is stubbed at A6 and real at A8. What must not drift in between is
the *shape* of what it writes, because that is what the console loads and what `register`
consumes. So this test runs the sibling project's real `convert()` on a generated splat and
asserts the file names it writes are exactly the ones the `splat` artifact declares -- the
same declaration StubRunner fabricates from.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from captures_bridge import CAPTURES_DIR, splat_tiles_convert
from conftest import seeded_workdir
from executor import execute
from recipe import load_recipe
from runners import RunnerSet
from stages import SPLAT_TILES

# fmt: off
PLY_FIELDS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]
# fmt: on


def write_ply(path: Path, count: int = 64) -> Path:
    """A minimal binary little-endian 3DGS PLY, the form splat_tiles.read_ply accepts."""
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in PLY_FIELDS]
    header.append("end_header")
    body = bytearray()
    for index in range(count):
        offset = index / count
        values = [
            offset,  # x
            offset * 2.0,  # y
            offset * 0.5,  # z
            0.3,
            0.4,
            0.5,  # f_dc_*
            3.0,  # opacity logit -> sigmoid ~0.95, well above opacity_min
            -2.0,
            -2.0,
            -2.0,  # log scales
            1.0,
            0.0,
            0.0,
            0.0,  # rotation
        ]
        body += struct.pack("<14f", *values)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + bytes(body))
    return path


def test_the_sibling_project_is_importable_without_being_copied() -> None:
    assert (CAPTURES_DIR / "splat_tiles.py").is_file()


def test_the_real_packer_writes_exactly_what_the_artifact_declares(tmp_path: Path) -> None:
    ply = write_ply(tmp_path / "splat.ply")
    out = tmp_path / "splat"

    stats = splat_tiles_convert(ply, out, lat=46.84, lon=-91.99, height=0.0)

    assert stats["gaussians"] > 0
    written = sorted(path.name for path in out.iterdir())
    # This is the assertion that keeps A6's stub and A8's real stage honest.
    assert written == sorted(SPLAT_TILES.required_members)


def test_the_stub_writes_the_same_file_names_as_the_real_packer(tmp_path: Path) -> None:
    real = tmp_path / "real"
    splat_tiles_convert(write_ply(tmp_path / "splat.ply"), real, 46.84, -91.99, 0.0)
    workdir = seeded_workdir(tmp_path / "run")

    execute(load_recipe("splat-ingest"), workdir, RunnerSet.stubbed())

    stubbed = workdir.out_dir("package") / SPLAT_TILES.name
    assert sorted(p.name for p in stubbed.iterdir()) == sorted(p.name for p in real.iterdir())


def test_the_real_packer_is_byte_stable(tmp_path: Path) -> None:
    """The sibling's determinism is load-bearing here too: CI gates it byte-for-byte."""
    ply = write_ply(tmp_path / "splat.ply")
    first, second = tmp_path / "a", tmp_path / "b"
    splat_tiles_convert(ply, first, 46.84, -91.99, 0.0)
    splat_tiles_convert(ply, second, 46.84, -91.99, 0.0)

    for name in SPLAT_TILES.required_members:
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize("member", SPLAT_TILES.required_members)
def test_required_members_are_declared_on_the_artifact(member: str) -> None:
    assert member in SPLAT_TILES.stub_members
