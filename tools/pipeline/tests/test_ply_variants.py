"""Every PLY shape A0 enumerated, and what each one does now.

A0 #3 is the whole point of this file: the reader's type map and its header parser had to
be fixed *together*, because fixing the type map alone turns a loud `KeyError: 'uint'`
into the silent one -- a mesh PLY whose trailing `element face` leaked into the vertex
dtype and came back with `|x| max = 1.7e38`. So each variant below is asserted to do one
of two things: ingest correctly, or refuse with a message that names what is wrong.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import gaussians
from captures_bridge import SplatFormatError
from conftest import FIXTURE_PLY

#: The fourteen properties every 3DGS PLY carries, and the values the builders write.
CANONICAL = gaussians.CANONICAL_PROPERTIES


def _rows(count: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    values = {name: rng.normal(size=count).astype(np.float32) for name in CANONICAL}
    values["x"] = np.array([0.0, 1.0, -1.0, 0.5][:count], dtype=np.float32)
    values["y"] = np.array([0.0, 2.0, -2.0, 0.25][:count], dtype=np.float32)
    values["z"] = np.array([0.0, 3.0, 0.5, 1.0][:count], dtype=np.float32)
    values["opacity"] = np.full(count, 2.0, dtype=np.float32)
    values["rot_0"] = np.ones(count, dtype=np.float32)
    for axis in (1, 2, 3):
        values[f"rot_{axis}"] = np.zeros(count, dtype=np.float32)
    return values


def _write(
    path: Path,
    *,
    header: list[str],
    body: bytes,
) -> Path:
    path.write_bytes(("\n".join(["ply", *header, "end_header"]) + "\n").encode("ascii") + body)
    return path


def _canonical_ply(
    path: Path, *, order: str = "<", byte_order: str = "binary_little_endian"
) -> Path:
    values = _rows()
    dtype = np.dtype([(name, order + "f4") for name in CANONICAL])
    record = np.empty(values["x"].shape[0], dtype=dtype)
    for name in CANONICAL:
        record[name] = values[name]
    return _write(
        path,
        header=[
            f"format {byte_order} 1.0",
            f"element vertex {record.shape[0]}",
            *[f"property float {name}" for name in CANONICAL],
        ],
        body=record.tobytes(),
    )


# --------------------------------------------------------------------------------------
# The shapes that ingest
# --------------------------------------------------------------------------------------


def test_the_committed_fixture_reads_as_twelve_thousand_gaussians() -> None:
    splat = gaussians.read_splat(FIXTURE_PLY)

    assert splat.count == 12_000
    assert splat.source_format == "ply"
    assert splat.dropped == ()
    assert splat.non_finite == 0
    low, high = splat.bbox()
    assert -3.0 < low[0] < 0.0 and 0.0 < high[2] < 10.0


def test_a_canonical_ply_ingests(tmp_path: Path) -> None:
    splat = gaussians.read_splat(_canonical_ply(tmp_path / "canonical.ply"))

    assert splat.count == 4
    assert list(splat.columns) == list(CANONICAL)
    assert splat.columns["x"].tolist() == [0.0, 1.0, -1.0, 0.5]


def test_spherical_harmonic_bands_are_read_and_deliberately_dropped(tmp_path: Path) -> None:
    """`f_rest_*` is 45 more floats per gaussian that `splat_tiles.convert` never reads.

    Carrying them into `canonical.ply` would quadruple it for nothing, so they are
    dropped -- and `source_meta.json` says so by name rather than silently.
    """
    values = _rows()
    extra = [f"f_rest_{i}" for i in range(45)]
    names = [*CANONICAL, *extra]
    dtype = np.dtype([(name, "<f4") for name in names])
    record = np.zeros(4, dtype=dtype)
    for name in CANONICAL:
        record[name] = values[name]
    path = _write(
        tmp_path / "sh.ply",
        header=[
            "format binary_little_endian 1.0",
            "element vertex 4",
            *[f"property float {name}" for name in names],
        ],
        body=record.tobytes(),
    )

    splat = gaussians.read_splat(path)

    assert splat.properties_in == tuple(names)
    assert splat.dropped == tuple(extra)
    assert splat.columns["x"].tolist() == [0.0, 1.0, -1.0, 0.5]


def test_a_trailing_element_face_does_not_leak_into_the_vertex_dtype(tmp_path: Path) -> None:
    """A0 #3's silent failure: `|x| max = 1.7e38`, dying twenty lines later in `convert`.

    The old parser kept collecting `property` lines after the second `element`, so
    `vertex_indices` joined the vertex dtype and every gaussian was read at the wrong
    stride. Same bytes, two faces appended; the vertices must come back unchanged.
    """
    values = _rows()
    dtype = np.dtype([(name, "<f4") for name in CANONICAL])
    record = np.empty(4, dtype=dtype)
    for name in CANONICAL:
        record[name] = values[name]
    faces = b"".join(struct.pack("<Biii", 3, 0, 1, 2) for _ in range(2))
    path = _write(
        tmp_path / "mesh.ply",
        header=[
            "format binary_little_endian 1.0",
            "element vertex 4",
            *[f"property float {name}" for name in CANONICAL],
            "element face 2",
            "property list uchar int vertex_indices",
        ],
        body=record.tobytes() + faces,
    )

    splat = gaussians.read_splat(path)

    assert splat.count == 4
    assert splat.columns["x"].tolist() == [0.0, 1.0, -1.0, 0.5]
    assert float(np.abs(splat.xyz).max()) < 10.0


def test_unusual_scalar_types_are_read_rather_than_raising_a_key_error(tmp_path: Path) -> None:
    """`uchar`, `ushort`, `uint`, `short`, `int8` and `double` are all legal PLY.

    The old type map knew five of the sixteen names, so anything else was an unhandled
    `KeyError` from inside the reader -- no file named, no property named.
    """
    dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f8"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
            ("quality", "u1"),
            ("cluster", "<u4"),
            ("band", "<i2"),
            ("flag", "i1"),
            ("pad", "<u2"),
        ]
    )
    record = np.zeros(3, dtype=dtype)
    record["x"] = [1.0, 2.0, 3.0]
    record["rot_0"] = 1.0
    record["cluster"] = [1, 4_000_000_000, 7]
    types = {
        "f8": "double",
        "f4": "float",
        "u1": "uchar",
        "u4": "uint",
        "i2": "short",
        "i1": "int8",
        "u2": "ushort",
    }
    properties = [
        f"property {types[dtype.fields[name][0].str[1:]]} {name}"  # type: ignore[index]
        for name in dtype.names or ()
    ]
    path = _write(
        tmp_path / "typed.ply",
        header=["format binary_little_endian 1.0", "element vertex 3", *properties],
        body=record.tobytes(),
    )

    splat = gaussians.read_splat(path)

    assert splat.columns["x"].tolist() == [1.0, 2.0, 3.0]
    assert "quality" in splat.properties_in
    assert set(splat.dropped) == {"quality", "cluster", "band", "flag", "pad"}


def test_big_endian_reads_the_same_values_as_little_endian(tmp_path: Path) -> None:
    little = gaussians.read_splat(_canonical_ply(tmp_path / "le.ply"))
    big = gaussians.read_splat(
        _canonical_ply(tmp_path / "be.ply", order=">", byte_order="binary_big_endian")
    )

    assert big.count == little.count
    for name in CANONICAL:
        assert big.columns[name].tolist() == little.columns[name].tolist()


def test_vertex_colours_become_spherical_harmonic_dc_terms(tmp_path: Path) -> None:
    """Some exporters write `red/green/blue/alpha` instead of `f_dc_*`/`opacity`."""
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    colours = ["red", "green", "blue", "alpha"]
    dtype = np.dtype([(name, "<f4") for name in names] + [(c, "u1") for c in colours])
    record = np.zeros(2, dtype=dtype)
    record["rot_0"] = 1.0
    record["red"] = [255, 0]
    record["green"] = [128, 0]
    record["blue"] = [0, 255]
    record["alpha"] = [255, 0]
    path = _write(
        tmp_path / "coloured.ply",
        header=[
            "format binary_little_endian 1.0",
            "element vertex 2",
            *[f"property float {name}" for name in names],
            *[f"property uchar {c}" for c in colours],
        ],
        body=record.tobytes(),
    )

    splat = gaussians.read_splat(path)

    # colour = SH_C0 * f_dc + 0.5, so a saturated red channel inverts back to 1.0.
    red = gaussians.SH_C0 * splat.columns["f_dc_0"] + 0.5
    assert red.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)
    # Alpha 255 and 0 are +/-inf as logits until they are clamped; they must be finite and
    # must still round back to the bytes they came from.
    assert bool(np.isfinite(splat.columns["opacity"]).all())
    alpha = 1.0 / (1.0 + np.exp(-splat.columns["opacity"]))
    assert np.round(alpha * 255).tolist() == [255.0, 0.0]


# --------------------------------------------------------------------------------------
# The shapes that refuse, by name
# --------------------------------------------------------------------------------------


def test_an_ascii_ply_is_refused_and_says_it_is_ascii(tmp_path: Path) -> None:
    path = tmp_path / "ascii.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0.0\n",
        encoding="ascii",
    )

    with pytest.raises(SplatFormatError, match="ASCII PLY"):
        gaussians.read_splat(path)


def test_a_compressed_playcanvas_export_is_refused_by_name(tmp_path: Path) -> None:
    """`element chunk` first, `uint` properties: the file A0 measured `KeyError: 'uint'` on.

    The header parses now -- the chunk block is skipped by its own stride and the vertex
    element is found -- and the refusal names the impl that will read it, rather than
    returning `packed_position` reinterpreted as a coordinate.
    """
    chunk = np.zeros(2, dtype=np.dtype([(f"min_{a}", "<f4") for a in "xyz"]))
    vertex = np.zeros(
        3,
        dtype=np.dtype(
            [
                ("packed_position", "<u4"),
                ("packed_rotation", "<u4"),
                ("packed_scale", "<u4"),
                ("packed_color", "<u4"),
            ]
        ),
    )
    path = _write(
        tmp_path / "compressed.ply",
        header=[
            "format binary_little_endian 1.0",
            "element chunk 2",
            *[f"property float min_{a}" for a in "xyz"],
            "element vertex 3",
            "property uint packed_position",
            "property uint packed_rotation",
            "property uint packed_scale",
            "property uint packed_color",
        ],
        body=chunk.tobytes() + vertex.tobytes(),
    )

    with pytest.raises(SplatFormatError) as raised:
        gaussians.read_splat(path)

    message = str(raised.value)
    assert "packed_position" in message
    assert "ply_compressed" in message
    assert "x, y, z" in message


def test_a_ply_with_no_vertex_element_lists_the_elements_it_has(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mesh-only.ply",
        header=[
            "format binary_little_endian 1.0",
            "element point 1",
            "property float x",
        ],
        body=b"\x00\x00\x00\x00",
    )

    with pytest.raises(SplatFormatError, match=r"no `element vertex`.*point"):
        gaussians.read_splat(path)


def test_a_truncated_ply_says_how_many_bytes_are_missing(tmp_path: Path) -> None:
    full = _canonical_ply(tmp_path / "short.ply").read_bytes()
    (tmp_path / "short.ply").write_bytes(full[:-20])

    with pytest.raises(SplatFormatError, match="truncated"):
        gaussians.read_splat(tmp_path / "short.ply")


def test_a_file_that_is_not_a_ply_at_all_says_so(tmp_path: Path) -> None:
    path = tmp_path / "notes.ply"
    path.write_bytes(b"this is not a PLY file\n")

    with pytest.raises(SplatFormatError, match="end_header"):
        gaussians.read_splat(path)


def test_an_upload_with_nothing_readable_in_it_names_what_it_found(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "scan.mp4").write_bytes(b"video")

    with pytest.raises(SplatFormatError, match="photo-reconstruct"):
        gaussians.pick_splat_file(upload)
