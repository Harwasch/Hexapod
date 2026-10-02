"""Splats on disk a chunk at a time: the reader agrees with the packager's, the writer with
`gaussians.write_ply`, and per-gaussian columns come back as they went in."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import gaussians
import splat_io
from captures_bridge import SplatFormatError, read_ply


def _ply(path: Path, rows: np.ndarray, *, order: str = "<", before: str = "") -> Path:
    kinds = {"f4": "float", "u1": "uchar", "i2": "short", "f8": "double"}
    header = f"ply\nformat binary_{'little' if order == '<' else 'big'}_endian 1.0\n{before}"
    header += f"element vertex {rows.shape[0]}\n"
    for name in rows.dtype.names or ():
        header += f"property {kinds[rows.dtype[name].str[1:]]} {name}\n"
    header += "end_header\n"
    path.write_bytes(header.encode("ascii") + rows.tobytes())
    return path


def _gsplat_rows(count: int, *, order: str = "<") -> np.ndarray:
    """gsplat v1.5.3's export (`exporter.splat2ply_bytes`): x y z, f_dc_0..2, f_rest_0..44,
    opacity, scale_0..2, rot_0..3 -- 59 float32s, 236 bytes a row."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(45)] + ["opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    rows = np.zeros(count, dtype=np.dtype([(n, order + "f4") for n in names]))
    rng = np.random.default_rng(count)
    for name in names:
        rows[name] = rng.normal(size=count)
    return rows


def test_the_scalar_table_is_the_packagers() -> None:
    # `splat_tiles` is importable once `captures_bridge` has put tools/captures on the path.
    assert dict(splat_io.PLY_SCALARS) == importlib.import_module("splat_tiles").PLY_SCALARS


@pytest.mark.parametrize("order", ["<", ">"])
def test_row_ranges_read_what_the_packager_reads(tmp_path: Path, order: str) -> None:
    """gsplat's SH3 layout, either byte order, behind an element the reader must skip."""
    rows = _gsplat_rows(10_007, order=order)
    before = "element camera 2\nproperty double fx\nproperty double fy\n"
    path = tmp_path / "gsplat.ply"
    raw = _ply(tmp_path / "plain.ply", rows, order=order).read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    head = head.replace(b"element vertex", before.encode() + b"element vertex")
    path.write_bytes(head + b"end_header\n" + np.zeros(4, dtype=order + "f8").tobytes() + body)

    reader = splat_io.SplatReader(path, chunk=999)
    whole = read_ply(path)
    assert reader.count == 10_007
    assert reader.properties == tuple(whole)
    assert reader.layout.itemsize == 236
    for name in ("x", "f_rest_44", "rot_3"):
        joined = np.concatenate([columns[name] for _, columns in reader.chunks([name])])
        assert joined.tobytes() == whole[name].tobytes()
    part = reader.read(5_000, 5_003)
    assert part["opacity"].tobytes() == whole["opacity"][5_000:5_003].tobytes()
    assert reader.memmap()["x"][17] == whole["x"][17]
    assert reader.checksum() == splat_io.file_checksum(path)


def test_a_wide_layout_is_read_in_fewer_rows(tmp_path: Path) -> None:
    path = _ply(tmp_path / "gsplat.ply", _gsplat_rows(10))
    assert splat_io.SplatReader(path).chunk == splat_io.CHUNK_BYTES // 236
    canonical = tmp_path / "canonical.ply"
    gaussians.write_ply(
        canonical, {n: np.zeros(3, np.float32) for n in gaussians.CANONICAL_PROPERTIES}
    )
    assert splat_io.SplatReader(canonical).chunk == splat_io.CHUNK


@pytest.mark.parametrize(
    ("content", "refusal"),
    [
        (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n", "ASCII"),
        (b"ply\nformat binary_little_endian 1.0\nend_header\n", "no `element vertex`"),
        (
            b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\n"
            b"end_header\n\0\0\0\0",
            "truncated",
        ),
        (
            b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            b"property list uchar int idx\nend_header\n",
            "list properties",
        ),
        (b"not a ply\nend_header\n", "does not start with 'ply'"),
    ],
)
def test_what_the_packager_refuses_is_refused_by_name(
    tmp_path: Path, content: bytes, refusal: str
) -> None:
    path = tmp_path / "bad.ply"
    path.write_bytes(content)
    with pytest.raises(SplatFormatError, match=refusal):
        splat_io.SplatReader(path)


@pytest.mark.parametrize("known", [True, False])
def test_the_writer_writes_write_plys_bytes(tmp_path: Path, known: bool) -> None:
    rng = np.random.default_rng(1)
    columns = {
        name: rng.normal(size=5_003).astype(np.float32) for name in gaussians.CANONICAL_PROPERTIES
    }
    gaussians.write_ply(tmp_path / "whole.ply", columns)
    out = tmp_path / "chunked.ply"
    with splat_io.PlyWriter(
        out, gaussians.CANONICAL_PROPERTIES, count=5_003 if known else None
    ) as writer:
        for start in range(0, 5_003, 1_000):
            writer.append({name: values[start : start + 1_000] for name, values in columns.items()})
    assert out.read_bytes() == (tmp_path / "whole.ply").read_bytes()
    assert not (tmp_path / "chunked.ply.rows").exists()
    # And the packager reads it back as it was written.
    assert read_ply(out)["rot_2"].tobytes() == columns["rot_2"].tobytes()


def test_a_writer_that_is_given_fewer_rows_than_it_promised_refuses(tmp_path: Path) -> None:
    writer = splat_io.PlyWriter(tmp_path / "short.ply", ["x"], count=3)
    writer.append({"x": np.zeros(2, np.float32)})
    with pytest.raises(ValueError, match="promised 3 rows and 2 were written"):
        writer.close()


def test_columns_and_npy_files_are_read_back_by_row_range(tmp_path: Path) -> None:
    store = splat_io.ColumnStore(tmp_path / "store")
    views = store.create("views", np.int32)
    tiers = store.create("tiers", np.uint8)
    for start in range(0, 1_000, 300):
        views.append(np.arange(start, min(1_000, start + 300), dtype=np.int32))
        tiers.append(np.full(min(300, 1_000 - start), 2, dtype=np.uint8))
    store.finish()
    np.testing.assert_array_equal(store.read("views", 295, 305), np.arange(295, 305))
    assert store.column("tiers").size == 1_000
    assert "views" in store and "gsd" not in store
    store.remove()
    assert not (tmp_path / "store").exists()

    array = np.linspace(0, 1, 777, dtype=np.float32)
    np.save(tmp_path / "error.npy", array)
    column = splat_io.npy_column(tmp_path / "error.npy")
    assert column.size == 777
    np.testing.assert_array_equal(column.read(700, 900), array[700:])
    np.save(tmp_path / "square.npy", np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError, match="1-D"):
        splat_io.npy_column(tmp_path / "square.npy")
