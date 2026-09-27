"""Splats on disk, a bounded chunk at a time: read, append, and spill per-gaussian columns.

**Why this exists.** Every stage after `train` used to read the whole splat into memory
(`gaussians.read_splat`: the file's bytes, then every property as float32, then the
canonical fourteen, then the copies each transform makes). Measured on the 2 GB Fly worker
that is ~0.75 GB a million gaussians (apps/api/app/worker/README.md), which is why
`gaussian_budget` capped training at 2M: the GPU could hold four times more, and the
worker could not hold what it trained. The rule is no bigger machines, so instead nothing
after training holds the splat: it is read in fixed-size row ranges, and every stage that
needs a statistic of all of it gets that statistic in passes over the ranges
(`outofcore.py`), never by loading the column.

Three pieces, each small:

* `SplatReader` -- rows `[start, stop)` of a binary PLY's vertex block as float32 columns.
  The header is parsed exactly as `tools/captures/splat_tiles.read_ply` parses it (either
  byte order; elements before `vertex` skipped by their stride; list properties refused),
  so every layout that reaches the pipeline is read the same way: `canonical.ply` and
  `trained.ply` (the fourteen `CANONICAL_PROPERTIES`, 56 bytes a row), gsplat's own export
  (`x y z`, `f_dc_*`, 45 `f_rest_*`, `opacity`, `scale_*`, `rot_*`: 236 bytes a row at
  SH degree 3; Inria's adds normals, 248), and the phone apps' PLYs with vertex colours.
  A range is read with one `read` at an offset, never through a mapping of the whole
  file: a mapping would make the file's size count against the process's address space
  and its touched pages against its resident set, and the bound here is meant to be what
  the process holds, measured honestly. `memmap` is there for a caller that wants random
  access and can afford the mapping.
* `PlyWriter` -- appends chunks to `canonical.ply`, byte-identical to
  `gaussians.write_ply` of the same rows (same header, same record layout), which is what
  `splat_tiles.read_ply` -- the packager -- reads. With the row count known up front (every
  caller here knows it: `place` writes as many as it read, `quality` counts its tiers
  before it writes) the header goes first; without one, rows are spooled beside the file
  and the header is prepended when it is closed.
* `ColumnStore` -- a directory of raw little-endian columns, one file each, appended a
  chunk at a time and read back by row range: where a stage keeps per-gaussian results
  (views, tiers, ...) between passes instead of holding them. `npy_column` reads a `.npy`
  file (the `holdout/` arrays) by row range the same way.

`CHUNK` rows is the unit, and `CHUNK_BYTES` of file its ceiling: at 2^18 a canonical chunk
is 14.7 MB of file and ~15 MB of columns, and a wider layout is read in proportionally
fewer rows (gsplat's SH3 rows in 71k-row chunks), so a chunk costs the same whatever the
file carries. The quality stage's heaviest per-chunk array is cameras x rows booleans,
which it bounds separately (`quality.SEEN_BYTES`).
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from captures_bridge import SplatFormatError

__all__ = [
    "CHUNK",
    "Column",
    "ColumnStore",
    "PlyLayout",
    "PlyWriter",
    "SplatReader",
    "npy_column",
    "ranges",
    "read_layout",
]

F32 = npt.NDArray[np.float32]

#: Rows per chunk: ~15 MB of canonical columns. See the module docstring.
CHUNK = 1 << 18
#: The most bytes of file one chunk reads: a wide layout is read in fewer rows.
CHUNK_BYTES = 16 << 20

#: Every scalar type a PLY header may name -- the same table as `splat_tiles.PLY_SCALARS`,
#: which is private to the packager; kept equal by `tests/test_splat_io.py`.
PLY_SCALARS: Mapping[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_BYTE_ORDER = {"binary_little_endian": "<", "binary_big_endian": ">"}
_MAX_HEADER_LINES = 10_000


def ranges(count: int, chunk: int = CHUNK) -> Iterator[tuple[int, int]]:
    """`[start, stop)` of each chunk of `count` rows, in order."""
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, not {chunk}")
    for start in range(0, count, chunk):
        yield start, min(count, start + chunk)


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlyLayout:
    """Where a PLY's vertex rows are and what each one holds."""

    path: Path
    count: int
    #: The vertex record, with the file's byte order.
    dtype: np.dtype[Any]
    #: Byte offset of the first vertex row.
    offset: int
    properties: tuple[str, ...]

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)


@dataclass
class _Element:
    name: str
    count: int
    names: list[str]
    types: list[str]
    lists: list[str]

    def dtype(self, order: str) -> np.dtype[Any]:
        return np.dtype(
            [(n, order + PLY_SCALARS[t]) for n, t in zip(self.names, self.types, strict=True)]
        )


def read_layout(path: Path) -> PlyLayout:
    """Parse the header only. Refusals say what is wrong, in `splat_tiles.read_ply`'s words."""
    lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise SplatFormatError(f"{path.name}: the header has no `end_header` line")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            lines.append(line)
            if len(lines) > _MAX_HEADER_LINES:
                raise SplatFormatError(f"{path.name}: no `end_header` in the first 10000 lines")
        header_end = handle.tell()
    order, elements = _parse_header(path, lines)
    vertex = next((element for element in elements if element.name == "vertex"), None)
    if vertex is None:
        found = ", ".join(element.name for element in elements) or "none"
        raise SplatFormatError(f"{path.name} has no `element vertex`; its elements are: {found}")
    if vertex.lists:
        raise SplatFormatError(
            f"{path.name}: the vertex element has list properties ({', '.join(vertex.lists)}), "
            f"which a gaussian splat never has"
        )
    if not vertex.names:
        raise SplatFormatError(f"{path.name}: the vertex element declares no properties")
    offset = header_end
    for earlier in elements:
        if earlier is vertex:
            break
        if earlier.lists:
            raise SplatFormatError(
                f"{path.name}: element {earlier.name!r} comes before the vertex element and "
                f"has a list property ({', '.join(earlier.lists)}), so the vertex data "
                f"cannot be located without parsing it"
            )
        offset += earlier.dtype(order).itemsize * earlier.count
    dtype = vertex.dtype(order)
    wanted = dtype.itemsize * vertex.count
    available = path.stat().st_size - offset
    if available < wanted:
        raise SplatFormatError(
            f"{path.name} is truncated: the header declares {vertex.count} vertices "
            f"({wanted} bytes of vertex data) and only {max(0, available)} bytes follow it"
        )
    return PlyLayout(
        path=path,
        count=vertex.count,
        dtype=dtype,
        offset=offset,
        properties=tuple(vertex.names),
    )


def _parse_header(path: Path, lines: list[str]) -> tuple[str, list[_Element]]:
    if not lines or lines[0].strip() != "ply":
        raise SplatFormatError(f"{path.name} does not start with 'ply'; it is not a PLY file")
    order = ""
    elements: list[_Element] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1] if len(parts) > 1 else ""
            if fmt == "ascii":
                raise SplatFormatError(
                    f"{path.name} is an ASCII PLY, and only binary PLY is supported. "
                    f"Re-export it as binary (in MeshLab: 'Binary encoding'), or convert "
                    f"it with a tool that writes binary_little_endian"
                )
            order = _BYTE_ORDER.get(fmt, "")
            if not order:
                raise SplatFormatError(
                    f"{path.name} declares format {fmt!r}; expected binary_little_endian, "
                    f"binary_big_endian or ascii"
                )
        elif parts[0] == "element":
            if len(parts) < 3 or not parts[2].lstrip("-").isdigit():
                raise SplatFormatError(f"{path.name}: malformed element line {line!r}")
            elements.append(_Element(parts[1], int(parts[2]), [], [], []))
        elif parts[0] == "property":
            if not elements:
                raise SplatFormatError(f"{path.name}: property before any element: {line!r}")
            if len(parts) > 1 and parts[1] == "list":
                elements[-1].lists.append(parts[-1])
                continue
            if len(parts) < 3 or parts[1] not in PLY_SCALARS:
                kind = parts[1] if len(parts) > 1 else "(none)"
                raise SplatFormatError(
                    f"{path.name}: property {parts[-1]!r} of element {elements[-1].name!r} "
                    f"has type {kind!r}, which is not a PLY scalar type. Known types: "
                    f"{', '.join(sorted(PLY_SCALARS))}"
                )
            elements[-1].types.append(parts[1])
            elements[-1].names.append(parts[-1])
    if not order:
        raise SplatFormatError(f"{path.name}: the header has no `format` line")
    return order, elements


class SplatReader:
    """Row ranges of a PLY's vertex block, as float32 columns (as `read_ply` returns them)."""

    def __init__(self, path: Path, *, chunk: int = CHUNK) -> None:
        self.layout = read_layout(path)
        # `chunk` rows, or fewer when that many rows of this layout exceed `CHUNK_BYTES`.
        self.chunk = max(1, min(chunk, CHUNK_BYTES // max(1, self.layout.itemsize)))

    @property
    def path(self) -> Path:
        return self.layout.path

    @property
    def count(self) -> int:
        return self.layout.count

    @property
    def properties(self) -> tuple[str, ...]:
        return self.layout.properties

    def records(self, start: int, stop: int) -> npt.NDArray[np.void]:
        """Rows `[start, stop)` as the file's own structured records."""
        start, stop = max(0, start), min(self.count, stop)
        if stop <= start:
            return np.zeros(0, dtype=self.layout.dtype)
        size = self.layout.itemsize
        want = (stop - start) * size
        buffer = bytearray(want)
        view = memoryview(buffer)
        with self.path.open("rb", buffering=0) as handle:
            handle.seek(self.layout.offset + start * size)
            got = 0
            while got < want:
                read = handle.readinto(view[got:])
                if not read:
                    raise SplatFormatError(f"{self.path.name}: the file ended inside a row")
                got += read
        return np.frombuffer(buffer, dtype=self.layout.dtype, count=stop - start)

    def read(self, start: int, stop: int, names: Sequence[str] | None = None) -> dict[str, F32]:
        """Rows `[start, stop)` of `names` (all properties by default), as float32."""
        rows = self.records(start, stop)
        wanted = self.properties if names is None else tuple(names)
        missing = [name for name in wanted if name not in self.properties]
        if missing:
            raise SplatFormatError(f"{self.path.name} has no {', '.join(missing)}")
        return {name: np.ascontiguousarray(rows[name], dtype=np.float32) for name in wanted}

    def chunks(
        self, names: Sequence[str] | None = None, *, chunk: int | None = None
    ) -> Iterator[tuple[int, dict[str, F32]]]:
        """`(start, columns)` for each chunk, in file order."""
        for start, stop in ranges(self.count, chunk or self.chunk):
            yield start, self.read(start, stop, names)

    def memmap(self) -> np.memmap[Any, np.dtype[Any]]:
        """The whole vertex block, mapped read-only -- for random access, at the cost the
        module docstring describes. Nothing in the pipeline's stages uses it."""
        return np.memmap(
            self.path,
            dtype=self.layout.dtype,
            mode="r",
            offset=self.layout.offset,
            shape=(self.count,),
        )

    def checksum(self) -> str:
        """`sha256:` of the whole file, streamed -- what `read_splat` records."""
        return file_checksum(self.path)


def file_checksum(path: Path, block: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(block), b""):
            digest.update(piece)
    return f"sha256:{digest.hexdigest()}"


# ---------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------


class PlyWriter:
    """Appends float32 rows of `properties` to a binary little-endian PLY.

    The bytes are `gaussians.write_ply`'s for the same rows: the header is the same four
    kinds of line, with no comment, timestamp or generator, and the rows are the same
    packed float32 records. `count` is the number of rows that will be appended; `close`
    refuses a different number, since the header has already promised it. With `count`
    None the rows are spooled to `<path>.rows` and the header written in front on `close`.
    """

    def __init__(self, path: Path, properties: Sequence[str], *, count: int | None = None) -> None:
        self.path = path
        self.properties = tuple(properties)
        self.expected = count
        self.written = 0
        self._dtype = np.dtype([(name, "<f4") for name in self.properties])
        self._spool = path.with_name(path.name + ".rows") if count is None else None
        target = self._spool if self._spool is not None else path
        self._handle = target.open("wb")
        if count is not None:
            self._handle.write(self._header(count))
        self._closed = False

    def _header(self, count: int) -> bytes:
        lines = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
        lines += [f"property float {name}" for name in self.properties]
        lines.append("end_header")
        return ("\n".join(lines) + "\n").encode("ascii")

    def append(self, columns: Mapping[str, npt.ArrayLike]) -> None:
        if self._closed:
            raise ValueError(f"{self.path.name} is already closed")
        arrays = {name: np.asarray(columns[name]) for name in self.properties}
        size = int(arrays[self.properties[0]].shape[0]) if self.properties else 0
        record = np.empty(size, dtype=self._dtype)
        for name in self.properties:
            record[name] = arrays[name]
        self._handle.write(record.tobytes())
        self.written += size

    def close(self) -> int:
        """Finish the file and return its size in bytes."""
        if self._closed:
            return self.path.stat().st_size
        self._handle.close()
        self._closed = True
        if self._spool is not None:
            with self.path.open("wb") as out, self._spool.open("rb") as rows:
                out.write(self._header(self.written))
                shutil.copyfileobj(rows, out, 1 << 22)
            self._spool.unlink()
        elif self.written != self.expected:
            raise ValueError(
                f"{self.path.name}: the header promised {self.expected} rows and "
                f"{self.written} were written"
            )
        return self.path.stat().st_size

    def abort(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
        if self._spool is not None:
            self._spool.unlink(missing_ok=True)

    def __enter__(self) -> PlyWriter:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if kind is None:
            self.close()
        else:
            self.abort()


# ---------------------------------------------------------------------------------------
# Per-gaussian columns between passes
# ---------------------------------------------------------------------------------------


class Column:
    """One column on disk, read by row range: a raw little-endian array at an offset."""

    def __init__(
        self, path: Path, dtype: npt.DTypeLike, *, offset: int = 0, size: int | None = None
    ) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.offset = offset
        stored = (path.stat().st_size - offset) // self.dtype.itemsize if path.exists() else 0
        self.size = stored if size is None else size

    def read(self, start: int, stop: int) -> npt.NDArray[Any]:
        start, stop = max(0, start), min(self.size, stop)
        if stop <= start:
            return np.zeros(0, dtype=self.dtype.newbyteorder("="))
        with self.path.open("rb") as handle:
            handle.seek(self.offset + start * self.dtype.itemsize)
            values = np.fromfile(handle, dtype=self.dtype, count=stop - start)
        if values.shape[0] != stop - start:
            raise ValueError(f"{self.path.name} ended at row {start + values.shape[0]}")
        return values.astype(self.dtype.newbyteorder("="), copy=False)


class _Appender:
    def __init__(self, path: Path, dtype: np.dtype[Any]) -> None:
        self.dtype = dtype
        self._handle = path.open("wb")

    def append(self, values: npt.ArrayLike) -> None:
        np.ascontiguousarray(values, dtype=self.dtype).tofile(self._handle)

    def close(self) -> None:
        self._handle.close()


class ColumnStore:
    """Per-gaussian columns spilled to a directory, one raw file each.

    Written with `create(name, dtype)` then `append` in row order, read back with
    `column(name).read(start, stop)`. The dtype is stored little-endian whatever the
    machine, so a store is readable where it was written and nowhere else needs it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._dtypes: dict[str, np.dtype[Any]] = {}
        self._open: dict[str, _Appender] = {}

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.bin"

    def create(self, name: str, dtype: npt.DTypeLike) -> _Appender:
        if name in self._open:
            raise ValueError(f"column {name!r} is already being written")
        stored = np.dtype(dtype).newbyteorder("<")
        self._dtypes[name] = stored
        appender = _Appender(self._path(name), stored)
        self._open[name] = appender
        return appender

    def finish(self, *names: str) -> None:
        for name in names or tuple(self._open):
            appender = self._open.pop(name, None)
            if appender is not None:
                appender.close()

    def column(self, name: str) -> Column:
        if name in self._open:
            self.finish(name)
        if name not in self._dtypes:
            raise KeyError(f"no column {name!r} in {self.directory}")
        return Column(self._path(name), self._dtypes[name])

    def read(self, name: str, start: int, stop: int) -> npt.NDArray[Any]:
        return self.column(name).read(start, stop)

    def __contains__(self, name: object) -> bool:
        return name in self._dtypes

    def remove(self) -> None:
        self.finish()
        shutil.rmtree(self.directory, ignore_errors=True)


def npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    """A `.npy` file's shape, dtype and data offset, from its header alone."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"{path.name}: .npy version {version} is not read here")
        offset = handle.tell()
    if dtype.hasobject:
        raise ValueError(f"{path.name} holds Python objects, not numbers")
    return tuple(int(n) for n in shape), dtype, offset


def npy_column(path: Path) -> Column:
    """A 1-D `.npy` file as a `Column`: its header parsed, its data read by row range.
    (A 1-D array is the same bytes in C and Fortran order.)"""
    shape, dtype, offset = npy_header(path)
    if len(shape) != 1:
        raise ValueError(f"{path.name} has shape {shape}; a per-gaussian column is 1-D")
    return Column(path, dtype, offset=offset, size=shape[0])
