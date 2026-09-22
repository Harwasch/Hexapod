"""Artifacts: what a stage declares it writes, and how it is identified afterwards.

An artifact's *name* is also its *path*. An artifact called `splat` produced by the stage
`package` lives at `stages/package/out/splat`; one called `georef.json` lives at
`stages/georeference/out/georef.json`. A later stage never computes that path: it asks for
the artifact by name and the executor hands it the resolved path. That is the whole reason
stages can be reordered or swapped without editing anything but the recipe.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from errors import RecipeError

__all__ = [
    "ArtifactDecl",
    "ArtifactManifest",
    "ArtifactRef",
    "checksum_of",
    "dir_checksum",
    "file_checksum",
    "size_of",
    "validate_artifact_name",
]

Kind = Literal["file", "dir"]

# Artifact names double as relative paths, so they are deliberately boring: no absolute
# paths, no `..`, no backslashes, no leading dot.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*(/[a-z0-9][a-z0-9_.-]*)*$")
_HASH_CHUNK = 1 << 20


def validate_artifact_name(name: str) -> str:
    if not _NAME_RE.match(name) or ".." in name:
        raise RecipeError(
            f"invalid artifact name {name!r}: expected a relative, lowercase path such as "
            f"'frames', 'georef.json' or 'motion/clip.f32'"
        )
    return name


@dataclass(frozen=True)
class ArtifactDecl:
    """What an implementation promises to write, declared next to the implementation.

    `required_members` is the part of a directory artifact's shape that every
    implementation must honour -- for `splat/` it is exactly what
    `tools/captures/splat_tiles.convert` writes, which is what makes a stubbed package
    stage and the real one interchangeable.

    `stub_members` is what StubRunner fabricates for a directory: a miniature of the real
    shape. It is a superset of `required_members` and never checked against real runs,
    because a real `frames/` holds 400 frames and a stub holds four.
    """

    name: str
    kind: Kind = "file"
    content_type: str = "application/octet-stream"
    summary: str = ""
    required_members: tuple[str, ...] = ()
    stub_members: tuple[str, ...] = ()
    stub_bytes: int = 256

    def __post_init__(self) -> None:
        validate_artifact_name(self.name)
        if self.kind == "file" and (self.required_members or self.stub_members):
            raise RecipeError(f"artifact {self.name!r} is a file and cannot declare members")
        if self.kind == "dir":
            for member in (*self.required_members, *self.stub_members):
                validate_artifact_name(member)

    @property
    def members_to_stub(self) -> tuple[str, ...]:
        seen = list(self.stub_members)
        seen += [m for m in self.required_members if m not in seen]
        return tuple(seen)


@dataclass(frozen=True)
class ArtifactRef:
    """An artifact that exists: where it is, how big it is, and what it hashes to."""

    name: str
    stage_id: str
    path: str
    kind: Kind
    content_type: str
    bytes: int
    checksum: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "stageId": self.stage_id,
            "path": self.path,
            "kind": self.kind,
            "contentType": self.content_type,
            "bytes": self.bytes,
            "checksum": self.checksum,
        }


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def dir_checksum(path: Path) -> str:
    """Hash of the sorted (relative path, size, file hash) triples beneath `path`.

    Order-independent and mtime-independent, so a directory artifact regenerated from the
    same inputs hashes the same on another machine -- the property the sibling project's
    byte-identity gate already depends on.
    """
    digest = hashlib.sha256()
    for member in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = member.relative_to(path).as_posix()
        digest.update(f"{rel}\0{member.stat().st_size}\0{file_checksum(member)}\n".encode())
    return f"sha256:{digest.hexdigest()}"


def checksum_of(path: Path) -> str:
    return dir_checksum(path) if path.is_dir() else file_checksum(path)


def size_of(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


@dataclass(frozen=True)
class ArtifactManifest:
    """`artifacts.json`: every artifact a run produced, in stage order.

    Deliberately free of timings and of anything else that varies between runs, so two runs
    of the same recipe over the same inputs produce a byte-identical manifest. Per-stage
    timings live in `stages/<id>/step.json`.
    """

    entries: tuple[ArtifactRef, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {"artifacts": [entry.to_dict() for entry in self.entries]}

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n", "utf-8")
