"""Buckets, scans and assets for the sidecar tests (attach and carry-forward).

Two moto buckets, as tests/test_publish_bucket.py has them: `PRIVATE` holds the runs and
the staging area, `PUBLIC` what a browser reads. A scan is a `tileset.json`, a few tiles,
and whatever sidecars a test puts beside them.
"""

from __future__ import annotations

import gzip
import json
import struct
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pytest
from moto import mock_aws
from sqlalchemy.orm import Session

from app.config import REPO_ROOT
from app.models import Asset
from app.models.enums import AssetProvider, Representation
from app.storage import ObjectStorage, S3Storage
from app.worker.publish import Publisher

PRIVATE = "twin-assets"
PUBLIC = "twin-public"
PUBLIC_URL = "https://tiles.example.com"

JOB = uuid.UUID("00000000-0000-0000-0000-0000000000cd")
#: A scan published before generations existed: the spool, pumpkin and camp are like this.
LEGACY_DIR = f"runs/{JOB}/package/splat/"
LEGACY_URL = f"{PUBLIC_URL}/{LEGACY_DIR}tileset.json"
GENERATION = "0123456789abcdef"
LIVE_DIR = f"runs/{JOB}/p{GENERATION}/package/splat/"
LIVE_URL = f"{PUBLIC_URL}/{LIVE_DIR}tileset.json"

TILES = ("0/0.glb", "0/1.glb", "0/2.glb")
TRANSFORM = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 100.0, 200.0, 300.0, 1.0]


def storage_over(bucket: str, *, public_base_url: str | None = None) -> S3Storage:
    return S3Storage(
        bucket=bucket,
        endpoint_url=None,
        access_key="key",
        secret_key="secret",
        region="us-east-1",
        public_base_url=public_base_url,
    )


def two_buckets() -> Iterator[Publisher]:
    """For a module's own `buckets` fixture: `yield from two_buckets()`."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=PRIVATE)
        client.create_bucket(Bucket=PUBLIC)
        yield Publisher(
            private=storage_over(PRIVATE),
            public=storage_over(PUBLIC, public_base_url=PUBLIC_URL),
        )


def a_tileset(
    tiles: tuple[str, ...] = TILES, extras: Mapping[str, Any] | None = None, error: float = 1.0
) -> dict[str, Any]:
    """A two-level tileset: the first tile at the root, the rest its children."""
    box = [0.0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5]
    root: dict[str, Any] = {
        "boundingVolume": {"box": box},
        "geometricError": error,
        "refine": "REPLACE",
        "transform": TRANSFORM,
        "content": {"uri": tiles[0]},
        "children": [
            {"boundingVolume": {"box": box}, "geometricError": 0, "content": {"uri": tile}}
            for tile in tiles[1:]
        ],
        "extras": {"gaussians": 3, **(extras or {})},
    }
    return {"asset": {"version": "1.1"}, "geometricError": 10, "root": root}


def put_scan(
    storage: ObjectStorage,
    directory: str,
    *,
    tiles: Mapping[str, bytes] | None = None,
    sidecars: Mapping[str, bytes] | None = None,
    extras: Mapping[str, Any] | None = None,
    document: Mapping[str, Any] | None = None,
) -> None:
    """A scan's directory: its tiles (`b"glb <name>"` unless given), sidecars, tileset."""
    tiles = tiles if tiles is not None else {name: f"glb {name}".encode() for name in TILES}
    for name, data in tiles.items():
        storage.put_object(directory + name, data, "model/gltf-binary")
    for name, data in (sidecars or {}).items():
        storage.put_object(directory + name, data, "application/octet-stream")
    doc = document if document is not None else a_tileset(tuple(tiles), extras)
    storage.put_object(directory + "tileset.json", json.dumps(doc).encode(), "application/json")


def tileset_at(storage: ObjectStorage, url: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(storage.get_object(url.removeprefix(f"{PUBLIC_URL}/")))
    return document


def directory_of(url: str) -> str:
    return url.removeprefix(f"{PUBLIC_URL}/").rsplit("/", 1)[0] + "/"


def files_under(storage: ObjectStorage, directory: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    token: str | None = None
    while True:
        page = storage.list_objects(directory, continuation_token=token)
        for item in page.objects:
            out[item.key[len(directory) :]] = storage.get_object(item.key)
        token = page.next_continuation_token
        if token is None:
            return out


def an_asset(db: Session, url: str, **render: Any) -> Asset:
    asset = Asset(
        name="Camp splat",
        representation=Representation.GAUSSIAN_SPLAT,
        provider=AssetProvider.TILES_3D_URL,
        source={"type": "3d-tiles-url", "url": url},
        render_config=dict(render),
        attribution=[],
        default_visible=True,
    )
    db.add(asset)
    db.commit()
    return asset


def stage(
    storage: ObjectStorage, asset_id: uuid.UUID, files: Mapping[str, bytes], token: str = "run-1"
) -> str:
    """Upload `files` to the asset's staging area, as a workflow would; returns the prefix."""
    prefix = f"staging/assets/{asset_id}/{token}/"
    for rel, data in files.items():
        storage.put_object(prefix + rel, data, "application/octet-stream")
    return prefix


INSTANCES_EXTRAS = {"uri": "instances.json", "count": 2}
#: A scan's confidence grid as the packer declares it (P4): what a fill is gated against.
GRID_SHA = "a1" * 32
GRID_EXTRAS = {
    "format": "hexapod.cellgrid",
    "version": 1,
    "kind": "confidence",
    "uri": "confidence.bin",
    "sha256": GRID_SHA,
}
GRID_FILES = {"confidence.bin": b"cells"}
#: Evidence of a fill made without the fill gate (before ADR 0016 § Fill): the camp's.
UNGATED_EVIDENCE: dict[str, Any] = {
    "kind": "inferred",
    "filler": "nvidia-fixer",
    "views": 12,
    "gaussians": 900,
    "meanConfidence": 0.4,
}


def gate_for(grid_sha: str = GRID_SHA, epoch: uuid.UUID | str | None = None) -> dict[str, Any]:
    """`evidence.rescanGate` of a fill gated against the grid `grid_sha` names."""
    return {
        "policy": "rescan-gate-v1",
        "siteId": "00000000-0000-4000-8000-000000005173",
        "epochId": str(epoch or "00000000-0000-4000-8000-0000000000e2"),
        "weakIn": ["00000000-0000-4000-8000-0000000000e1"],
        "comparedWith": ["00000000-0000-4000-8000-0000000000e1"],
        "level": 2,
        "cellM": 1.0,
        "i0": -4,
        "j0": -4,
        "ni": 40,
        "nj": 32,
        "cells": 90,
        "maskSha256": "4d" * 32,
        "confidenceSha256": grid_sha,
    }


FILL_EXTRAS: list[dict[str, Any]] = [
    {
        "uri": "inferred/fixer/tileset.json",
        "evidence": {**UNGATED_EVIDENCE, "rescanGate": gate_for()},
    }
]
UNGATED_FILL_EXTRAS: list[dict[str, Any]] = [
    {"uri": "inferred/fixer/tileset.json", "evidence": UNGATED_EVIDENCE}
]
COLLISION_EXTRAS = {"format": "hexapod.collision", "version": 1, "uri": "collision.bin"}
FILL_FILES = {
    "inferred/fixer/tileset.json": json.dumps(a_tileset(("f0.glb",))).encode(),
    "inferred/fixer/f0.glb": b"fill glb",
}


# --- real splat tiles ----------------------------------------------------------------

#: The committed fixture tree packed into 21 level-of-detail tiles, and the same tree packed
#: again with spherical harmonics (tools/captures splat_tiles.py `--sh-degree`): every tile's
#: bytes differ, every position is where it was. `rig.json` beside the first lists the tiles'
#: position checksums, stamped by tools/captures (`rig_tiles.stamp`).
TREE_LOD = REPO_ROOT / "data" / "tiles" / "synthetic-tree-lod"
TREE_SH = REPO_ROOT / "data" / "tiles" / "synthetic-tree-sh"


def fixture_scan(directory: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A committed tileset: its document and every tile it names, by uri."""
    document: dict[str, Any] = json.loads((directory / "tileset.json").read_text())
    uris: list[str] = []
    stack: list[Any] = [document["root"]]
    while stack:
        tile = stack.pop()
        uri = tile.get("content", {}).get("uri")
        if uri:
            uris.append(uri)
        stack.extend(tile.get("children", []))
    return document, {uri: (directory / uri).read_bytes() for uri in sorted(uris)}


SPZ_MAGIC = 0x5053474E


#: `splat_tiles.build_glb`'s node matrix (column-major): the z-up data turned y-up, which
#: the viewer's axis correction turns back.
Z_UP_TO_Y_UP = [1.0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def _spz(
    positions: np.ndarray | list[list[float]],
    *,
    version: int,
    fractional_bits: int,
    fill: int,
    sh_degree: int,
) -> bytes:
    xyz = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    count = xyz.shape[0]
    fixed = np.round(xyz * (1 << fractional_bits)).astype(np.int64) & 0xFFFFFF
    packed = np.stack([(fixed >> shift) & 0xFF for shift in (0, 8, 16)], axis=-1)
    rotation = 3 if version == 2 else 4
    rest = bytes([fill]) * ((1 + 3 + 3 + rotation) * count)
    harmonics = bytes([fill]) * (3 * (0, 3, 8, 15)[sh_degree] * count)
    header = struct.pack("<IIIBBBB", SPZ_MAGIC, version, count, sh_degree, fractional_bits, 0, 0)
    return gzip.compress(header + packed.astype(np.uint8).tobytes() + rest + harmonics, mtime=0)


def splat_glb(
    positions: np.ndarray | list[list[float]],
    *,
    version: int = 2,
    fractional_bits: int = 12,
    fill: int = 0,
    sh_degree: int = 0,
    also: tuple[np.ndarray | list[list[float]], ...] = (),
    edit: Callable[[dict[str, Any]], object] | None = None,
) -> bytes:
    """A splat GLB laid out exactly as `splat_tiles.build_glb` writes one (its JSON keys,
    extensions, accessors and material): one scene, one node (its z-up-to-y-up matrix) and
    one primitive holding an SPZ stream of these positions (24-bit
    fixed point), every other byte `fill`, `sh_degree` bands of harmonics after them -- so
    two tiles can hold the same positions in different bytes. `also`: more primitives in
    the same mesh, a buffer view each; `edit` changes the glTF JSON before it is written
    (another node matrix, a node hierarchy)."""
    streams = [
        _spz(
            points,
            version=version,
            fractional_bits=fractional_bits,
            fill=fill,
            sh_degree=sh_degree,
        )
        for points in (positions, *also)
    ]
    views: list[dict[str, Any]] = []
    binary = b""
    for spz in streams:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(spz)})
        binary += spz + b"\0" * (-len(spz) % 4)
    counts = [np.asarray(points).reshape(-1, 3).shape[0] for points in (positions, *also)]
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "twin splat_tiles"},
        "extensionsUsed": [
            "KHR_materials_unlit",
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
        ],
        "extensionsRequired": [
            "KHR_gaussian_splatting",
            "KHR_gaussian_splatting_compression_spz_2",
        ],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "matrix": list(Z_UP_TO_Y_UP)}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": 0,
                        "material": 0,
                        "attributes": {
                            "POSITION": 4 * n,
                            "COLOR_0": 4 * n + 1,
                            "KHR_gaussian_splatting:SCALE": 4 * n + 2,
                            "KHR_gaussian_splatting:ROTATION": 4 * n + 3,
                        },
                        "extensions": {
                            "KHR_gaussian_splatting": {
                                "extensions": {
                                    "KHR_gaussian_splatting_compression_spz_2": {"bufferView": n}
                                }
                            }
                        },
                    }
                    for n in range(len(streams))
                ]
            }
        ],
        "materials": [{"extensions": {"KHR_materials_unlit": {}}}],
        "accessors": [
            accessor
            for count in counts
            for accessor in (
                {"componentType": 5126, "count": count, "type": "VEC3"},
                {"componentType": 5121, "normalized": True, "count": count, "type": "VEC4"},
                {"componentType": 5126, "count": count, "type": "VEC3"},
                {"componentType": 5126, "count": count, "type": "VEC4"},
            )
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    if edit is not None:
        edit(gltf)
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


# --- the API's recompute of the fill gate, for tests of something else ------------------


def waive_fill_gate(monkeypatch: pytest.MonkeyPatch) -> list[list[Any]]:
    """Admit every inferred layer at the API's recompute of the fill gate -- on attach
    (`fill_gate.verify`) and on a republish (`fill_gate.carry_check`) -- and record the
    layers it was asked to judge, one list per attach (one entry per carried layer).

    For the attach's and the carry's own mechanics (merging layers by uri, the row lock,
    what a republish keeps), whose fills sit on single scans with no re-scan. The gate
    itself -- targets recomputed from `epoch_maps`, every gaussian held to them -- is held
    in tests/test_fill_gate.py on a real two-epoch site; the metadata half
    (`sidecars.fill_gate_problem`) still runs here."""
    from app.services import fill_gate

    asked: list[list[Any]] = []

    def admit(_db: Any, _storage: Any, _asset: Any, layers: list[Any], **_: Any) -> None:
        if layers:
            asked.append(list(layers))

    def carry_check(*_args: Any, **_kwargs: Any) -> Any:
        def check(entry: Any, *_: Any) -> None:
            asked.append([entry])

        return check

    monkeypatch.setattr(fill_gate, "verify", admit)
    monkeypatch.setattr(fill_gate, "carry_check", carry_check)
    return asked


def admit_fills(_entry: Any, *_args: Any) -> None:
    """A carry's `fill_check` that admits every layer: for the carry's own rules (the
    metadata half, what goes with a split), the recompute waived as `waive_fill_gate` does."""
    return None
