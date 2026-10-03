"""Buckets, scans and assets for the sidecar tests (attach and carry-forward).

Two moto buckets, as tests/test_publish_bucket.py has them: `PRIVATE` holds the runs and
the staging area, `PUBLIC` what a browser reads. A scan is a `tileset.json`, a few tiles,
and whatever sidecars a test puts beside them.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import boto3
from moto import mock_aws
from sqlalchemy.orm import Session

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
FILL_EXTRAS: list[dict[str, Any]] = [
    {
        "uri": "inferred/fixer/tileset.json",
        "evidence": {
            "kind": "inferred",
            "filler": "nvidia-fixer",
            "views": 12,
            "gaussians": 900,
            "meanConfidence": 0.4,
        },
    }
]
COLLISION_EXTRAS = {"format": "hexapod.collision", "version": 1, "uri": "collision.bin"}
FILL_FILES = {
    "inferred/fixer/tileset.json": json.dumps(a_tileset(("f0.glb",))).encode(),
    "inferred/fixer/f0.glb": b"fill glb",
}
