"""Attaching sidecars: one publisher, and every attach a new generation.

`POST /assets/{id}/sidecars` replaces the workflows that rewrote a published
`tileset.json` in place (publish-instances.yml, publish-fill.yml, collision-backfill.yml,
streamed-lod-backfill.yml, living-plants.yml). What is held here: the asset moves to a
new generation holding its tiles, every sidecar it already had and the new files, with
the merged root extras in a `tileset.json` written last, all of it immutable; the old
generation is never written; the staged files are checked before anything is copied; two
attaches at once both land; and a legacy prefix (a scan published before generations, as
the spool, pumpkin and camp were) is attached to the same way.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import _db
from app.config import REPO_ROOT, Settings
from app.main import create_app
from app.models import Asset
from app.schemas.sidecar import SidecarAttach
from app.services import attach as attach_service
from app.services import sidecars
from app.services.errors import ConflictError
from app.services.published import IMMUTABLE_CACHE, is_published
from app.storage import S3Storage
from app.storage.factory import get_public_storage, get_storage
from app.worker.publish import Publisher
from tests.sidecar_fixtures import (
    COLLISION_EXTRAS,
    FILL_EXTRAS,
    FILL_FILES,
    INSTANCES_EXTRAS,
    LEGACY_DIR,
    LEGACY_URL,
    LIVE_DIR,
    LIVE_URL,
    PUBLIC,
    PUBLIC_URL,
    TILES,
    an_asset,
    directory_of,
    files_under,
    put_scan,
    stage,
    tileset_at,
    two_buckets,
)

TOKEN = "correct-horse-battery-staple"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def buckets() -> Iterator[Publisher]:
    yield from two_buckets()


def client_for(db: Session, buckets: Publisher, settings: Settings | None = None) -> TestClient:
    app = create_app(settings) if settings is not None else create_app()

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    app.dependency_overrides[get_storage] = lambda: buckets.private
    app.dependency_overrides[get_public_storage] = lambda: buckets.public
    return TestClient(app)


@pytest.fixture
def client(db: Session, buckets: Publisher) -> Iterator[TestClient]:
    with client_for(db, buckets) as test_client:
        yield test_client


def a_live_scan(buckets: Publisher) -> None:
    """A scan in a generation, already with a backfilled grid and an inferred fill."""
    put_scan(
        buckets.public,
        LIVE_DIR,
        sidecars={"collision.bin": b"grid", **FILL_FILES},
        extras={"collision": COLLISION_EXTRAS, "inferredLayers": FILL_EXTRAS},
    )


INSTANCES_FILES = {"instances.json": b'{"format":"hexapod.instances"}', "instances.emb": b"\0\1"}


def attach(
    client: TestClient, asset: Asset, prefix: str, **body: Any
) -> Any:  # the httpx2 response
    payload = {"stagingPrefix": prefix, "basedOn": asset.source["url"], **body}
    return client.post(f"/api/v1/assets/{asset.id}/sidecars", json=payload)


def head(key: str) -> dict[str, Any]:
    return dict(boto3.client("s3", region_name="us-east-1").head_object(Bucket=PUBLIC, Key=key))


# --- the attach itself ------------------------------------------------------------------


def test_an_attach_cuts_a_new_generation_with_everything_and_moves_the_asset(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    a_live_scan(buckets)
    asset = an_asset(db, LIVE_URL)
    before = files_under(buckets.public, LIVE_DIR)
    prefix = stage(buckets.private, asset.id, INSTANCES_FILES)

    response = attach(client, asset, prefix, extras={"instances": INSTANCES_EXTRAS})

    assert response.status_code == 200, response.text
    body = response.json()
    url = body["url"]
    assert url != LIVE_URL and body["previousUrl"] == LIVE_URL
    assert re.fullmatch(
        rf"{PUBLIC_URL}/runs/[^/]+/p{body['generation']}/package/splat/tileset\.json", url
    )
    assert body["asset"]["source"]["url"] == url
    db.expire_all()
    assert db.get(Asset, asset.id).source["url"] == url  # type: ignore[union-attr]

    # Tiles, the sidecars it had, and the new ones; nothing else.
    now = files_under(buckets.public, directory_of(url))
    assert set(now) == {*TILES, "collision.bin", *FILL_FILES, *INSTANCES_FILES, "tileset.json"}
    for name in (*TILES, "collision.bin", *FILL_FILES):
        assert now[name] == before[name], name
    for name, data in INSTANCES_FILES.items():
        assert now[name] == data
    assert body["attached"] == ["instances"]
    assert body["carried"] == ["collision", "inferredLayers"]
    assert body["copied"] == len(TILES) + 1 + len(FILL_FILES)

    # The merged root extras, and the tileset otherwise unchanged.
    document = tileset_at(buckets.public, url)
    extras = document["root"].pop("extras")
    assert extras == {
        "gaussians": 3,
        "collision": COLLISION_EXTRAS,
        "inferredLayers": FILL_EXTRAS,
        "instances": INSTANCES_EXTRAS,
    }
    old = json.loads(before["tileset.json"])
    old["root"].pop("extras")
    assert document == old
    assert body["extras"] == sorted(extras)

    # The live generation was not written, and the staged files are gone.
    assert files_under(buckets.public, LIVE_DIR) == before
    assert files_under(buckets.private, prefix) == {}


def test_everything_in_the_new_generation_is_immutable_and_labelled(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    a_live_scan(buckets)
    asset = an_asset(db, LIVE_URL)
    prefix = stage(buckets.private, asset.id, INSTANCES_FILES)
    url = attach(client, asset, prefix, extras={"instances": INSTANCES_EXTRAS}).json()["url"]
    directory = directory_of(url)
    types = {
        "tileset.json": "application/json",
        "instances.json": "application/json",
        "instances.emb": "application/octet-stream",
        "collision.bin": "application/octet-stream",
        "0/0.glb": "model/gltf-binary",
        "inferred/fixer/f0.glb": "model/gltf-binary",
    }
    for name, kind in types.items():
        found = head(directory + name)
        assert (found["CacheControl"], found["ContentType"]) == (IMMUTABLE_CACHE, kind), name
    assert all(is_published(directory + name) for name in files_under(buckets.public, directory))


class Ordered:
    """The public bucket, recording when each copy finished and when the root went up."""

    def __init__(self, inner: S3Storage) -> None:
        self.inner = inner
        self.events: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def available(self) -> bool:
        return True

    @property
    def bucket(self) -> str:
        return self.inner.bucket

    def copy_object(self, *args: Any, **kwargs: Any) -> Any:
        time.sleep(0.01)
        result = self.inner.copy_object(*args, **kwargs)
        with self.lock:
            self.events.append(("copy", args[2]))
        return result

    def put_object(self, key: str, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            self.events.append(("put", key))
        return self.inner.put_object(key, *args, **kwargs)


def test_the_tileset_is_written_last(db: Session, buckets: Publisher) -> None:
    a_live_scan(buckets)
    asset = an_asset(db, LIVE_URL)
    prefix = stage(buckets.private, asset.id, INSTANCES_FILES)
    assert isinstance(buckets.public, S3Storage)
    public = Ordered(buckets.public)
    result = attach_service.attach_sidecars(
        db,
        asset.id,
        SidecarAttach(
            staging_prefix=prefix, based_on=LIVE_URL, extras={"instances": INSTANCES_EXTRAS}
        ),
        staging=buckets.private,
        public=public,
    )
    assert public.events[-1] == ("put", result.url.removeprefix(f"{PUBLIC_URL}/"))
    assert [kind for kind, _ in public.events].count("put") == 1
    assert len(public.events) == result.copied + len(INSTANCES_FILES) + 1


def test_a_key_set_to_null_is_removed_and_a_directory_unit_is_replaced_whole(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """A smaller streamed-LOD build replaces all of `sog/`, so no stale chunk survives;
    `null` takes a key off the root."""
    put_scan(
        buckets.public,
        LIVE_DIR,
        sidecars={"sog/lod-meta.json": b"{}", "sog/0.webp": b"a", "sog/1.webp": b"b"},
        extras={"nativeLod": "sog/lod-meta.json", "stale": {"note": 1}},
    )
    asset = an_asset(db, LIVE_URL)
    prefix = stage(buckets.private, asset.id, {"sog/lod-meta.json": b"{2}", "sog/0.webp": b"A"})
    body = attach(client, asset, prefix, extras={"stale": None}).json()
    now = files_under(buckets.public, directory_of(body["url"]))
    assert {k: v for k, v in now.items() if k.startswith("sog/")} == {
        "sog/lod-meta.json": b"{2}",
        "sog/0.webp": b"A",
    }
    extras = tileset_at(buckets.public, body["url"])["root"]["extras"]
    assert "stale" not in extras and extras["nativeLod"] == "sog/lod-meta.json"
    assert body["attached"] == ["nativeLod"]


def test_the_rig_url_moves_with_the_url_and_an_attached_kind_clears_its_flag(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    put_scan(buckets.public, LIVE_DIR)
    asset = an_asset(db, LIVE_URL, clampToGround=True)
    asset.sidecar_flags = [
        sidecars.flag(
            "instances", action="Objects need re-segmenting", reason="x", job_id=None, at=_now()
        ),
        sidecars.flag("rig", action="Plants need re-rigging", reason="x", job_id=None, at=_now()),
        sidecars.flag("collision", action="Collision", reason="x", job_id=None, at=_now()),
    ]
    db.commit()
    prefix = stage(
        buckets.private,
        asset.id,
        {**INSTANCES_FILES, "rig.json": b"{}", "motion.json": b"{}", "plants.json": b"{}"},
    )
    body = attach(client, asset, prefix, extras={"instances": INSTANCES_EXTRAS}, rigUrl="rig.json")
    assert body.status_code == 200, body.text
    read = body.json()["asset"]
    assert read["renderConfig"]["rigUrl"] == "rig.json"
    assert read["renderConfig"]["clampToGround"] is True
    assert [flag["kind"] for flag in read["sidecarFlags"]] == ["collision"]
    assert sorted(body.json()["attached"]) == ["instances", "rig"]


def _now() -> Any:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC)


def test_a_second_fill_is_declared_beside_the_first_not_instead_of_it(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """`inferredLayers` is a list, and each fill workflow sends only its own entry: the
    attach merges the list by uri, so the second fill does not drop the first's."""
    a_live_scan(buckets)  # already declares inferred/fixer
    asset = an_asset(db, LIVE_URL)
    other = {**FILL_EXTRAS[0], "uri": "inferred/other/tileset.json"}
    files = {f"inferred/other/{k.rsplit('/', 1)[1]}": v for k, v in FILL_FILES.items()}
    body = attach(
        client, asset, stage(buckets.private, asset.id, files), extras={"inferredLayers": [other]}
    ).json()
    layers = tileset_at(buckets.public, body["url"])["root"]["extras"]["inferredLayers"]
    assert [layer["uri"] for layer in layers] == [
        "inferred/fixer/tileset.json",
        "inferred/other/tileset.json",
    ]
    # Sent again, an entry replaces its own, in place.
    moved = db.get(Asset, asset.id)
    assert moved is not None
    again = {**other, "evidence": {**other["evidence"], "views": 99}}
    body = attach(
        client,
        moved,
        stage(buckets.private, asset.id, files, "again"),
        extras={"inferredLayers": [again]},
    ).json()
    layers = tileset_at(buckets.public, body["url"])["root"]["extras"]["inferredLayers"]
    assert [layer["evidence"]["views"] for layer in layers] == [12, 99]


# --- what an attach replaces ----------------------------------------------------------

MATERIALS_EXTRAS = {"uri": "materials.json", "count": 2}
TELEMETRY_EXTRAS = {"uri": "telemetry.json", "streams": 1}


def a_scan_with_objects(buckets: Publisher) -> None:
    """A generation with objects (both files), the materials and telemetry keyed by their
    ids, and a backfilled grid."""
    put_scan(
        buckets.public,
        LIVE_DIR,
        sidecars={
            **INSTANCES_FILES,
            "materials.json": b'{"old": "materials"}',
            "telemetry.json": b'{"old": "telemetry"}',
            "collision.bin": b"grid",
        },
        extras={
            "instances": INSTANCES_EXTRAS,
            "materials": MATERIALS_EXTRAS,
            "telemetry": TELEMETRY_EXTRAS,
            "collision": COLLISION_EXTRAS,
        },
    )


def test_new_objects_drop_and_flag_what_was_keyed_by_the_old_ones(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """A re-segmentation renumbers the objects. The old `instances.emb` (rows are the old
    ids), `materials.json` and `telemetry.json` (keyed by them) would describe the wrong
    objects, so none of them is copied forward: the emb goes with the `instances.json` it
    belonged to, and materials and telemetry are dropped, keys and files, and flagged --
    as a republish that dropped the objects would (carry.py)."""
    a_scan_with_objects(buckets)
    asset = an_asset(db, LIVE_URL)
    prefix = stage(buckets.private, asset.id, {"instances.json": b'{"new": "objects"}'})

    response = attach(client, asset, prefix, extras={"instances": INSTANCES_EXTRAS})

    assert response.status_code == 200, response.text
    body = response.json()
    now = files_under(buckets.public, directory_of(body["url"]))
    assert set(now) == {*TILES, "instances.json", "collision.bin", "tileset.json"}
    assert now["instances.json"] == b'{"new": "objects"}'
    extras = tileset_at(buckets.public, body["url"])["root"]["extras"]
    assert set(extras) == {"gaussians", "instances", "collision"}
    assert body["attached"] == ["instances"]
    assert body["carried"] == ["collision"]
    assert body["dropped"] == ["materials", "telemetry"]
    assert body["removed"] == ["instances.emb", "materials.json", "telemetry.json"]
    flags = {flag["kind"]: flag for flag in body["asset"]["sidecarFlags"]}
    assert set(flags) == {"materials", "telemetry"}
    for name, flag in flags.items():
        assert flag["action"] == sidecars.KINDS_BY_NAME[name].action
        assert "an attach replaced instances" in flag["reason"]
        assert flag["jobId"] is None
    db.expire_all()
    stored = db.get(Asset, asset.id)
    assert stored is not None
    assert sorted(flag["kind"] for flag in stored.sidecar_flags) == ["materials", "telemetry"]


def test_objects_attached_with_their_materials_keep_both_and_flag_nothing(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """Sent in one request, the materials are the caller's word that they name the new
    ids: both land, nothing is dropped, and a flag either had is cleared."""
    put_scan(
        buckets.public,
        LIVE_DIR,
        sidecars={**INSTANCES_FILES, "materials.json": b'{"old": "materials"}'},
        extras={"instances": INSTANCES_EXTRAS, "materials": MATERIALS_EXTRAS},
    )
    asset = an_asset(db, LIVE_URL)
    asset.sidecar_flags = [
        sidecars.flag("materials", action="Materials", reason="x", job_id=None, at=_now())
    ]
    db.commit()
    new = {
        "instances.json": b'{"new": "objects"}',
        "instances.emb": b"\2\3",
        "materials.json": b'{"new": "materials"}',
    }
    prefix = stage(buckets.private, asset.id, new)

    response = attach(
        client,
        asset,
        prefix,
        extras={"instances": INSTANCES_EXTRAS, "materials": MATERIALS_EXTRAS},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    now = files_under(buckets.public, directory_of(body["url"]))
    assert {name: now[name] for name in new} == new
    extras = tileset_at(buckets.public, body["url"])["root"]["extras"]
    assert {"instances", "materials"} <= set(extras)
    assert body["attached"] == ["instances", "materials"]
    assert (body["dropped"], body["removed"]) == ([], [])
    assert body["asset"]["sidecarFlags"] == []


def test_an_independent_kind_leaves_the_objects_and_what_is_keyed_by_them_alone(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """An inferred layer is placed in the scan's frame and names no object: attaching one
    replaces nothing of the objects, the materials or the telemetry."""
    a_scan_with_objects(buckets)
    asset = an_asset(db, LIVE_URL)
    before = files_under(buckets.public, LIVE_DIR)
    prefix = stage(buckets.private, asset.id, FILL_FILES)

    response = attach(client, asset, prefix, extras={"inferredLayers": FILL_EXTRAS})

    assert response.status_code == 200, response.text
    body = response.json()
    now = files_under(buckets.public, directory_of(body["url"]))
    for name in (*INSTANCES_FILES, "materials.json", "telemetry.json", "collision.bin"):
        assert now[name] == before[name], name
    extras = tileset_at(buckets.public, body["url"])["root"]["extras"]
    assert extras["materials"] == MATERIALS_EXTRAS
    assert extras["telemetry"] == TELEMETRY_EXTRAS
    assert body["attached"] == ["inferredLayers"]
    assert body["carried"] == ["collision", "instances", "materials", "telemetry"]
    assert (body["dropped"], body["removed"]) == ([], [])
    assert body["asset"]["sidecarFlags"] == []


def test_a_kind_set_to_null_goes_with_its_files_and_what_was_keyed_by_it(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """`null` removes a kind, not just its key: undeclared files left behind would still
    be found beside the tileset. The materials keyed by the removed objects go too,
    flagged; the telemetry, set to null in the same request, goes unflagged."""
    a_scan_with_objects(buckets)
    asset = an_asset(db, LIVE_URL)
    prefix = f"staging/assets/{asset.id}/run-1/"

    response = attach(client, asset, prefix, extras={"instances": None, "telemetry": None})

    assert response.status_code == 200, response.text
    body = response.json()
    now = files_under(buckets.public, directory_of(body["url"]))
    assert set(now) == {*TILES, "collision.bin", "tileset.json"}
    assert set(tileset_at(buckets.public, body["url"])["root"]["extras"]) == {
        "gaussians",
        "collision",
    }
    assert body["dropped"] == ["materials"]
    assert [flag["kind"] for flag in body["asset"]["sidecarFlags"]] == ["materials"]
    assert "an attach removed instances" in body["asset"]["sidecarFlags"][0]["reason"]


# --- legacy prefixes -----------------------------------------------------------------


def test_a_legacy_prefix_is_attached_to_by_cutting_its_first_generation(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """The spool, pumpkin and camp were published before generations, and the workflows
    put their objects beside them in place. The first attach copies all of it -- tiles and
    those sidecars, with their extras -- into a generation, and leaves the prefix alone."""
    put_scan(
        buckets.public,
        LEGACY_DIR,
        sidecars={**INSTANCES_FILES, "collision.bin": b"grid"},
        extras={"instances": INSTANCES_EXTRAS, "collision": COLLISION_EXTRAS},
    )
    asset = an_asset(db, LEGACY_URL)
    before = files_under(buckets.public, LEGACY_DIR)
    prefix = stage(buckets.private, asset.id, FILL_FILES)

    body = attach(client, asset, prefix, extras={"inferredLayers": FILL_EXTRAS}).json()

    url = body["url"]
    job = LEGACY_DIR.split("/")[1]
    assert url == f"{PUBLIC_URL}/runs/{job}/p{body['generation']}/package/splat/tileset.json"
    now = files_under(buckets.public, directory_of(url))
    assert set(now) == {*TILES, *INSTANCES_FILES, "collision.bin", *FILL_FILES, "tileset.json"}
    assert set(tileset_at(buckets.public, url)["root"]["extras"]) == {
        "gaussians",
        "instances",
        "collision",
        "inferredLayers",
    }
    assert body["carried"] == ["collision", "instances"]
    assert files_under(buckets.public, LEGACY_DIR) == before


def test_a_second_attach_builds_on_the_first_even_when_based_on_the_legacy_url(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    """publish-instances.yml reads its scans' legacy URLs (infra/modal/segment.py SCANS).
    Once a fill has moved the asset to a generation, the instances computed against the
    legacy tiles still go on: the tiles are the same, so they are still true."""
    put_scan(buckets.public, LEGACY_DIR)
    asset = an_asset(db, LEGACY_URL)
    first = attach(client, asset, stage(buckets.private, asset.id, FILL_FILES, "fill"))
    assert first.status_code == 200
    second = client.post(
        f"/api/v1/assets/{asset.id}/sidecars",
        json={
            "stagingPrefix": stage(buckets.private, asset.id, INSTANCES_FILES, "seg"),
            "basedOn": LEGACY_URL,
            "extras": {"instances": INSTANCES_EXTRAS},
        },
    )
    assert second.status_code == 200, second.text
    assert second.json()["previousUrl"] == first.json()["url"]
    now = files_under(buckets.public, directory_of(second.json()["url"]))
    assert {*FILL_FILES, *INSTANCES_FILES} <= set(now)


def test_an_attach_computed_on_tiles_a_republish_replaced_is_refused(
    db: Session, buckets: Publisher, client: TestClient
) -> None:
    put_scan(buckets.public, LEGACY_DIR)
    put_scan(buckets.public, LIVE_DIR, tiles={name: b"refined " + name.encode() for name in TILES})
    asset = an_asset(db, LIVE_URL)
    before = files_under(buckets.public, "")
    response = client.post(
        f"/api/v1/assets/{asset.id}/sidecars",
        json={
            "stagingPrefix": stage(buckets.private, asset.id, INSTANCES_FILES),
            "basedOn": LEGACY_URL,
            "extras": {"instances": INSTANCES_EXTRAS},
        },
    )
    assert response.status_code == 409
    assert "no longer the ones" in response.json()["detail"]
    assert files_under(buckets.public, "") == before
    db.expire_all()
    assert db.get(Asset, asset.id).source["url"] == LIVE_URL  # type: ignore[union-attr]


# --- what may be staged ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "body", "words"),
    [
        ({"page.html": b"<script>"}, {}, "extension"),
        ({".hidden.json": b"{}"}, {}, "plain path"),
        ({"tileset.json": b"{}"}, {}, "written by the attach"),
        ({"0/0.glb": b"evil"}, {}, "one of the scan's tiles"),
        ({"objects/3/tileset.json": b"{}"}, {}, "rewrites the scan's tiles"),
        ({"instances.json": b"{}"}, {"files": ["instances.json", "instances.emb"]}, "missing"),
        ({"instances.json": b"{}"}, {"extras": {"gaussians": 9}}, "belongs to the tileset"),
        ({"instances.json": b"{}"}, {"extras": {"objects": []}}, "cannot be attached"),
        ({"a.json": b"{}"}, {"extras": {"instances": {"uri": "nope.json"}}}, "neither staged"),
        ({"a.json": b"{}"}, {"extras": {"instances": {"uri": "../x.json"}}}, "plain path"),
        ({"a.json": b"{}"}, {"rigUrl": "rig.json"}, "neither staged"),
        ({}, {}, "nothing to attach"),
    ],
)
def test_what_cannot_be_attached_is_refused_before_anything_is_copied(
    db: Session,
    buckets: Publisher,
    client: TestClient,
    files: dict[str, bytes],
    body: dict[str, Any],
    words: str,
) -> None:
    put_scan(buckets.public, LIVE_DIR)
    asset = an_asset(db, LIVE_URL)
    before = files_under(buckets.public, "")
    response = attach(client, asset, stage(buckets.private, asset.id, files), **body)
    assert response.status_code == 422, response.text
    assert words in response.json()["detail"]
    assert files_under(buckets.public, "") == before
    db.expire_all()
    assert db.get(Asset, asset.id).source["url"] == LIVE_URL  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "prefix",
    [
        "staging/assets/{other}/run-1/",  # another asset's files
        "staging/assets/{asset}/",  # no token
        "staging/assets/{asset}/a/b/",  # a token with a slash
        "runs/{asset}/run-1/",  # outside staging altogether
        "staging/assets/{asset}/run-1",  # not a directory
    ],
)
def test_only_this_assets_staging_area_is_read(
    db: Session, buckets: Publisher, client: TestClient, prefix: str
) -> None:
    put_scan(buckets.public, LIVE_DIR)
    asset = an_asset(db, LIVE_URL)
    filled = prefix.format(asset=asset.id, other=uuid.uuid4())
    buckets.private.put_object(filled.rstrip("/") + "/instances.json", b"{}", "application/json")
    response = attach(client, asset, filled)
    assert response.status_code == 422
    assert "stagingPrefix" in response.json()["detail"]


def test_a_sidecar_over_the_size_limit_is_refused(
    db: Session, buckets: Publisher, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    put_scan(buckets.public, LIVE_DIR)
    asset = an_asset(db, LIVE_URL)
    monkeypatch.setattr(sidecars, "MAX_SIDECAR_BYTES", 4)
    response = attach(client, asset, stage(buckets.private, asset.id, {"a.bin": b"12345"}))
    assert response.status_code == 422
    assert "at most" in response.json()["detail"]


def test_the_extensions_a_sidecar_may_have_are_ones_the_tile_proxy_serves() -> None:
    """The proxy 404s an extension not on its list, so a sidecar with one would be a dead
    file on the globe."""
    source = (REPO_ROOT / "functions" / "r2" / "[[path]].js").read_text()
    served = dict(re.findall(r'\["(\w+)", "([\w/.-]+)"\]', source))
    for extension, kind in sidecars.SIDECAR_TYPES.items():
        assert served.get(extension) == kind, extension


@pytest.mark.parametrize(
    ("source", "status"),
    [
        ({"type": "cesium-ion", "assetId": 7}, 409),
        ({"type": "3d-tiles-url", "url": "https://elsewhere.example.com/x/tileset.json"}, 409),
        ({"type": "3d-tiles-url", "url": f"{PUBLIC_URL}/sites/camp/tileset.json"}, 409),
    ],
)
def test_an_asset_that_is_not_a_run_tileset_in_the_public_bucket_has_no_generation(
    db: Session, buckets: Publisher, client: TestClient, source: dict[str, Any], status: int
) -> None:
    asset = an_asset(db, LIVE_URL)
    asset.source = source
    db.commit()
    prefix = stage(buckets.private, asset.id, INSTANCES_FILES)
    response = client.post(
        f"/api/v1/assets/{asset.id}/sidecars",
        json={"stagingPrefix": prefix, "basedOn": LIVE_URL},
    )
    assert response.status_code == status


def test_an_unknown_asset_is_a_404(buckets: Publisher, client: TestClient) -> None:
    missing = uuid.uuid4()
    response = client.post(
        f"/api/v1/assets/{missing}/sidecars",
        json={"stagingPrefix": f"staging/assets/{missing}/r/", "basedOn": LIVE_URL},
    )
    assert response.status_code == 404


# --- who may attach -------------------------------------------------------------------


def test_an_attach_needs_the_write_token(db: Session, buckets: Publisher) -> None:
    put_scan(buckets.public, LIVE_DIR)
    asset = an_asset(db, LIVE_URL)
    prefix = stage(buckets.private, asset.id, INSTANCES_FILES)
    payload = {"stagingPrefix": prefix, "basedOn": LIVE_URL}
    with client_for(db, buckets, Settings(api_write_token=TOKEN)) as client:
        refused = client.post(f"/api/v1/assets/{asset.id}/sidecars", json=payload)
        assert refused.status_code == 401
        assert files_under(buckets.private, prefix)  # nothing read, nothing deleted
        wrong = client.post(
            f"/api/v1/assets/{asset.id}/sidecars",
            json=payload,
            headers={"Authorization": "Bearer wrong"},
        )
        assert wrong.status_code == 401
        accepted = client.post(f"/api/v1/assets/{asset.id}/sidecars", json=payload, headers=AUTH)
        assert accepted.status_code == 200, accepted.text


def test_without_object_storage_an_attach_is_a_503(db: Session, buckets: Publisher) -> None:
    from app.storage import NullStorage

    asset = an_asset(db, LIVE_URL)
    with client_for(db, buckets) as client:
        client.app.dependency_overrides[get_public_storage] = NullStorage  # type: ignore[attr-defined]
        response = client.post(
            f"/api/v1/assets/{asset.id}/sidecars",
            json={"stagingPrefix": f"staging/assets/{asset.id}/r/", "basedOn": LIVE_URL},
        )
    assert response.status_code == 503


# --- two at once ----------------------------------------------------------------------


class Gate:
    """The public bucket, whose first copy waits until the test opens the gate."""

    def __init__(self, inner: S3Storage) -> None:
        self.inner = inner
        self.reached = threading.Event()
        self.open = threading.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def available(self) -> bool:
        return True

    @property
    def bucket(self) -> str:
        return self.inner.bucket

    def copy_object(self, *args: Any, **kwargs: Any) -> Any:
        self.reached.set()
        assert self.open.wait(timeout=30), "the gate was never opened"
        return self.inner.copy_object(*args, **kwargs)


def waiting_on_a_lock(engine: Engine) -> bool:
    with engine.connect() as connection:
        return bool(
            connection.execute(text("SELECT count(*) FROM pg_locks WHERE NOT granted")).scalar()
        )


def test_two_attaches_at_once_both_land_the_second_on_the_first(
    db: Session, buckets: Publisher, engine: Engine
) -> None:
    """The instances workflow and the fill workflow, started from the same scan. Without
    the row lock both read the same generation and the second repoints the asset at a copy
    without the first's files. With it, the second waits, then builds on the first."""
    put_scan(buckets.public, LEGACY_DIR)
    asset = an_asset(db, LEGACY_URL)
    asset_id = asset.id
    assert isinstance(buckets.public, S3Storage)
    slow = Gate(buckets.public)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    results: dict[str, Any] = {}

    def run(name: str, public: Any, files: dict[str, bytes], extras: dict[str, Any]) -> None:
        with factory() as session:
            try:
                results[name] = attach_service.attach_sidecars(
                    session,
                    asset_id,
                    SidecarAttach(
                        staging_prefix=stage(buckets.private, asset_id, files, name),
                        based_on=LEGACY_URL,
                        extras=extras,
                    ),
                    staging=buckets.private,
                    public=public,
                )
            except Exception as error:  # surfaced below
                results[name] = error

    first = threading.Thread(
        target=run, args=("fill", slow, FILL_FILES, {"inferredLayers": FILL_EXTRAS})
    )
    first.start()
    assert slow.reached.wait(timeout=30)  # the first holds the lock, mid-copy
    second = threading.Thread(
        target=run,
        args=("seg", buckets.public, INSTANCES_FILES, {"instances": INSTANCES_EXTRAS}),
    )
    second.start()
    deadline = time.monotonic() + 30
    while not waiting_on_a_lock(engine):
        assert time.monotonic() < deadline, "the second attach never waited for the first"
        time.sleep(0.05)
    slow.open.set()
    first.join(timeout=60)
    second.join(timeout=60)

    fill, seg = results["fill"], results["seg"]
    assert not isinstance(fill, Exception), fill
    assert not isinstance(seg, Exception), seg
    assert fill.previous_url == LEGACY_URL
    assert seg.previous_url == fill.url  # built on the first, not on what both read
    db.expire_all()
    assert db.get(Asset, asset_id).source["url"] == seg.url  # type: ignore[union-attr]
    extras = tileset_at(buckets.public, seg.url)["root"]["extras"]
    assert {"inferredLayers", "instances"} <= set(extras)
    assert {*FILL_FILES, *INSTANCES_FILES} <= set(
        files_under(buckets.public, directory_of(seg.url))
    )


def test_an_attach_that_cannot_get_the_lock_in_time_is_a_409(
    db: Session, buckets: Publisher, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    put_scan(buckets.public, LIVE_DIR)
    asset = an_asset(db, LIVE_URL)
    monkeypatch.setattr(attach_service, "LOCK_WAIT", "200ms")
    with engine.connect() as holder:
        transaction = holder.begin()
        holder.execute(text("SELECT id FROM assets WHERE id = :id FOR UPDATE"), {"id": asset.id})
        try:
            with pytest.raises(ConflictError, match="in progress"):
                attach_service.attach_sidecars(
                    db,
                    asset.id,
                    SidecarAttach(
                        staging_prefix=stage(buckets.private, asset.id, INSTANCES_FILES),
                        based_on=LIVE_URL,
                    ),
                    staging=buckets.private,
                    public=buckets.public,
                )
        finally:
            transaction.rollback()


# --- the classification ---------------------------------------------------------------


def test_every_kind_is_classified_by_what_it_depends_on() -> None:
    """docs/SCENE_OBJECTS.md section 8 prints this table; the code is what decides."""
    depends = {kind.name: kind.depends.value for kind in sidecars.KINDS}
    assert depends == {
        "instances": "positions",
        "skin": "positions",
        "materials": "instances",
        "telemetry": "instances",
        "objects": "tiles",
        "collision": "tiles",
        "viewCones": "tiles",
        "inferredLayers": "frame",
        "nativeLod": "tiles",
        "rig": "positions",
    }
    assert {kind.name for kind in sidecars.KINDS if not kind.attachable} == {"objects"}
    # A kind keyed by ids says whose; an attach that replaces those drops it.
    follows = {kind.name: kind.follows for kind in sidecars.KINDS if kind.follows}
    assert follows == {"materials": "instances", "telemetry": "instances"}
    assert all(
        (kind.depends is sidecars.Dependence.INSTANCES) == (kind.follows is not None)
        for kind in sidecars.KINDS
    )
    assert sidecars.followers(["instances"]) == ["materials", "telemetry"]
    assert sidecars.followers(["inferredLayers", "collision"]) == []
    # Every kind bound to positions says where it lists them, in a file it owns.
    for kind in sidecars.KINDS:
        assert bool(kind.checksums) == (kind.depends is sidecars.Dependence.POSITIONS), kind
        assert all(kind.owns(file) for file, _ in kind.checksums), kind


def test_discovery_finds_each_kind_by_its_key_or_its_files() -> None:
    extras = {
        "gaussians": 3,
        "instances": INSTANCES_EXTRAS,
        "inferredLayers": FILL_EXTRAS,
        "mystery": {"uri": "m.bin"},
    }
    rels = [
        "instances.json",
        "instances.emb",
        "inferred/fixer/tileset.json",
        "inferred/fixer/f0.glb",
        "sog/lod-meta.json",
        "sog/0.webp",
        "rig.json",
        "m.bin",
    ]
    assert sidecars.discover(extras, rels) == {
        "instances": ("instances.emb", "instances.json"),
        "inferredLayers": ("inferred/fixer/f0.glb", "inferred/fixer/tileset.json"),
        "nativeLod": ("sog/0.webp", "sog/lod-meta.json"),
        "rig": ("rig.json",),
    }
    assert sidecars.unknown_extras(extras) == ["mystery"]
