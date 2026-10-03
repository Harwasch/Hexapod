"""Carry-forward: a worker republish keeps the sidecars that still hold, and flags the rest.

A run's own outputs are its tiles, `collision.bin` and `viewcones.bin`. What was attached
beside the live tiles since -- objects, an inferred fill, the streamed LOD, a plant rig --
used to vanish when a republish repointed the asset at the run's new generation. Now each
kind is carried by what it depends on (app/services/sidecars.py, app/worker/carry.py):
bound to the splats, only onto the very same tiles; keyed by instance ids, with
`instances`; in the scan's frame only (an inferred fill), always. Objects, skins and a rig
are bound to the splats' positions tile by tile, so for them "the same splats" is checked
the way the viewer checks it: every new tile's position checksum must be one the binding
lists -- a re-pack with another spherical-harmonics degree keeps them. What is dropped is
flagged on the asset, where the console and the API show it.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from app.models import Asset, Capture
from app.models.enums import CaptureKind, CaptureStatus, GeorefMethod, ScaleSource
from app.schemas.sidecar import SidecarAttach
from app.schemas.site import SiteCreate
from app.services import sites as site_service
from app.services.assets import asset_to_read
from app.services.attach import attach_sidecars
from app.storage import S3Storage
from app.worker import positions, registration
from app.worker.carry import plan_carry
from app.worker.claim import claim_next
from app.worker.publish import Publisher
from app.worker.registration import LiveMoved, Registration, publish_outputs, register
from tests.conftest import site_payload
from tests.sidecar_fixtures import (
    COLLISION_EXTRAS,
    FILL_EXTRAS,
    FILL_FILES,
    INSTANCES_EXTRAS,
    LEGACY_DIR,
    LEGACY_URL,
    PUBLIC_URL,
    TILES,
    TREE_LOD,
    TREE_SH,
    directory_of,
    files_under,
    fixture_scan,
    put_scan,
    splat_glb,
    stage,
    tileset_at,
    two_buckets,
)
from tests.test_worker import config, make_capture, queue_job, run_job

RUN = uuid.UUID("00000000-0000-0000-0000-0000000000e1")
RUN_DIR = f"runs/{RUN}/package/splat/"
INSTANCES_FILES = {"instances.json": b'{"tiles":{}}', "instances.emb": b"\0\1"}
SOG_FILES = {"sog/lod-meta.json": b"{}", "sog/0.webp": b"webp"}
RIG_FILES = {"rig.json": b"{}", "motion.json": b"{}", "plants.json": b"{}"}
MATERIALS_EXTRAS = {"uri": "materials.json", "count": 1}


@pytest.fixture
def buckets() -> Iterator[Publisher]:
    yield from two_buckets()


def a_registration() -> Registration:
    return Registration(
        slug="camp",
        title="Camp",
        recipe="splat-ingest",
        lat=0.0,
        lon=0.0,
        height=0.0,
        georef_method=GeorefMethod.NONE,
        scale_source=ScaleSource.UNRESOLVED,
        uncertainty_m=0.0,
        document={},
    )


def a_run(buckets: Publisher, *, tag: bytes = b"", collision: bool = True) -> None:
    """The run's packaged tileset in the private bucket, as the packer writes it."""
    if not collision:
        buckets.private.delete_object(f"{RUN_DIR}collision.bin")
    put_scan(
        buckets.private,
        RUN_DIR,
        tiles={name: b"glb " + tag + name.encode() for name in TILES},
        sidecars={"collision.bin": b"grid " + tag} if collision else {},
        extras={"collision": COLLISION_EXTRAS} if collision else {},
    )


def republish(db: Session, buckets: Publisher, capture: Capture) -> uuid.UUID | None:
    return register(
        db,
        buckets.private,
        publish=buckets,
        capture=capture,
        job_id=RUN,
        registration=a_registration(),
        tiles_stage_id="package",
    )


def the_splat(db: Session, capture: Capture) -> Asset:
    db.expire_all()
    found = db.get(Capture, capture.id)
    assert found is not None and found.site_id is not None
    asset = registration.splat_asset(db, found.site_id)
    assert asset is not None
    return asset


def attach(
    db: Session, buckets: Publisher, asset: Asset, files: dict[str, bytes], **body: Any
) -> str:
    result = attach_sidecars(
        db,
        asset.id,
        SidecarAttach(
            staging_prefix=stage(buckets.private, asset.id, files, uuid.uuid4().hex),
            based_on=asset.source["url"],
            **body,
        ),
        staging=buckets.private,
        public=buckets.public,
    )
    return result.url


def a_scan_with_everything(db: Session, buckets: Publisher) -> tuple[Capture, Asset]:
    """A run published and registered, then objects, materials, a fill, the streamed LOD
    and a plant rig attached beside it."""
    a_run(buckets)
    capture = Capture(slug="camp", name="Camp", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    republish(db, buckets, capture)
    asset = the_splat(db, capture)
    attach(
        db,
        buckets,
        asset,
        {
            **INSTANCES_FILES,
            "materials.json": b"{}",
            **FILL_FILES,
            **SOG_FILES,
            **RIG_FILES,
        },
        extras={
            "instances": INSTANCES_EXTRAS,
            "materials": MATERIALS_EXTRAS,
            "inferredLayers": FILL_EXTRAS,
            "nativeLod": "sog/lod-meta.json",
        },
        rig_url="rig.json",
    )
    return capture, the_splat(db, capture)


def test_a_republish_of_the_same_tiles_carries_every_sidecar(
    db: Session, buckets: Publisher
) -> None:
    """A retried register, a worker that lost its lease after publishing: the same bytes,
    so everything attached beside them is still true of them."""
    capture, asset = a_scan_with_everything(db, buckets)
    live = asset.source["url"]

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    url = moved.source["url"]
    assert url != live  # a generation of its own: it carries what the run's first did not
    files = files_under(buckets.public, directory_of(url))
    assert {*INSTANCES_FILES, "materials.json", *FILL_FILES, *SOG_FILES, *RIG_FILES} <= set(files)
    extras = tileset_at(buckets.public, url)["root"]["extras"]
    assert set(extras) == {
        "gaussians",
        "collision",
        "instances",
        "materials",
        "inferredLayers",
        "nativeLod",
    }
    assert moved.render_config["rigUrl"] == "rig.json"
    assert moved.sidecar_flags == []


def test_a_republish_of_new_tiles_drops_what_was_bound_to_the_old_and_flags_it(
    db: Session,
    buckets: Publisher,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Refine: new splats. Objects (and the materials keyed by their ids), the streamed
    LOD and the rig were computed on the old splats and are dropped, each with a flag; the
    inferred fill is in the scan's frame and is carried; the run's own collision grid
    replaces the old one."""
    capture, _ = a_scan_with_everything(db, buckets)
    a_run(buckets, tag=b"refined ")
    # alembic's fileConfig (the `engine` fixture) disables the loggers that exist by then.
    monkeypatch.setattr(logging.getLogger("app.worker"), "disabled", False)

    with caplog.at_level(logging.WARNING, logger="app.worker"):
        republish(db, buckets, capture)

    moved = the_splat(db, capture)
    url = moved.source["url"]
    files = files_under(buckets.public, directory_of(url))
    assert set(files) == {*TILES, "collision.bin", *FILL_FILES, "tileset.json"}
    assert files["collision.bin"] == b"grid refined "  # the run's own
    assert files["0/0.glb"] == b"glb refined 0/0.glb"
    extras = tileset_at(buckets.public, url)["root"]["extras"]
    assert set(extras) == {"gaussians", "collision", "inferredLayers"}
    assert extras["inferredLayers"] == FILL_EXTRAS
    assert "rigUrl" not in moved.render_config

    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert set(flags) == {"instances", "materials", "nativeLod", "rig"}
    assert flags["instances"]["action"] == "Objects need re-segmenting"
    assert "new tiles" in flags["instances"]["reason"]
    assert "instances was not carried" in flags["materials"]["reason"]
    assert flags["instances"]["jobId"] == str(RUN)
    read = asset_to_read(moved)
    assert {flag.kind for flag in read.sidecar_flags} == set(flags)
    assert "Objects need re-segmenting" in caplog.text

    # Segmenting again clears exactly that flag.
    attach(db, buckets, moved, INSTANCES_FILES, extras={"instances": INSTANCES_EXTRAS})
    assert {f["kind"] for f in the_splat(db, capture).sidecar_flags} == {
        "materials",
        "nativeLod",
        "rig",
    }


def test_a_republish_from_a_legacy_prefix_carries_the_fill_and_flags_the_rest(
    db: Session, buckets: Publisher
) -> None:
    """The camp as it is: published before generations, objects, a fill and a collision
    backfill written beside it in place. Its first republish, from a run whose packer wrote
    no grid, keeps the fill and says what has to be redone."""
    put_scan(
        buckets.public,
        LEGACY_DIR,
        sidecars={**INSTANCES_FILES, "collision.bin": b"grid", **FILL_FILES},
        extras={
            "instances": INSTANCES_EXTRAS,
            "collision": COLLISION_EXTRAS,
            "inferredLayers": FILL_EXTRAS,
            "mystery": {"note": "nobody knows"},
        },
    )
    before = files_under(buckets.public, LEGACY_DIR)
    site = site_service.create_site(
        db,
        SiteCreate.model_validate(
            site_payload(
                slug="camp",
                assets=[
                    {
                        "name": "Camp splat",
                        "representation": "gaussian-splat",
                        "source": {"type": "3d-tiles-url", "url": LEGACY_URL},
                    }
                ],
            )
        ),
    )
    capture = Capture(slug="camp", name="Camp", kind=CaptureKind.GAUSSIAN_SPLAT, site_id=site.id)
    db.add(capture)
    db.commit()
    a_run(buckets, tag=b"new ", collision=False)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    url = moved.source["url"]
    assert url.startswith(f"{PUBLIC_URL}/runs/{RUN}/p")
    files = files_under(buckets.public, directory_of(url))
    assert set(files) == {*TILES, *FILL_FILES, "tileset.json"}
    extras = tileset_at(buckets.public, url)["root"]["extras"]
    assert set(extras) == {"gaussians", "inferredLayers"}
    flags = {flag["kind"]: flag["action"] for flag in moved.sidecar_flags}
    assert flags == {
        "instances": "Objects need re-segmenting",
        "collision": "Collision needs a backfill",
        "mystery": "Unrecognised sidecar extras.mystery needs re-attaching",
    }
    assert files_under(buckets.public, LEGACY_DIR) == before
    assert the_splat(db, capture).id == moved.id


def test_a_run_that_brings_its_own_kind_clears_that_kinds_flag(
    db: Session, buckets: Publisher
) -> None:
    capture, _ = a_scan_with_everything(db, buckets)
    a_run(buckets, tag=b"refined ", collision=False)
    republish(db, buckets, capture)
    assert "collision" in {f["kind"] for f in the_splat(db, capture).sidecar_flags}
    a_run(buckets, tag=b"again ", collision=True)
    republish(db, buckets, capture)
    assert "collision" not in {f["kind"] for f in the_splat(db, capture).sidecar_flags}


def test_register_does_not_repoint_an_asset_an_attach_moved_while_it_published(
    db: Session, buckets: Publisher
) -> None:
    """The publish carried from one generation; an attach cut another meanwhile.
    Repointing would drop what the attach added, so nothing is written."""
    capture, asset = a_scan_with_everything(db, buckets)
    live = asset.source["url"]
    a_run(buckets, tag=b"refined ")
    published = publish_outputs(
        buckets.private,
        publish=buckets,
        job_id=RUN,
        registration=a_registration(),
        tiles_stage_id="package",
        carry_from=live,
    )
    assert published.carry.based_on == live
    attached = attach(db, buckets, asset, {"telemetry.json": b"{}"}, extras={"telemetry": {}})

    with pytest.raises(LiveMoved) as moved:
        register(
            db,
            buckets.private,
            publish=buckets,
            capture=capture,
            job_id=RUN,
            registration=a_registration(),
            tiles_stage_id="package",
            published=published,
        )
    db.rollback()
    assert moved.value.url == attached
    assert the_splat(db, capture).source["url"] == attached


def test_an_unreadable_live_generation_withholds_the_publish(
    db: Session, buckets: Publisher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What it carries is unknown, and dropping every sidecar because the store had a bad
    minute would take objects off a site with nothing wrong with it."""
    _, asset = a_scan_with_everything(db, buckets)
    live = asset.source["url"]
    assert isinstance(buckets.public, S3Storage)

    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("R2 is having a moment")

    monkeypatch.setattr(buckets.public, "list_objects", broken)
    published = publish_outputs(
        buckets.private,
        publish=buckets,
        job_id=RUN,
        registration=a_registration(),
        tiles_stage_id="package",
        carry_from=live,
    )
    assert published.withheld and published.tileset is None


def test_a_plan_from_a_url_outside_the_bucket_carries_nothing(buckets: Publisher) -> None:
    a_run(buckets)
    plan = plan_carry(
        buckets,
        live_url="https://elsewhere.example.com/x/tileset.json",
        tiles_prefix=RUN_DIR,
        entry=f"{RUN_DIR}tileset.json",
    )
    assert (plan.based_on, plan.objects, plan.dropped, plan.document) == (
        "https://elsewhere.example.com/x/tileset.json",
        (),
        (),
        None,
    )


def test_with_one_bucket_a_republish_drops_and_flags_what_it_cannot_carry(
    buckets: Publisher,
) -> None:
    one = Publisher(private=buckets.private, public=buckets.private)
    put_scan(
        buckets.private,
        f"runs/{uuid.uuid4()}/package/splat/",
        sidecars=FILL_FILES,
        extras={"inferredLayers": FILL_EXTRAS},
    )
    live_dir = next(
        item.key for item in buckets.private.list_objects("runs/").objects if "inferred" in item.key
    ).split("inferred/")[0]
    a_run(buckets)
    plan = plan_carry(
        one,
        live_url=buckets.private.public_url(live_dir + "tileset.json"),
        tiles_prefix=RUN_DIR,
        entry=f"{RUN_DIR}tileset.json",
    )
    assert [gone.kind for gone in plan.dropped] == ["inferredLayers"]
    assert "one bucket" in plan.dropped[0].reason


def test_a_split_fill_goes_with_the_split(buckets: Publisher) -> None:
    """A split declares its fills in `inferredLayers`, under `fills/`: bound to the split,
    which rewrote the old tiles, so they are not carried when the tiles change."""
    live_dir = f"runs/{RUN}/pfedcba9876543210/package/splat/"
    split_fill = {"uri": "fills/3/tileset.json", "evidence": FILL_EXTRAS[0]["evidence"]}
    put_scan(
        buckets.public,
        live_dir,
        sidecars={**FILL_FILES, "fills/3/tileset.json": b"{}", "objects/3/tileset.json": b"{}"},
        extras={"inferredLayers": [*FILL_EXTRAS, split_fill], "objects": [{"uri": "x"}]},
    )
    a_run(buckets, tag=b"new ")
    plan = plan_carry(
        buckets,
        live_url=f"{PUBLIC_URL}/{live_dir}tileset.json",
        tiles_prefix=RUN_DIR,
        entry=f"{RUN_DIR}tileset.json",
    )
    assert sorted(rel for _, rel in plan.objects) == sorted(FILL_FILES)
    assert plan.document is not None
    extras = json.loads(plan.document)["root"]["extras"]
    assert extras["inferredLayers"] == FILL_EXTRAS
    assert "objects" in {gone.kind for gone in plan.dropped}


# --- bound to positions: the viewer's own test ---------------------------------------------

#: What a binding to the fixture tree's positions holds, beside the tiles: objects keyed by
#: every tile's checksum, materials by their ids, the rig tools/captures stamped with the same
#: checksums and a plant binding keyed by them -- and the streamed LOD, which is the old
#: splats themselves and holds only for the same bytes.
TREE_CHECKSUMS: list[str] = json.loads((TREE_LOD / "rig.json").read_text())["tileChecksums"]
TREE_BOUND = {
    "instances.json": json.dumps({"tiles": {c: [1, 1] for c in TREE_CHECKSUMS}}).encode(),
    "instances.emb": b"\0\1",
    "materials.json": b"{}",
    "rig.json": (TREE_LOD / "rig.json").read_bytes(),
    "motion.json": b"{}",
    "plants.json": json.dumps({"tiles": {c: [0, 1] for c in TREE_CHECKSUMS}}).encode(),
    **SOG_FILES,
}
POSITION_BOUND = {"instances", "materials", "rig"}


def a_tree_run(buckets: Publisher, tiles: dict[str, bytes], document: dict[str, Any]) -> None:
    """The run's package from a committed fixture tree, with its own grid and view cones."""
    put_scan(
        buckets.private,
        RUN_DIR,
        tiles=tiles,
        sidecars={"collision.bin": b"grid", "viewcones.bin": b"cones"},
        document=document,
    )


def a_tree_with_objects(db: Session, buckets: Publisher) -> Capture:
    """The fixture tree published and registered, then objects, materials, the streamed LOD
    and a plant rig attached beside it."""
    document, tiles = fixture_scan(TREE_LOD)
    a_tree_run(buckets, tiles, document)
    capture = Capture(slug="tree", name="Tree", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    republish(db, buckets, capture)
    attach(
        db,
        buckets,
        the_splat(db, capture),
        TREE_BOUND,
        extras={
            "instances": INSTANCES_EXTRAS,
            "materials": MATERIALS_EXTRAS,
            "nativeLod": "sog/lod-meta.json",
        },
        rig_url="rig.json",
    )
    return capture


def test_a_repack_that_keeps_every_position_keeps_what_is_bound_to_positions(
    db: Session, buckets: Publisher
) -> None:
    """The tree packed again with spherical harmonics: every tile's bytes are new, so the
    byte fingerprint differs, and every tile's positions are the ones the objects, the
    materials and the rig were bound to. They are carried; the streamed LOD, which is the
    old encoding of the splats, is not."""
    capture = a_tree_with_objects(db, buckets)
    document, sh = fixture_scan(TREE_SH)
    a_tree_run(buckets, sh, document)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert files["splat.glb"] == sh["splat.glb"]
    bound = {rel for rel in TREE_BOUND if not rel.startswith("sog/")}
    assert {rel: files[rel] for rel in bound} == {rel: TREE_BOUND[rel] for rel in bound}
    assert not any(rel.startswith("sog/") for rel in files)
    extras = tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert {"instances", "materials", "collision", "viewCones"} <= set(extras)
    assert "nativeLod" not in extras
    assert moved.render_config["rigUrl"] == "rig.json"
    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert set(flags) == {"nativeLod"}
    assert flags["nativeLod"]["action"] == "Streamed LOD needs a backfill"


def test_moved_positions_drop_what_was_bound_to_them_and_flag_it(
    db: Session, buckets: Publisher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One splat of the root tile a quantum higher: that tile's checksum is in no binding,
    so the viewer would refuse it, and the objects, the materials keyed by their ids and the
    rig all go, each flagged. Nothing past that first tile is hashed."""
    capture = a_tree_with_objects(db, buckets)
    document, sh = fixture_scan(TREE_SH)
    nudged = positions.tile_positions(sh["splat.glb"]).astype("float64")
    nudged[0, 2] += 1 / 4096
    sh["splat.glb"] = splat_glb(nudged)
    assert positions.tile_checksum(sh["splat.glb"]) not in TREE_CHECKSUMS
    a_tree_run(buckets, sh, document)
    hashed: list[bytes] = []
    real = positions.tile_checksum

    def counting(data: bytes) -> str:
        hashed.append(data)
        return real(data)

    monkeypatch.setattr(positions, "tile_checksum", counting)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert not set(TREE_BOUND) & set(files)
    assert "rigUrl" not in moved.render_config
    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert set(flags) == POSITION_BOUND | {"nativeLod"}
    assert flags["instances"]["action"] == "Objects need re-segmenting"
    assert "positions are not all ones it was bound to" in flags["instances"]["reason"]
    assert flags["rig"]["action"] == "Plants need re-rigging"
    assert "instances was not carried" in flags["materials"]["reason"]
    assert hashed == [sh["splat.glb"]]


def test_positions_are_read_from_the_workdir_where_the_worker_has_them(
    db: Session, buckets: Publisher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = a_tree_with_objects(db, buckets)
    live = the_splat(db, capture).source["url"]
    document, sh = fixture_scan(TREE_SH)
    a_tree_run(buckets, sh, document)
    for uri, data in sh.items():
        (tmp_path / uri).write_bytes(data)
    assert isinstance(buckets.private, S3Storage)
    real = buckets.private.get_object

    def no_tiles(key: str) -> bytes:
        assert not key.endswith(".glb"), key
        return real(key)

    monkeypatch.setattr(buckets.private, "get_object", no_tiles)

    plan = plan_carry(
        buckets,
        live_url=live,
        tiles_prefix=RUN_DIR,
        entry=f"{RUN_DIR}tileset.json",
        tiles_dir=tmp_path,
    )

    assert set(plan.carried) >= POSITION_BOUND
    assert [gone.kind for gone in plan.dropped] == ["nativeLod"]


def test_a_tile_that_is_not_a_splat_vouches_for_no_binding(db: Session, buckets: Publisher) -> None:
    capture = a_tree_with_objects(db, buckets)
    live = the_splat(db, capture).source["url"]
    document, sh = fixture_scan(TREE_SH)
    sh["splat_7.glb"] = b"not a glb"
    a_tree_run(buckets, sh, document)

    plan = plan_carry(buckets, live_url=live, tiles_prefix=RUN_DIR, entry=f"{RUN_DIR}tileset.json")

    assert {gone.kind for gone in plan.dropped} == POSITION_BOUND | {"nativeLod"}


# --- the runner: an attach that lands mid-publish ----------------------------------------


def test_the_runner_publishes_again_when_an_attach_lands_mid_publish(
    db: Session,
    sessions: sessionmaker[Session],
    storage: S3Storage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second run of a capture publishes; before it registers, an attach moves the
    site. `register` refuses (`LiveMoved`), and the runner publishes again from where the
    site now is, then registers -- and the run completes."""
    capture = make_capture(db, slug="orchard")
    first = queue_job(db, capture, "t-ingest")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()
    assert run_job(sessions, storage, first.id, config(tmp_path)) == "complete"

    publishes: list[str | None] = []
    real_publish, real_register = registration.publish_outputs, registration.register
    elsewhere = "https://elsewhere.example.com/attached/tileset.json"

    def publish(*args: Any, **kwargs: Any) -> Any:
        publishes.append(kwargs.get("carry_from"))
        return real_publish(*args, **kwargs)

    def register_after_an_attach(*args: Any, **kwargs: Any) -> Any:
        if len(publishes) == 1:
            with sessions() as other:  # an attach, from the API, in between
                other.execute(
                    update(Asset).values(source={"type": "3d-tiles-url", "url": elsewhere})
                )
                other.commit()
        return real_register(*args, **kwargs)

    monkeypatch.setattr(registration, "publish_outputs", publish)
    monkeypatch.setattr(registration, "register", register_after_an_attach)

    second = queue_job(db, capture, "t-ingest")
    session = sessions()
    assert claim_next(session, worker_id="worker-a", lease_s=30) is not None
    session.close()
    assert run_job(sessions, storage, second.id, config(tmp_path, worker_id="worker-a")) == (
        "complete"
    )

    assert len(publishes) == 2
    assert str(publishes[0]).endswith(f"runs/{first.id}/package/splat/tileset.json")
    assert publishes[1] == elsewhere
    db.expire_all()
    done = db.get(Capture, capture.id)
    assert done is not None and done.status is CaptureStatus.COMPLETE and done.site_id
    splat = registration.splat_asset(db, done.site_id)
    assert splat is not None
    assert str(splat.source["url"]).endswith(f"runs/{second.id}/package/splat/tileset.json")
