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

import dataclasses
import json
import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select, update
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
    GRID_EXTRAS,
    GRID_FILES,
    INSTANCES_EXTRAS,
    LEGACY_DIR,
    LEGACY_URL,
    PUBLIC_URL,
    TILES,
    TRANSFORM,
    TREE_LOD,
    TREE_SH,
    UNGATED_FILL_EXTRAS,
    admit_fills,
    directory_of,
    files_under,
    fixture_scan,
    gate_for,
    put_scan,
    splat_glb,
    stage,
    tileset_at,
    two_buckets,
    waive_fill_gate,
)
from tests.test_worker import config, make_capture, queue_job, run_job

RUN = uuid.UUID("00000000-0000-0000-0000-0000000000e1")
RUN_DIR = f"runs/{RUN}/package/splat/"
INSTANCES_FILES = {"instances.json": b'{"tiles":{}}', "instances.emb": b"\0\1"}
SOG_FILES = {"sog/lod-meta.json": b"{}", "sog/0.webp": b"webp"}
RIG_FILES = {"rig.json": b"{}", "motion.json": b"{}", "plants.json": b"{}"}
MATERIALS_EXTRAS = {"uri": "materials.json", "count": 1}


@pytest.fixture(autouse=True)
def _fill_gate_waived(monkeypatch: pytest.MonkeyPatch) -> list[list[Any]]:
    """What is held here is what a republish carries; the fills these tests attach sit on
    single scans with nothing re-scanned, so the API's recompute of the fill gate is
    waived (tests/test_fill_gate.py holds it; `waive_fill_gate`)."""
    return waive_fill_gate(monkeypatch)


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


def a_run(
    buckets: Publisher, *, tag: bytes = b"", collision: bool = True, grid: bool = False
) -> None:
    """The run's packaged tileset in the private bucket, as the packer writes it; `grid`:
    with the confidence grid a measured run packs (`GRID_EXTRAS`)."""
    if not collision:
        buckets.private.delete_object(f"{RUN_DIR}collision.bin")
    if not grid:
        buckets.private.delete_object(f"{RUN_DIR}confidence.bin")
    put_scan(
        buckets.private,
        RUN_DIR,
        tiles={name: b"glb " + tag + name.encode() for name in TILES},
        sidecars={
            **({"collision.bin": b"grid " + tag} if collision else {}),
            **(GRID_FILES if grid else {}),
        },
        extras={
            **({"collision": COLLISION_EXTRAS} if collision else {}),
            **({"confidence": GRID_EXTRAS} if grid else {}),
        },
    )


def fill_for(asset: Asset) -> list[dict[str, Any]]:
    """`FILL_EXTRAS` gated for `asset`'s epoch against `GRID_EXTRAS`: what publish-fill.yml
    would send for it."""
    entry = FILL_EXTRAS[0]
    gate = gate_for(epoch=asset.epoch_id)
    return [{**entry, "evidence": {**entry["evidence"], "rescanGate": gate}}]


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
    """A measured run published and registered, then objects, materials, a fill (gated
    against its grid), the streamed LOD and a plant rig attached beside it."""
    a_run(buckets, grid=True)
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
            "inferredLayers": fill_for(asset),
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
        "confidence",
        "instances",
        "materials",
        "inferredLayers",
        "nativeLod",
    }
    assert extras["inferredLayers"] == fill_for(asset)
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
    run's own collision grid replaces the old one. The inferred fill is in the scan's
    frame, but it was gated by the old tiles' confidence grid, which these tiles do not
    carry: dropped and flagged to fill again (ADR 0016 § Fill), as the grid is."""
    capture, _ = a_scan_with_everything(db, buckets)
    a_run(buckets, tag=b"refined ")
    # alembic's fileConfig (the `engine` fixture) disables the loggers that exist by then.
    monkeypatch.setattr(logging.getLogger("app.worker"), "disabled", False)

    with caplog.at_level(logging.WARNING, logger="app.worker"):
        republish(db, buckets, capture)

    moved = the_splat(db, capture)
    url = moved.source["url"]
    files = files_under(buckets.public, directory_of(url))
    assert set(files) == {*TILES, "collision.bin", "tileset.json"}
    assert files["collision.bin"] == b"grid refined "  # the run's own
    assert files["0/0.glb"] == b"glb refined 0/0.glb"
    extras = tileset_at(buckets.public, url)["root"]["extras"]
    assert set(extras) == {"gaussians", "collision"}
    assert "rigUrl" not in moved.render_config

    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert set(flags) == {
        "instances",
        "materials",
        "nativeLod",
        "rig",
        "confidence",
        "inferredLayers",
    }
    assert flags["inferredLayers"]["action"] == "Inferred fill needs filling again"
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
        "confidence",
        "inferredLayers",
    }


def test_a_republish_from_a_legacy_prefix_drops_its_ungated_fill_and_flags_the_rest(
    db: Session, buckets: Publisher
) -> None:
    """The camp as it is: published before generations, objects, a fill and a collision
    backfill written beside it in place. Its first republish, from a run whose packer wrote
    no grid, says what has to be redone -- the fill too: it was made before the fill gate,
    for no re-scan, so it is not carried (ADR 0016 § Fill)."""
    put_scan(
        buckets.public,
        LEGACY_DIR,
        sidecars={**INSTANCES_FILES, "collision.bin": b"grid", **FILL_FILES},
        extras={
            "instances": INSTANCES_EXTRAS,
            "collision": COLLISION_EXTRAS,
            "inferredLayers": UNGATED_FILL_EXTRAS,
            "mystery": {"note": "nobody knows"},
            # The root tile's own least-confident cell (P4): the tileset's, not a sidecar.
            "confidenceMin": 37,
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
    assert set(files) == {*TILES, "tileset.json"}
    extras = tileset_at(buckets.public, url)["root"]["extras"]
    assert set(extras) == {"gaussians"}
    flags = {flag["kind"]: flag["action"] for flag in moved.sidecar_flags}
    assert flags == {
        "instances": "Objects need re-segmenting",
        "collision": "Collision needs a backfill",
        "inferredLayers": "Inferred fill needs filling again",
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
        sidecars={
            **FILL_FILES,
            **GRID_FILES,
            "fills/3/tileset.json": b"{}",
            "objects/3/tileset.json": b"{}",
        },
        extras={
            "inferredLayers": [*FILL_EXTRAS, split_fill],
            "objects": [{"uri": "x"}],
            "confidence": GRID_EXTRAS,
        },
    )
    # New tiles measured with the very grid the fill was gated by: the fill holds.
    a_run(buckets, tag=b"new ", grid=True)
    plan = plan_carry(
        buckets,
        live_url=f"{PUBLIC_URL}/{live_dir}tileset.json",
        tiles_prefix=RUN_DIR,
        entry=f"{RUN_DIR}tileset.json",
        fill_check=admit_fills,
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


def test_a_far_sites_repack_keeps_what_is_bound_to_its_site_frame_positions(
    db: Session, buckets: Publisher
) -> None:
    """A site 10.4 km across, whose tiles carry translation-only transforms, is bound by
    site-frame checksums (data/tiles/synthetic-far source/expected.json). Packed again in
    other bytes with every position where it was, the binding holds: the worker composes
    each tile's transform before it hashes, as the viewer does."""
    far = TREE_LOD.parent / "synthetic-far"
    expected = json.loads((far / "source" / "expected.json").read_text())
    document, tiles = fixture_scan(far / "splat")
    a_tree_run(buckets, tiles, document)
    capture = Capture(slug="far", name="Far", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    republish(db, buckets, capture)
    bound = {
        "instances.json": json.dumps(
            {
                "tiles": {
                    row["checksum"]: [1, row["gaussians"]] for row in expected["tiles"].values()
                }
            }
        ).encode(),
        "instances.emb": b"\0\1",
    }
    attach(db, buckets, the_splat(db, capture), bound, extras={"instances": INSTANCES_EXTRAS})
    repacked = {
        uri: splat_glb(
            positions.tile_positions(data, None),
            fractional_bits=expected["tiles"][uri]["fractionalBits"],
            fill=9,
        )
        for uri, data in tiles.items()
    }
    assert all(repacked[uri] != tiles[uri] for uri in tiles)
    a_tree_run(buckets, repacked, document)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert files["splat_4-4-2.glb"] == repacked["splat_4-4-2.glb"]
    assert files["instances.json"] == bound["instances.json"]
    assert "instances" in tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert "instances" not in {flag["kind"] for flag in moved.sidecar_flags}


def test_a_tileset_whose_tile_frames_cannot_be_read_carries_no_binding(
    db: Session, buckets: Publisher
) -> None:
    """The same far tiles, but one child transform turns its tile as well as moving it. No
    reader composes rotations (`positions.tile_offsets` refuses it), so no tile's site-frame
    position is known and nothing bound to positions is carried: dropped and flagged, and
    the republish itself still goes through."""
    far = TREE_LOD.parent / "synthetic-far"
    expected = json.loads((far / "source" / "expected.json").read_text())
    document, tiles = fixture_scan(far / "splat")
    a_tree_run(buckets, tiles, document)
    capture = Capture(slug="far-turned", name="Far turned", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    republish(db, buckets, capture)
    bound = {
        "instances.json": json.dumps(
            {
                "tiles": {
                    row["checksum"]: [1, row["gaussians"]] for row in expected["tiles"].values()
                }
            }
        ).encode(),
        "instances.emb": b"\0\1",
    }
    attach(db, buckets, the_splat(db, capture), bound, extras={"instances": INSTANCES_EXTRAS})
    turned = json.loads(json.dumps(document))
    child = next(c for c in turned["root"]["children"] if c.get("transform"))
    child["transform"][0:2] = [0.0, 1.0]
    child["transform"][4:6] = [-1.0, 0.0]
    with pytest.raises(positions.TileFormatError, match="translation"):
        positions.tile_offsets(turned)
    a_tree_run(buckets, tiles, turned)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert "instances.json" not in files
    assert "instances" not in tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert flags["instances"]["action"] == "Objects need re-segmenting"


def test_moved_positions_drop_what_was_bound_to_them_and_flag_it(
    db: Session, buckets: Publisher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One splat of the root tile a quantum higher: that tile's checksum is in no binding,
    so the viewer would refuse it, and the objects, the materials keyed by their ids and the
    rig all go, each flagged. Nothing past that first tile is hashed."""
    capture = a_tree_with_objects(db, buckets)
    document, sh = fixture_scan(TREE_SH)
    nudged = positions.tile_positions(sh["splat.glb"], None).astype("float64")
    nudged[0, 2] += 1 / 4096
    sh["splat.glb"] = splat_glb(nudged)
    assert positions.tile_checksum(sh["splat.glb"], None) not in TREE_CHECKSUMS
    a_tree_run(buckets, sh, document)
    hashed: list[bytes] = []
    real = positions.tile_checksum

    def counting(data: bytes, offset: Any) -> str:
        hashed.append(data)
        return real(data, offset)

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


# --- epochs (decision D3): carry within one epoch only ------------------------------------


def a_second_epoch(db: Session, buckets: Publisher, first: Capture) -> Capture:
    """Another capture of the same site, naming it, registered as its epoch 2."""
    found = db.get(Capture, first.id)
    assert found is not None and found.site_id is not None
    october = Capture(
        slug="camp-october",
        name="Camp in October",
        kind=CaptureKind.GAUSSIAN_SPLAT,
        metadata_={"siteId": str(found.site_id)},
    )
    db.add(october)
    db.commit()
    a_run(buckets, tag=b"october ")
    republish(db, buckets, october)
    return october


def test_a_re_run_carries_from_its_own_epoch_only(db: Session, buckets: Publisher) -> None:
    """Objects attached to epoch 1 were segmented on epoch 1's tiles: epoch 2, captured
    another day, starts with none, and its re-run carries only what epoch 2 had."""
    first, asset = a_scan_with_everything(db, buckets)
    epoch_1_url = asset.source["url"]
    october = a_second_epoch(db, buckets, first)

    fresh = the_splat(db, october)
    assert fresh.id != asset.id and fresh.epoch_id != asset.epoch_id
    assert fresh.sidecar_flags == []
    assert "instances" not in tileset_at(buckets.public, fresh.source["url"])["root"]["extras"]
    a_run(buckets, tag=b"october refined ")
    republish(db, buckets, october)

    rerun = the_splat(db, october)
    assert rerun.id != fresh.id and rerun.epoch_id == fresh.epoch_id
    files = files_under(buckets.public, directory_of(rerun.source["url"]))
    assert not set(INSTANCES_FILES) & set(files)
    assert "instances" not in tileset_at(buckets.public, rerun.source["url"])["root"]["extras"]
    db.expire_all()
    unchanged = db.get(Asset, asset.id)
    assert unchanged is not None and unchanged.source["url"] == epoch_1_url
    assert unchanged.valid_to is None


def test_an_attach_to_another_epoch_does_not_stall_a_re_run(
    db: Session, buckets: Publisher
) -> None:
    """The LiveMoved check is on the epoch's own current asset: objects attached to epoch 1
    while epoch 2 republishes are not a reason to publish epoch 2 again."""
    first, asset = a_scan_with_everything(db, buckets)
    october = a_second_epoch(db, buckets, first)
    live = registration.live_tileset_url(db, october)
    assert live is not None and live != asset.source["url"]
    a_run(buckets, tag=b"october refined ")
    published = publish_outputs(
        buckets.private,
        publish=buckets,
        job_id=RUN,
        registration=a_registration(),
        tiles_stage_id="package",
        carry_from=live,
    )
    attach(db, buckets, asset, {"telemetry.json": b"{}"}, extras={"telemetry": {}})

    register(
        db,
        buckets.private,
        publish=buckets,
        capture=october,
        job_id=RUN,
        registration=a_registration(),
        tiles_stage_id="package",
        published=published,
    )

    assert the_splat(db, october).source["url"] == published.tileset


# --- the confidence grid: the run's own, bound to its tiles ---------------------------------

CONFIDENCE_EXTRAS = GRID_EXTRAS


#: A run's height surface as the packer declares it (P5: `convert(surface=...)`).
SURFACE_EXTRAS = {
    "format": "hexapod.cellgrid",
    "version": 1,
    "kind": "surface",
    "uri": "surface.bin",
    "sha256": "c3" * 32,
}


def a_measured_run(
    buckets: Publisher, *, tag: bytes = b"", confidence: bool = True, surface: bool = False
) -> None:
    """The run's package with its collision grid and, when it measured them, its confidence
    grid and height surface beside the tiles (P4, P5: `splat_tiles.convert(confidence=...,
    surface=...)`)."""
    put_scan(
        buckets.private,
        RUN_DIR,
        tiles={name: b"glb " + tag + name.encode() for name in TILES},
        sidecars={
            "collision.bin": b"grid " + tag,
            **({"confidence.bin": b"cells " + tag} if confidence else {}),
            **({"surface.bin": b"heights " + tag} if surface else {}),
        },
        extras={
            "collision": COLLISION_EXTRAS,
            **({"confidence": CONFIDENCE_EXTRAS} if confidence else {}),
            **({"surface": SURFACE_EXTRAS} if surface else {}),
        },
    )
    for name, shipped in (("confidence.bin", confidence), ("surface.bin", surface)):
        if not shipped:
            buckets.private.delete_object(f"{RUN_DIR}{name}")


def a_measured_scan(db: Session, buckets: Publisher, *, surface: bool = False) -> Capture:
    """The measured run published and registered, its grid recorded as the epoch's map."""
    from tests.confidence_fixtures import descriptor

    a_measured_run(buckets, surface=surface)
    capture = Capture(slug="field", name="Field", kind=CaptureKind.GAUSSIAN_SPLAT)
    db.add(capture)
    db.commit()
    register(
        db,
        buckets.private,
        publish=buckets,
        capture=capture,
        job_id=RUN,
        registration=dataclasses.replace(a_registration(), confidence=descriptor("e1")),
        tiles_stage_id="package",
    )
    return capture


def the_map_url(db: Session, asset: Asset) -> str | None:
    from app.models import EpochMap

    db.expire_all()
    row = db.scalars(select(EpochMap).where(EpochMap.asset_id == asset.id)).first()
    return row.url if row is not None else None


def test_a_new_run_s_own_confidence_grid_replaces_the_old_one(
    db: Session, buckets: Publisher
) -> None:
    """A Refine measured again: its grid is the run's, the old one is not carried."""
    capture = a_measured_scan(db, buckets)
    a_measured_run(buckets, tag=b"refined ")

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert files["confidence.bin"] == b"cells refined "
    extras = tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert extras["confidence"] == CONFIDENCE_EXTRAS
    assert moved.sidecar_flags == []


def test_new_tiles_without_a_grid_drop_the_old_one_and_say_so(
    db: Session, buckets: Publisher
) -> None:
    """A grid of the old reconstruction is not true of new tiles: dropped and flagged,
    never shown over splats it was not measured on."""
    capture = a_measured_scan(db, buckets)
    a_measured_run(buckets, tag=b"new ", confidence=False)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert "confidence.bin" not in files
    extras = tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert "confidence" not in extras
    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert set(flags) == {"confidence"}
    assert flags["confidence"]["action"] == "Confidence needs re-measuring"
    assert "new tiles" in flags["confidence"]["reason"]


def test_the_same_tiles_keep_their_grid_through_an_attach_and_a_republish(
    db: Session, buckets: Publisher
) -> None:
    capture = a_measured_scan(db, buckets)
    attach(
        db,
        buckets,
        the_splat(db, capture),
        INSTANCES_FILES,
        extras={"instances": INSTANCES_EXTRAS},
    )
    attached = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(attached.source["url"]))
    assert files["confidence.bin"] == b"cells "
    # The revision's map moved with its generation.
    assert the_map_url(db, attached) == attached.source["url"].replace(
        "tileset.json", "confidence.bin"
    )
    # The same tiles again, from a run that measured nothing.
    a_measured_run(buckets, confidence=False)

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert files["confidence.bin"] == b"cells " and "instances.json" in files
    extras = tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert extras["confidence"] == CONFIDENCE_EXTRAS
    assert moved.sidecar_flags == []
    # A run that measured nothing republished the same tiles: the grid was carried, and
    # so is the map, beside the new revision's URL.
    assert the_map_url(db, moved) == moved.source["url"].replace("tileset.json", "confidence.bin")


def test_a_height_surface_is_the_run_s_own_and_bound_to_its_tiles(
    db: Session, buckets: Publisher
) -> None:
    """surface.bin as confidence.bin: a Refine's own replaces it, new tiles without one drop
    it and say so, and the same tiles keep it -- heights of one reconstruction are not
    heights of another."""
    capture = a_measured_scan(db, buckets, surface=True)
    live = the_splat(db, capture)
    assert files_under(buckets.public, directory_of(live.source["url"]))["surface.bin"] == (
        b"heights "
    )

    a_measured_run(buckets, tag=b"refined ", surface=True)
    republish(db, buckets, capture)
    refined = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(refined.source["url"]))
    assert files["surface.bin"] == b"heights refined "
    extras = tileset_at(buckets.public, refined.source["url"])["root"]["extras"]
    assert extras["surface"] == SURFACE_EXTRAS
    assert refined.sidecar_flags == []

    # The same tiles from a run that measured nothing: carried.
    a_measured_run(buckets, tag=b"refined ", confidence=False)
    republish(db, buckets, capture)
    same = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(same.source["url"]))
    assert files["surface.bin"] == b"heights refined "
    assert tileset_at(buckets.public, same.source["url"])["root"]["extras"]["surface"] == (
        SURFACE_EXTRAS
    )
    assert same.sidecar_flags == []

    # New tiles with neither grid: both dropped, each flagged with its own action.
    a_measured_run(buckets, tag=b"new ", confidence=False)
    republish(db, buckets, capture)
    moved = the_splat(db, capture)
    files = files_under(buckets.public, directory_of(moved.source["url"]))
    assert "surface.bin" not in files and "confidence.bin" not in files
    extras = tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert "surface" not in extras
    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert set(flags) == {"confidence", "surface"}
    assert flags["surface"]["action"] == "Height surface needs re-measuring"
    assert "new tiles" in flags["surface"]["reason"]


def the_measured_asset(db: Session, buckets: Publisher) -> Asset:
    """The measured scan's asset: a revision of an epoch, its tiles carrying a grid."""
    return the_splat(db, a_measured_scan(db, buckets))


# --- the fill gate (ADR 0016 § Fill): a fill holds only while its grid does -------------

GATED_LIVE_DIR = f"runs/{RUN}/pfedcba9876543210/package/splat/"


def plan_onto(
    buckets: Publisher,
    layers: list[dict[str, Any]],
    *,
    run_grid: dict[str, Any] | None,
    tag: bytes = b"new ",
    fill_check: Any = admit_fills,
) -> Any:
    """A live generation with a grid and `layers`, and a run whose tiles carry `run_grid`
    (or none): what the carry plans."""
    files = {**GRID_FILES}
    for layer in layers:
        folder = layer["uri"].rsplit("/", 1)[0]
        files.update({f"{folder}/tileset.json": b"{}", f"{folder}/f0.glb": b"fill"})
    put_scan(
        buckets.public,
        GATED_LIVE_DIR,
        sidecars=files,
        extras={"confidence": GRID_EXTRAS, "inferredLayers": layers},
    )
    put_scan(
        buckets.private,
        RUN_DIR,
        tiles={name: b"glb " + tag + name.encode() for name in TILES},
        sidecars=GRID_FILES if run_grid is not None else {},
        extras={"confidence": run_grid} if run_grid is not None else {},
    )
    return plan_carry(
        buckets,
        live_url=f"{PUBLIC_URL}/{GATED_LIVE_DIR}tileset.json",
        tiles_prefix=RUN_DIR,
        entry=f"{RUN_DIR}tileset.json",
        fill_check=fill_check,
    )


def _layers(plan: Any) -> list[dict[str, Any]] | None:
    if plan.document is None:
        return None
    found: list[dict[str, Any]] | None = json.loads(plan.document)["root"]["extras"].get(
        "inferredLayers"
    )
    return found


def test_a_gated_fill_is_carried_while_the_new_tiles_carry_its_grid(buckets: Publisher) -> None:
    plan = plan_onto(buckets, FILL_EXTRAS, run_grid=GRID_EXTRAS)
    assert _layers(plan) == FILL_EXTRAS
    assert "inferredLayers" in plan.carried
    assert "inferredLayers" not in {gone.kind for gone in plan.dropped}
    assert set(FILL_FILES) <= {rel for _, rel in plan.objects}


def test_with_no_fill_gate_to_ask_no_fill_is_carried(buckets: Publisher) -> None:
    """A carry that cannot recompute the gate never carries a layer on its metadata alone."""
    plan = plan_onto(buckets, FILL_EXTRAS, run_grid=GRID_EXTRAS, tag=b"", fill_check=None)
    assert not _layers(plan)
    assert not any(rel.startswith("inferred/") for _, rel in plan.objects)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert "no fill gate" in gone.reason


def test_a_fill_the_gate_refuses_is_not_carried(buckets: Publisher) -> None:
    asked: list[Any] = []

    def refuse(entry: Any, transform: Any, read: Any, grid: Any) -> str:
        asked.append((entry, transform, read(entry["uri"]), grid))
        return "3 of its 9 gaussians are outside the cells a re-scan left weak"

    plan = plan_onto(buckets, FILL_EXTRAS, run_grid=GRID_EXTRAS, tag=b"", fill_check=refuse)
    assert not _layers(plan)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert "outside the cells" in gone.reason
    # Asked with the new tileset's frame, the live layer's own bytes and the new grid's
    # whole descriptor (where its cells lie, not only their bytes' sha256).
    (entry, transform, data, grid), *_ = asked
    assert (entry, transform, data, grid) == (FILL_EXTRAS[0], TRANSFORM, b"{}", GRID_EXTRAS)


def test_a_fill_gated_by_another_grid_is_dropped_and_flagged(buckets: Publisher) -> None:
    """A re-run measured its own grid: the cells the fill was gated by may no longer be
    weak, so it is not carried onto the new tiles, and the asset says to fill again."""
    plan = plan_onto(buckets, FILL_EXTRAS, run_grid={**GRID_EXTRAS, "sha256": "b2" * 32})
    assert not _layers(plan)
    assert not any(rel.startswith("inferred/") for _, rel in plan.objects)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert gone.action == "Inferred fill needs filling again"
    assert "confidence grid" in gone.reason
    assert "inferredLayers" not in plan.carried


def test_new_tiles_with_no_grid_drop_every_gated_fill(buckets: Publisher) -> None:
    plan = plan_onto(buckets, FILL_EXTRAS, run_grid=None)
    assert not _layers(plan)
    assert "inferredLayers" in {gone.kind for gone in plan.dropped}


def test_an_ungated_fill_is_dropped_even_onto_the_same_tiles(buckets: Publisher) -> None:
    """A fill from before the gate is not carried by any republish, the very same tiles
    included; a gated one beside it is."""
    gated = {**FILL_EXTRAS[0], "uri": "inferred/gated/tileset.json"}
    plan = plan_onto(buckets, [*UNGATED_FILL_EXTRAS, gated], run_grid=GRID_EXTRAS, tag=b"")
    assert _layers(plan) == [gated]
    assert not any(rel.startswith("inferred/fixer/") for _, rel in plan.objects)
    assert any(rel.startswith("inferred/gated/") for _, rel in plan.objects)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert "no rescanGate" in gone.reason


def test_a_re_run_that_measured_again_flags_its_fill_on_the_asset(
    db: Session, buckets: Publisher
) -> None:
    capture = a_measured_scan(db, buckets)
    asset = the_splat(db, capture)
    attach(db, buckets, asset, FILL_FILES, extras={"inferredLayers": fill_for(asset)})
    a_measured_run(buckets, tag=b"refined ")
    # The refined run measured a grid of its own.
    put_scan(
        buckets.private,
        RUN_DIR,
        tiles={name: b"glb refined " + name.encode() for name in TILES},
        sidecars={"collision.bin": b"grid refined ", "confidence.bin": b"cells refined "},
        extras={
            "collision": COLLISION_EXTRAS,
            "confidence": {**CONFIDENCE_EXTRAS, "sha256": "c3" * 32},
        },
    )

    republish(db, buckets, capture)

    moved = the_splat(db, capture)
    extras = tileset_at(buckets.public, moved.source["url"])["root"]["extras"]
    assert "inferredLayers" not in extras
    flags = {flag["kind"]: flag for flag in moved.sidecar_flags}
    assert flags["inferredLayers"]["action"] == "Inferred fill needs filling again"


def test_the_sidecar_rules_name_the_api_s_own_gate_policy() -> None:
    from app.analysis import rescan_gate
    from app.services import sidecars

    assert sidecars.FILL_GATE_POLICY == rescan_gate.POLICY
