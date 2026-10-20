"""The API's own fill gate (ADR 0016 § Fill; `app/services/fill_gate.py`).

Codex R3 on #60: the attach checked only the layer's own `rescanGate` metadata, which a
write-token holder can copy off readable metadata -- even for an epoch nothing re-scanned.
The API now recomputes the targets from the `epoch_maps` rows it stores (the fill-targets
rule) and holds every gaussian of every tile of the layer to them.

The site is the yard's two fixture epochs registered through `register` (e1, then its
re-scan e2, tests/test_fill_targets_api.py), each current revision's tiles published beside
its grid. The layers are real splat GLBs (`splat_glb`) whose gaussians sit at chosen cells.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.analysis import rescan_gate
from app.models import Asset, Epoch
from app.schemas.confidence import FillTargetsRead
from app.schemas.sidecar import SidecarAttach
from app.services import attach as attach_service
from app.services import confidence_maps, epochs
from app.services.errors import InvalidInputError
from app.storage import S3Storage
from tests.confidence_fixtures import descriptor
from tests.sidecar_fixtures import TRANSFORM, a_tileset, put_scan, splat_glb, stage
from tests.test_fill_targets_api import two_epochs

BASE = "https://cdn.example.com/twin-worker-test/"
LAYER = "inferred/gated/"


@pytest.fixture(autouse=True)
def _fresh_grids() -> None:
    confidence_maps.GRIDS.clear()


def published(db: Session, storage: S3Storage, epoch: Epoch, name: str) -> Asset:
    """The epoch's current revision, its tiles put beside the grid `register` recorded."""
    asset = epochs.current_asset(db, epoch.id)
    assert asset is not None
    directory = str(asset.source["url"]).removeprefix(BASE).rsplit("/", 1)[0] + "/"
    put_scan(storage, directory, extras={"confidence": descriptor(name)})
    return asset


def site(db: Session, storage: S3Storage, tmp_path: Path) -> tuple[Asset, Asset]:
    e1, e2 = two_epochs(db, storage, tmp_path)
    return published(db, storage, e1, "e1"), published(db, storage, e2, "e2")


def targets(db: Session, storage: S3Storage, asset: Asset) -> FillTargetsRead:
    assert asset.site_id is not None
    return confidence_maps.fill_targets(db, storage, asset.site_id, asset.epoch_id)


def centre(read: FillTargetsRead, *, target: bool) -> list[float]:
    """The middle of the first cell of the window that is (or is not) a target."""
    mask = rescan_gate.unpack(read.mask, read.ni, read.nj)
    dj, di = np.argwhere(mask if target else ~mask)[0]
    return [(read.i0 + di + 0.5) * read.cell_m, (read.j0 + dj + 0.5) * read.cell_m, 0.5]


def gate_of(read: FillTargetsRead, **changes: Any) -> dict[str, Any]:
    """The `rescanGate` a fill made for `read` records (`RescanGate.evidence`)."""
    gate = {
        "policy": read.policy,
        "siteId": str(read.site_id),
        "epochId": str(read.epoch_id),
        "weakIn": [str(e) for e in read.weak_in],
        "comparedWith": [str(e) for e in read.compared_with],
        "level": read.level,
        "cellM": read.cell_m,
        "i0": read.i0,
        "j0": read.j0,
        "ni": read.ni,
        "nj": read.nj,
        "cells": read.cells,
        "maskSha256": hashlib.sha256(base64.b64decode(read.mask)).hexdigest(),
        "confidenceSha256": descriptor("e2")["sha256"],
    }
    return {**gate, **changes}


def layer_files(
    parent: list[list[float]], leaf: list[list[float]], transform: list[float] = TRANSFORM
) -> dict[str, bytes]:
    """A two-level inferred layer: `parent` gaussians in its root tile, `leaf` in its child."""
    document = a_tileset(("f0.glb", "f1.glb"))
    document["root"]["transform"] = transform
    return {
        f"{LAYER}tileset.json": json.dumps(document).encode(),
        f"{LAYER}f0.glb": splat_glb(parent),
        f"{LAYER}f1.glb": splat_glb(leaf),
    }


def attach(
    db: Session, storage: S3Storage, asset: Asset, files: dict[str, bytes], gate: dict[str, Any]
) -> str:
    entry = {"uri": f"{LAYER}tileset.json", "evidence": {"kind": "inferred", "rescanGate": gate}}
    result = attach_service.attach_sidecars(
        db,
        asset.id,
        SidecarAttach(
            staging_prefix=stage(storage, asset.id, files, uuid.uuid4().hex),
            based_on=asset.source["url"],
            extras={"inferredLayers": [entry]},
        ),
        staging=storage,
        public=storage,
    )
    return result.url


def test_a_fill_inside_the_apis_own_targets_is_attached(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)

    url = attach(db, storage, asset, layer_files([inside], [inside]), gate_of(read))

    document = json.loads(storage.get_object(url.removeprefix(BASE)))
    assert [e["uri"] for e in document["root"]["extras"]["inferredLayers"]] == [
        f"{LAYER}tileset.json"
    ]


def test_a_forged_gate_on_an_epoch_nothing_re_scanned_is_refused(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Codex R3 #60: policy, grid sha256 and epoch id are readable metadata. On the site's
    first measured epoch they passed the attach, though nothing was ever re-scanned."""
    first, second = site(db, storage, tmp_path)
    inside = centre(targets(db, storage, second), target=True)
    forged = {
        "policy": "rescan-gate-v1",
        "confidenceSha256": descriptor("e1")["sha256"],
        "epochId": str(first.epoch_id),
    }
    with pytest.raises(InvalidInputError, match="re-scanned"):
        attach(db, storage, first, layer_files([inside], [inside]), forged)


def test_a_gate_copied_from_the_targets_admits_no_gaussian_outside_them(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The evidence can be made to match the API's answer exactly; the gaussians cannot
    lie about where they are."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    with pytest.raises(InvalidInputError, match="1 of its 2 gaussians are outside"):
        attach(db, storage, asset, layer_files([inside], [outside]), gate_of(read))
    far = [inside[0] + 1000.0, inside[1], inside[2]]
    with pytest.raises(InvalidInputError, match="outside the cells"):
        attach(db, storage, asset, layer_files([inside], [far]), gate_of(read))


def test_every_tile_is_checked_not_only_the_leaves(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A parent tile is drawn at coarse detail: its gaussians are held to the targets too."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    with pytest.raises(InvalidInputError, match="1 of its 2 gaussians are outside"):
        attach(db, storage, asset, layer_files([outside], [inside]), gate_of(read))


def test_a_tile_named_twice_is_checked_at_every_place_it_is_drawn(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Codex B5b N1 (probed): two sibling tiles name the same GLB, the second moved into a
    cell that is not a target. The viewer draws it at both; the gate kept one offset per
    uri and saw only the allowed copy."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    files = layer_files([inside], [inside])
    document = json.loads(files[f"{LAYER}tileset.json"])
    child = document["root"]["children"][0]
    shift = [
        1.0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        outside[0] - inside[0],
        outside[1] - inside[1],
        0,
        1,
    ]
    document["root"]["children"].append({**child, "transform": shift})
    files[f"{LAYER}tileset.json"] = json.dumps(document).encode()
    with pytest.raises(InvalidInputError, match="1 of its 3 gaussians are outside"):
        attach(db, storage, asset, files, gate_of(read))


def test_a_fill_for_other_targets_is_filled_again(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)
    stale = gate_of(read, maskSha256="00" * 32, cells=read.cells + 1)
    with pytest.raises(InvalidInputError, match="cells, maskSha256 differ"):
        attach(db, storage, asset, layer_files([inside], [inside]), stale)


def test_a_layer_in_another_frame_is_refused(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)
    moved = [*TRANSFORM[:12], 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(InvalidInputError, match="not in the scan's frame"):
        attach(db, storage, asset, layer_files([inside], [inside], moved), gate_of(read))


def test_a_layer_whose_tiles_are_not_splats_is_refused(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)
    files = {**layer_files([inside], [inside]), f"{LAYER}f1.glb": b"fill glb"}
    with pytest.raises(InvalidInputError, match="cannot be read as splats"):
        attach(db, storage, asset, files, gate_of(read))


def test_a_fill_variants_layers_are_held_to_the_same_targets(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A bake-off's fill variant is an inferred layer the viewer can show too."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)

    def variant(files: dict[str, bytes]) -> str:
        moved = {k.replace(LAYER, "variants/fill/hedge/"): v for k, v in files.items()}
        layer = {
            "uri": "variants/fill/hedge/tileset.json",
            "evidence": {"kind": "inferred", "rescanGate": gate_of(read)},
        }
        entry = {"name": "hedge", "label": "Hedge", "inferredLayers": [layer]}
        return attach_service.attach_sidecars(
            db,
            asset.id,
            SidecarAttach(
                staging_prefix=stage(storage, asset.id, moved, uuid.uuid4().hex),
                based_on=asset.source["url"],
                extras={"variants": {"fill": [entry]}},
            ),
            staging=storage,
            public=storage,
        ).url

    with pytest.raises(InvalidInputError, match="outside the cells"):
        variant(layer_files([inside], [outside]))
    db.refresh(asset)
    assert variant(layer_files([inside], [inside]))


def _attach_files(
    db: Session, storage: S3Storage, asset: Asset, files: dict[str, bytes], extras: dict[str, Any]
) -> str:
    db.refresh(asset)
    return attach_service.attach_sidecars(
        db,
        asset.id,
        SidecarAttach(
            staging_prefix=stage(storage, asset.id, files, uuid.uuid4().hex),
            based_on=asset.source["url"],
            extras=extras,
        ),
        staging=storage,
        public=storage,
    ).url


def test_replacing_a_layers_files_with_no_extras_is_still_gated(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Codex R3 #60: an attach that stages new files for a declared layer and sends an
    empty `extras` replaced the layer's tiles while its declaration -- and the evidence in
    it -- stayed, so nothing judged the new geometry."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    attach(db, storage, asset, layer_files([inside], [inside]), gate_of(read))

    # A staged file replaces the layer's whole folder (`inferred/<name>/`, its unit).
    with pytest.raises(InvalidInputError, match="outside the cells"):
        _attach_files(db, storage, asset, layer_files([inside], [outside]), {})
    with pytest.raises(InvalidInputError, match="outside the cells"):
        _attach_files(db, storage, asset, layer_files([outside], [inside]), {})
    # Files for the layer that stay inside the targets are a fine re-attach.
    assert _attach_files(db, storage, asset, layer_files([inside], [inside, inside]), {})


def test_an_unchanged_variant_declaration_does_not_carry_new_files_past_the_gate(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The variant shortcut skipped an entry the asset already had; its files are judged
    whenever the request stages any of them."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    folder = "variants/fill/hedge/"
    layer = {
        "uri": f"{folder}tileset.json",
        "evidence": {"kind": "inferred", "rescanGate": gate_of(read)},
    }
    extras = {
        "variants": {"fill": [{"name": "hedge", "label": "Hedge", "inferredLayers": [layer]}]}
    }
    files = {k.replace(LAYER, folder): v for k, v in layer_files([inside], [inside]).items()}
    _attach_files(db, storage, asset, files, extras)

    with pytest.raises(InvalidInputError, match="outside the cells"):
        _attach_files(db, storage, asset, {f"{folder}f1.glb": splat_glb([outside])}, extras)
    with pytest.raises(InvalidInputError, match="outside the cells"):
        _attach_files(db, storage, asset, {f"{folder}f1.glb": splat_glb([outside])}, {})


# --- a republish carries a layer only if the same gate admits it (Codex R3 #60, 7) --------

RUNS = "twin-f4-runs"


def _run_bucket() -> S3Storage:
    """A private bucket beside `storage` (inside the same moto): the run's own outputs."""
    import boto3

    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=RUNS)
    return S3Storage(
        bucket=RUNS,
        endpoint_url=None,
        access_key="key",
        secret_key="secret",
        region="us-east-1",
        public_base_url=None,
    )


def _live(
    storage: S3Storage, asset: Asset, files: dict[str, bytes], extras: dict[str, Any]
) -> None:
    """The live generation of `asset` as it stands, with `files` and `extras` beside its
    tiles -- written straight to the bucket, as a layer from before the gate (or one that
    reached it some other way) would be."""
    directory = str(asset.source["url"]).removeprefix(BASE).rsplit("/", 1)[0] + "/"
    put_scan(storage, directory, sidecars=files, extras={"confidence": descriptor("e2"), **extras})


def _republish(
    db: Session, storage: S3Storage, asset: Asset, after: Callable[[], None] | None = None
) -> Any:
    """What a re-run of `asset`'s epoch carries: the same tiles and the same grid, so every
    sidecar would be carried were it not for the gate. `after` is the check's
    end-of-transaction hook, as the runner gives it."""
    from app.services import fill_gate
    from app.worker.carry import plan_carry
    from app.worker.publish import Publisher
    from tests.confidence_fixtures import blob

    private = _run_bucket()
    run = f"runs/{uuid.uuid4()}/package/splat/"
    put_scan(
        private,
        run,
        sidecars={"confidence.bin": blob("e2")},
        extras={"confidence": descriptor("e2")},
    )
    epoch = db.get(Epoch, asset.epoch_id)
    assert epoch is not None
    return plan_carry(
        Publisher(private=private, public=storage),
        live_url=str(asset.source["url"]),
        tiles_prefix=run,
        entry=f"{run}tileset.json",
        fill_check=fill_gate.carry_check(db, storage, epoch.capture_id, after=after),
    )


def _extras(plan: Any) -> dict[str, Any]:
    """The new tileset's root extras; the run's own (no layers) when nothing was merged."""
    if plan.document is None:
        return {}
    extras: dict[str, Any] = json.loads(plan.document)["root"]["extras"]
    return extras


def _entry(read: FillTargetsRead, folder: str = LAYER) -> dict[str, Any]:
    return {
        "uri": f"{folder}tileset.json",
        "evidence": {"kind": "inferred", "rescanGate": gate_of(read)},
    }


def test_a_republish_carries_a_layer_the_gate_admits(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)
    _live(storage, asset, layer_files([inside], [inside]), {"inferredLayers": [_entry(read)]})

    plan = _republish(db, storage, asset)

    assert _extras(plan)["inferredLayers"] == [_entry(read)]
    assert {f"{LAYER}f0.glb", f"{LAYER}f1.glb"} <= {rel for _, rel in plan.objects}
    assert not [gone for gone in plan.dropped if gone.kind == "inferredLayers"]


def test_a_republish_never_carries_a_layer_the_gate_refuses(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Metadata that names the API's targets exactly, on a layer whose gaussians are not in
    them: the republish recomputes the gate, drops the layer and its files, and says why."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    _live(storage, asset, layer_files([outside], [inside]), {"inferredLayers": [_entry(read)]})

    plan = _republish(db, storage, asset)

    assert "inferredLayers" not in _extras(plan)
    assert not any(rel.startswith(LAYER) for _, rel in plan.objects)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert "outside the cells" in gone.reason
    assert gone.action == "Inferred fill needs filling again"
    assert "inferredLayers" not in plan.carried


def test_a_republish_holds_fill_variants_to_the_gate(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Codex R3 #60 finding 7: the republish judged top-level `inferredLayers` only, and
    carried `variants` -- a fill variant's layers included -- as it found them."""
    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside, outside = centre(read, target=True), centre(read, target=False)
    good, bad = "variants/fill/good/", "variants/fill/bad/"
    files = {
        **{k.replace(LAYER, good): v for k, v in layer_files([inside], [inside]).items()},
        **{k.replace(LAYER, bad): v for k, v in layer_files([inside], [outside]).items()},
    }
    fill = [
        {"name": "good", "label": "Good", "inferredLayers": [_entry(read, good)]},
        {"name": "bad", "label": "Bad", "inferredLayers": [_entry(read, bad)]},
    ]
    _live(storage, asset, files, {"variants": {"fill": fill}})

    plan = _republish(db, storage, asset)

    assert _extras(plan)["variants"] == {"fill": [fill[0]]}
    carried = {rel for _, rel in plan.objects}
    assert f"{good}f1.glb" in carried and not any(rel.startswith(bad) for rel in carried)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "variants"]
    assert "bad" in gone.reason and "outside the cells" in gone.reason


def test_a_republish_with_no_earlier_epoch_carries_no_fill(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """On the site's first epoch nothing was re-scanned: no fill holds there, whatever its
    evidence says."""
    first, second = site(db, storage, tmp_path)
    read = targets(db, storage, second)
    inside = centre(read, target=True)
    directory = str(first.source["url"]).removeprefix(BASE).rsplit("/", 1)[0] + "/"
    forged = {**_entry(read), "evidence": {"kind": "inferred", "rescanGate": gate_of(read)}}
    forged["evidence"]["rescanGate"]["confidenceSha256"] = descriptor("e1")["sha256"]
    put_scan(
        storage,
        directory,
        sidecars=layer_files([inside], [inside]),
        extras={"confidence": descriptor("e1"), "inferredLayers": [forged]},
    )
    from app.services import fill_gate
    from app.worker.carry import plan_carry
    from app.worker.publish import Publisher
    from tests.confidence_fixtures import blob

    private = _run_bucket()
    run = f"runs/{uuid.uuid4()}/package/splat/"
    put_scan(
        private,
        run,
        sidecars={"confidence.bin": blob("e1")},
        extras={"confidence": descriptor("e1")},
    )
    epoch = db.get(Epoch, first.epoch_id)
    assert epoch is not None
    plan = plan_carry(
        Publisher(private=private, public=storage),
        live_url=str(first.source["url"]),
        tiles_prefix=run,
        entry=f"{run}tileset.json",
        fill_check=fill_gate.carry_check(db, storage, epoch.capture_id),
    )
    assert "inferredLayers" not in _extras(plan)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert "re-scanned" in gone.reason


def test_no_session_is_idle_in_a_transaction_while_the_carry_gate_downloads_grids(
    db: Session, storage: S3Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carry gate reads the site's `epoch_maps` rows, then downloads their grids -- up
    to nine of 32 MB, with the publish's copy still to come. A session idle in a transaction
    that long is killed by the database (`runner._end_transaction`), so during every grid
    download no session may be: checked in Postgres itself, as
    test_worker_registration.py's reference-download test does, with the runner's hook."""
    from sqlalchemy import create_engine, text

    from tests.conftest import TEST_DATABASE_URL

    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)
    _live(storage, asset, layer_files([inside], [inside]), {"inferredLayers": [_entry(read)]})
    db.commit()
    confidence_maps.GRIDS.clear()

    probe = create_engine(TEST_DATABASE_URL)
    seen: list[tuple[str, int]] = []

    def idle_in_transaction() -> int:
        with probe.connect() as connection:
            return int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = "
                        "current_database() AND pid <> pg_backend_pid() AND "
                        "state LIKE 'idle in transaction%'"
                    )
                ).scalar_one()
            )

    def spied(name: str) -> Callable[..., Any]:
        real = getattr(storage, name)

        def call(key: str, *args: Any, **kwargs: Any) -> Any:
            if key.endswith("confidence.bin"):
                seen.append((f"{name} {key}", idle_in_transaction()))
            return real(key, *args, **kwargs)

        return call

    for name in ("get_object", "get_object_tail", "download_file", "head_object"):
        monkeypatch.setattr(storage, name, spied(name))
    plan = _republish(db, storage, asset, after=db.commit)
    probe.dispose()

    assert _extras(plan)["inferredLayers"] == [_entry(read)]
    downloads = [key for key, _ in seen if not key.startswith("head_object")]
    # e2's grid and e1's: each fetched once, and never inside a transaction.
    assert len({key.split(" ", 1)[1] for key in downloads}) == 2, seen
    assert all(idle == 0 for _key, idle in seen), seen


def test_a_grid_the_prefetch_did_not_foresee_is_refused_not_fetched_in_a_transaction(
    db: Session, storage: S3Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should the targets need a map the rows read first did not name (an epoch published
    in between), the gate refuses rather than download it inside the transaction."""
    from app.services import fill_gate

    _, asset = site(db, storage, tmp_path)
    read = targets(db, storage, asset)
    inside = centre(read, target=True)
    _live(storage, asset, layer_files([inside], [inside]), {"inferredLayers": [_entry(read)]})
    db.commit()
    confidence_maps.GRIDS.clear()
    real = fill_gate._map_rows
    monkeypatch.setattr(fill_gate, "_map_rows", lambda db, epoch: real(db, epoch)[:1])
    fetched: list[str] = []
    real_tail = storage.get_object_tail

    def tail(key: str, max_bytes: int) -> tuple[bytes, int]:
        fetched.append(key)
        return real_tail(key, max_bytes)

    monkeypatch.setattr(storage, "get_object_tail", tail)

    plan = _republish(db, storage, asset, after=db.commit)

    assert "inferredLayers" not in _extras(plan)
    (gone,) = [gone for gone in plan.dropped if gone.kind == "inferredLayers"]
    assert "is missing" in gone.reason
    assert not [key for key in fetched if key.endswith("confidence.bin")]
