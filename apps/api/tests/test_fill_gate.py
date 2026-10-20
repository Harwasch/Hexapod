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
