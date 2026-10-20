"""The fill gate, recomputed by the API from what it stores (ADR 0016 § Fill).

An attach that adds or changes an inferred layer (`app/services/attach.py`) is a write-token
holder's claim that the layer fills only cells a re-scan left weak. The layer's own
`evidence.rescanGate` says so, but it is the client's word: anyone with the token can copy
the policy, the grid's sha256 and the epoch id off readable metadata. So the API never takes
it. It recomputes the gate itself and holds the layer's geometry to it:

1. **The targets are the API's own**: `confidence_maps.fill_targets` (the endpoint's rule,
   `app/analysis/rescan_gate.py`) over the `epoch_maps` rows of the asset's epoch and of the
   earlier measured epochs of its site. No earlier measured epoch (nothing was re-scanned),
   no map for this epoch, or no cell still weak is a refusal: there is nothing to fill.
2. **The asset is the epoch's current revision**, so the map the targets came from is the one
   its tiles carry.
3. **The evidence names exactly these targets** (same site, epoch, level, window, cell count
   and mask bytes): a layer filled for targets the API no longer gives -- a re-run of the
   epoch, an earlier epoch measured since -- is filled again, not attached.
4. **The layer is in the scan's frame**: its root transform is the scan's, so its positions
   are in the tileset frame the targets are cells of.
5. **Every gaussian of every tile** -- parents as well as leaves, each tile's origin
   composed (`app/worker/positions.py`, the reader the carry's checksums use) -- is in a
   target cell. Nothing outside them reaches the live site.

Only step 3 reads the client's metadata, and only to refuse: nothing it says admits a layer.
A layer whose tiles cannot be read as splats is refused, not waved through.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import posixpath
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from app.analysis import cell_grid, rescan_gate
from app.models import Asset
from app.schemas.confidence import FillTargetsRead
from app.services import confidence_maps, epochs, sidecars
from app.services.errors import ConflictError, InvalidInputError
from app.storage import ObjectStorage
from app.worker import positions

#: Bytes of one layer's tiles read to check its gaussians: far past any fill (the yard's is
#: kilobytes; a million gaussians are ~30 MB), short of reading a whole scan.
MAX_LAYER_BYTES = 512 * 1024 * 1024
#: What the evidence must name the same as the API's targets (`rescan_gate.verify_layer`).
_SAME = ("policy", "siteId", "epochId", "level", "i0", "j0", "ni", "nj", "cells", "maskSha256")

#: The bytes of a file of the new generation, by its path beside `tileset.json`; None when
#: it is not there.
Reader = Callable[[str], bytes | None]


@dataclass(frozen=True)
class Targets:
    """The API's fill targets for one asset, unpacked."""

    read: FillTargetsRead
    mask: NDArray[np.bool_]
    mask_sha256: str

    def evidence(self) -> dict[str, object]:
        """What a layer filled for these targets says in `rescanGate` (`_SAME`'s keys)."""
        read = self.read
        return {
            "policy": read.policy,
            "siteId": str(read.site_id),
            "epochId": str(read.epoch_id),
            "level": read.level,
            "i0": read.i0,
            "j0": read.j0,
            "ni": read.ni,
            "nj": read.nj,
            "cells": read.cells,
            "maskSha256": self.mask_sha256,
        }

    def outside(self, xyz: NDArray[Any]) -> int:
        """How many of `xyz` (tileset frame; z ignored, cells are columns) are in no target
        cell: outside the window, not finite, or in a cell that is not a target."""
        read = self.read
        cell = math.ldexp(cell_grid.LATTICE_BASE_M, read.level)
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        with np.errstate(invalid="ignore", over="ignore"):
            fi = np.floor(points[:, 0] / cell) - read.i0
            fj = np.floor(points[:, 1] / cell) - read.j0
        inside = np.isfinite(fi) & np.isfinite(fj)
        inside &= (fi >= 0) & (fi < read.ni) & (fj >= 0) & (fj < read.nj)
        hit = np.zeros(points.shape[0], dtype=bool)
        hit[inside] = self.mask[fj[inside].astype(np.int64), fi[inside].astype(np.int64)]
        return int(np.count_nonzero(~hit))


def targets_for(db: Session, storage: ObjectStorage, asset: Asset) -> Targets:
    """The API's fill targets for `asset`'s epoch, from the stored maps; an
    `InvalidInputError` that says why there are none to fill."""
    if asset.epoch_id is None or asset.site_id is None:
        raise InvalidInputError(
            "the asset is a revision of no epoch (seeded or legacy): nothing was measured, so "
            "no cell can be still weak after a re-scan and no fill is attached to it"
        )
    current = epochs.current_asset(db, asset.epoch_id)
    if current is None or current.id != asset.id:
        raise InvalidInputError(
            f"the asset is not the current revision of epoch {asset.epoch_id}: fill targets "
            "are the current revision's"
        )
    try:
        read = confidence_maps.fill_targets(db, storage, asset.site_id, asset.epoch_id)
    except ConflictError as error:
        raise InvalidInputError(f"the API has no fill targets for it: {error}") from error
    if read.cells == 0:
        raise InvalidInputError(
            f"no cell of epoch {read.epoch_id} is still weak after a re-scan: there is "
            "nothing to fill"
        )
    mask = rescan_gate.unpack(read.mask, read.ni, read.nj)
    digest = hashlib.sha256(base64.b64decode(read.mask)).hexdigest()
    return Targets(read=read, mask=mask, mask_sha256=digest)


def check_layer(
    entry: Mapping[str, Any],
    *,
    targets: Targets,
    scan_transform: object,
    read: Reader,
) -> None:
    """Refuses (`InvalidInputError`) unless the layer `entry` declares fills only the API's
    `targets` (module steps 3-5); `read` gives the new generation's files."""
    uri = str(entry.get("uri") or "")
    rel = uri.removeprefix("./")
    evidence = entry.get("evidence")
    gate = evidence.get("rescanGate") if isinstance(evidence, Mapping) else None
    if not isinstance(gate, Mapping):
        raise InvalidInputError(f"inferred layer {uri!r} carries no rescanGate")
    now = targets.evidence()
    changed = [key for key in _SAME if now[key] != gate.get(key)]
    if changed:
        raise InvalidInputError(
            f"inferred layer {uri!r} was filled for other targets than the API's fill targets "
            f"for this epoch now ({', '.join(changed)} differ): fill it again against them"
        )
    document = _json(read(rel), uri)
    root = document.get("root")
    if not isinstance(root, Mapping) or root.get("transform") != scan_transform:
        raise InvalidInputError(
            f"inferred layer {uri!r} is not in the scan's frame: its root transform is not "
            "the scan's tileset's"
        )
    try:
        offsets = positions.tile_offsets(document)
    except positions.TileFormatError as error:
        raise InvalidInputError(f"inferred layer {uri!r}: {error}") from error
    if not offsets:
        raise InvalidInputError(f"inferred layer {uri!r} names no tiles")
    folder = posixpath.dirname(rel)
    total = outside = count = 0
    for content in sorted(offsets):
        try:
            sidecars.check_relative(content)
        except sidecars.SidecarPathError as error:
            raise InvalidInputError(f"inferred layer {uri!r}: tile {error}") from error
        path = posixpath.join(folder, content) if folder else content
        data = read(path)
        if data is None:
            raise InvalidInputError(f"inferred layer {uri!r}: its tile {path} is not there")
        total += len(data)
        if total > MAX_LAYER_BYTES:
            raise InvalidInputError(
                f"inferred layer {uri!r} holds more than {MAX_LAYER_BYTES} bytes of tiles: "
                "too large to check, so not attached"
            )
        try:
            xyz = positions.tile_positions(data, offsets[content])
        except positions.TileFormatError as error:
            raise InvalidInputError(
                f"inferred layer {uri!r}: its tile {path} cannot be read as splats ({error}), "
                "so its gaussians cannot be held to the fill targets"
            ) from error
        count += int(xyz.shape[0])
        outside += targets.outside(xyz)
    if outside:
        raise InvalidInputError(
            f"inferred layer {uri!r}: {outside} of its {count} gaussians are outside the "
            "cells a re-scan left weak (the API's fill targets for this epoch)"
        )
    if count == 0:
        raise InvalidInputError(f"inferred layer {uri!r} holds no gaussians")


def _json(data: bytes | None, uri: str) -> dict[str, Any]:
    if data is None:
        raise InvalidInputError(f"inferred layer {uri!r}: its tileset is not there")
    try:
        document = json.loads(data)
    except ValueError as error:
        raise InvalidInputError(f"inferred layer {uri!r}: its tileset is not JSON") from error
    if not isinstance(document, dict):
        raise InvalidInputError(f"inferred layer {uri!r}: its tileset is not a JSON object")
    return document


def verify(
    db: Session,
    storage: ObjectStorage,
    asset: Asset,
    layers: Sequence[Mapping[str, Any]],
    *,
    scan_transform: object,
    read: Reader,
) -> None:
    """The API's own gate over `layers` (module docstring): the targets recomputed once,
    every layer held to them. Nothing to judge, nothing read."""
    if not layers:
        return
    targets = targets_for(db, storage, asset)
    for entry in layers:
        check_layer(entry, targets=targets, scan_transform=scan_transform, read=read)
