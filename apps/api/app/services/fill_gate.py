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
   composed (`app/worker/positions.py`, the reader the carry's checksums use), and at
   every place a tile is drawn (`tile_placements`: two tiles naming one GLB draw it at
   both), each primitive where the GLB's nodes put it (`positions.tile_draws`: every
   primitive through the node hierarchy, and Cesium's own reading) -- is in a target cell.
   Nothing outside them reaches the live site.

Only step 3 reads the client's metadata, and only to refuse: nothing it says admits a layer.
A layer whose tiles cannot be read as splats is refused, not waved through.

A republish asks the same of every layer it would carry (`carry_check`, used by
`app/worker/carry.py`): the targets of the run's epoch, recomputed, computed on the very grid
the new tiles carry, and every gaussian in them -- or the layer is not carried. It runs
beside a publish with no transaction open, so it never downloads inside one
(`_targets_outside`): the `epoch_maps` rows are read, the transaction ended, the grids
downloaded to disk, and only then the targets computed from those files.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import posixpath
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from app.analysis import cell_grid, rescan_gate
from app.models import Asset, Epoch
from app.schemas.confidence import FillTargetsRead
from app.services import confidence_maps, epochs, sidecars
from app.services.errors import ConflictError, InvalidInputError
from app.storage import ObjectStorage
from app.storage.base import StoredObject
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
    #: The epoch's confidence grid the targets were computed on, as `grid_identity` names
    #: it: its planes' bytes and where they lie.
    grid: str | None = None

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
    `InvalidInputError` that says why there are none to fill. The grids are read from
    `storage` as the targets are computed, inside the caller's transaction (the attach's,
    which holds the asset's row lock throughout); `_targets_outside` is the variant that
    ends it first."""
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
    epoch = db.get(Epoch, asset.epoch_id)
    try:
        read = confidence_maps.fill_targets(db, storage, asset.site_id, asset.epoch_id)
        grid = confidence_maps.measured_map(db, epoch).descriptor if epoch is not None else {}
    except ConflictError as error:
        raise InvalidInputError(f"the API has no fill targets for it: {error}") from error
    if read.cells == 0:
        raise InvalidInputError(
            f"no cell of epoch {read.epoch_id} is still weak after a re-scan: there is "
            "nothing to fill"
        )
    mask = rescan_gate.unpack(read.mask, read.ni, read.nj)
    digest = hashlib.sha256(base64.b64decode(read.mask)).hexdigest()
    return Targets(read=read, mask=mask, mask_sha256=digest, grid=grid_identity(grid))


def grid_identity(descriptor: object) -> str | None:
    """Which grid `descriptor` is, as a carry must match it: the sha256 of the whole
    descriptor (`confidence_maps.descriptor_sha256`, the grid cache's key), so the planes'
    sha256 *and* every level's window origin (`i0`, `j0`), size and level, the lattice, the
    frame and the channels. The same bytes at other origins are other cells. None for a
    descriptor that is not one (no planes' sha256)."""
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("sha256"), str):
        return None
    try:
        return confidence_maps.descriptor_sha256(descriptor)
    except (TypeError, ValueError):
        return None


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
        placements = positions.tile_placements(document)
    except positions.TileFormatError as error:
        raise InvalidInputError(f"inferred layer {uri!r}: {error}") from error
    if not placements:
        raise InvalidInputError(f"inferred layer {uri!r} names no tiles")
    folder = posixpath.dirname(rel)
    total = outside = count = 0
    files: dict[str, bytes] = {}
    # Every place a tile is drawn, not every uri: two tiles naming one GLB draw it twice.
    for content, offset in placements:
        try:
            sidecars.check_relative(content)
        except sidecars.SidecarPathError as error:
            raise InvalidInputError(f"inferred layer {uri!r}: tile {error}") from error
        path = posixpath.join(folder, content) if folder else content
        data = files.get(path)
        if data is None:
            data = read(path)
            if data is None:
                raise InvalidInputError(f"inferred layer {uri!r}: its tile {path} is not there")
            files[path] = data
        # Counted per placement: the work, not the download, is what this bounds.
        total += len(data)
        if total > MAX_LAYER_BYTES:
            raise InvalidInputError(
                f"inferred layer {uri!r} holds more than {MAX_LAYER_BYTES} bytes of tiles: "
                "too large to check, so not attached"
            )
        try:
            drawn = positions.drawn_positions(data, offset)
        except positions.TileFormatError as error:
            raise InvalidInputError(
                f"inferred layer {uri!r}: its tile {path} cannot be read as splats ({error}), "
                "so its gaussians cannot be held to the fill targets"
            ) from error
        # Each further draw of the tile is another pass over its gaussians.
        total += len(data) * (len(drawn) - 1)
        if total > MAX_LAYER_BYTES:
            raise InvalidInputError(
                f"inferred layer {uri!r} draws more than {MAX_LAYER_BYTES} bytes of tiles: "
                "too large to check, so not attached"
            )
        for xyz in drawn:
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


#: What a republish asks of one inferred layer it would carry (`app/worker/carry.py`
#: `FillCheck`): the entry, the new tileset's root transform, a reader of the live
#: generation's files and the descriptor of the grid the new tiles carry; None to carry it,
#: else why not.
CarryCheck = Callable[[Mapping[str, Any], object, Reader, Mapping[str, Any] | None], str | None]


def carry_check(
    db: Session,
    storage: ObjectStorage,
    capture_id: uuid.UUID | None,
    *,
    after: Callable[[], None] | None = None,
) -> CarryCheck:
    """The gate a republish of `capture_id`'s epoch holds each carried layer to: the API's
    targets for the epoch, recomputed once (and only when a layer is asked about), from the
    maps in `storage` -- `after` then runs, so a caller that must not sit in a transaction
    can end the one the reads opened. A layer is carried only when those targets were
    computed on the very grid the new tiles carry -- the same planes at the same window
    origins, level and frame (`grid_identity`); a re-run that measured its own grid has
    other targets, not yet recorded -- and it passes `check_layer` on them."""
    found: list[Targets | str] = []

    def targets() -> Targets | str:
        if found:
            return found[0]
        try:
            found.append(_targets_outside(db, storage, capture_id, after or (lambda: None)))
        except InvalidInputError as error:
            found.append(str(error))
        finally:
            if after is not None:
                after()
        return found[0]

    def check(
        entry: Mapping[str, Any],
        scan_transform: object,
        read: Reader,
        grid: Mapping[str, Any] | None,
    ) -> str | None:
        got = targets()
        if isinstance(got, str):
            return got
        carried = grid_identity(grid)
        if carried is None or got.grid != carried:
            return (
                "the epoch's fill targets were computed on a confidence grid the new tiles do "
                "not carry: the cells it filled may no longer be weak"
            )
        try:
            check_layer(entry, targets=got, scan_transform=scan_transform, read=read)
        except InvalidInputError as error:
            return str(error)
        return None

    return check


class _Fetched:
    """The grids `_targets_outside` downloaded, served from disk to `load_grid` as if from
    `storage`: what it reads of a store (`public_url`, `head_object`, `get_object_tail`). A
    key that was not downloaded is missing -- `load_grid`'s refusal -- so a grid the
    targets need and the prefetch did not foresee (an epoch published in between) is
    never downloaded inside the transaction: the gate refuses instead."""

    def __init__(self, storage: ObjectStorage, files: Mapping[str, Path]) -> None:
        self._storage, self._files = storage, dict(files)

    def public_url(self, key: str) -> str:
        return self._storage.public_url(key)

    def head_object(self, key: str) -> StoredObject | None:
        path = self._files.get(key)
        if path is None:
            return None
        return StoredObject(
            key=key,
            url=self.public_url(key),
            content_type="application/octet-stream",
            size=path.stat().st_size,
        )

    def get_object_tail(self, key: str, max_bytes: int) -> tuple[bytes, int]:
        data = self._files[key].read_bytes()
        return data[-max_bytes:] if max_bytes < len(data) else data, len(data)


def _map_rows(db: Session, epoch: Epoch) -> list[tuple[str, str]]:
    """`(url, descriptor sha256)` of every map `confidence_maps.fill_targets` reads for
    `epoch`: its current revision's, and those of the `MAX_EARLIER` newest earlier measured
    epochs of its site (the route decodes no more; its one more only says `capped`) --
    found by the route's own bounded query (`confidence_maps.earlier_measured`), one
    statement however many unmeasured epochs lie between. Keyed as the grid cache is (`descriptor_sha256`),
    and read as plain values, so nothing of the session is needed once the transaction
    ends."""
    try:
        own = confidence_maps.measured_map(db, epoch)
    except ConflictError:
        return []
    earlier = confidence_maps.earlier_measured(db, epoch, confidence_maps.MAX_EARLIER)
    return [
        (row.url, confidence_maps.descriptor_sha256(row.descriptor))
        for row in (own, *(found for _, found in earlier))
    ]


def _targets_outside(
    db: Session,
    storage: ObjectStorage,
    capture_id: uuid.UUID | None,
    end: Callable[[], None],
) -> Targets:
    """`targets_for` the current revision of `capture_id`'s epoch with no grid downloaded
    inside a transaction: the rows are read and `end` called; the grids downloaded to a
    temporary folder (each refused past `MAX_GRID_BYTES` before it is fetched); then the
    targets computed by `confidence_maps.fill_targets` itself -- its rule, its selection --
    reading those files (`_Fetched`), and `end` again by the caller."""
    epoch = epochs.epoch_of_capture(db, capture_id) if capture_id is not None else None
    current = epochs.current_asset(db, epoch.id) if epoch is not None else None
    if epoch is None or current is None:
        raise InvalidInputError(
            "the run's capture is no epoch with a current revision: nothing it shows was "
            "measured, so no fill holds on it"
        )
    current_id = current.id
    rows = _map_rows(db, epoch)
    end()
    with tempfile.TemporaryDirectory(prefix="fill-gate-") as folder:
        files: dict[str, Path] = {}
        for index, (url, sha) in enumerate(rows):
            key = confidence_maps.storage_key(storage, url)
            if key is None or confidence_maps.GRIDS.get((url, sha)) is not None:
                continue
            head = storage.head_object(key)
            if head is None or head.size > confidence_maps.MAX_GRID_BYTES:
                continue
            path = Path(folder) / f"{index}.bin"
            storage.download_file(key, path)
            files[key] = path
        asset = db.get(Asset, current_id)
        if asset is None:
            raise InvalidInputError("the epoch's current revision is gone")
        return targets_for(db, cast(ObjectStorage, _Fetched(storage, files)), asset)
