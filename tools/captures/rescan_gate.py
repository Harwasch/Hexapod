"""The fill gate: an inferred fill is made only where a re-scan left cells weak.

ADR 0016 § Fill. Generation is never asked to cover what nobody has tried to capture again.
Which cells those are is the API's to say: `GET /api/v1/sites/{id}/fill-targets?epochId=B`
(policy `rescan-gate-v1`, apps/api/app/analysis/rescan_gate.py) answers with epoch B's own
**finest** cells that are weak in B, where the newest earlier measured epoch whose footprint
covers the cell was weak there too (one that never reached it says nothing; one that saw it
well fixed it). A target is never a coarsened parent of a cell B saw well. This module reads
that answer (`RescanGate.from_json`) exactly as the API's own `unpack` does:

* the window is `ni x nj` cells of the site lattice at `level` (`cell_grid`: cells of
  `0.25 m * 2**level`, `i = floor(x / cell)` east and `j = floor(y / cell)` north of the
  tileset frame's origin), starting at cell `(i0, j0)`;
* `mask` is base64 of `nj` rows, **south to north** (row 0 is `j0`), each `rowBytes`
  bytes; in a row, bit `k` (**least significant first**) of byte `b` is cell
  `i0 + 8 b + k`. 1 is a target. The bits past `ni` in a row's last byte are 0;
* `satisfiedBy` is base64 of one byte per target, in the mask's order (rows south to north,
  `i` ascending): the index into `comparedWith` of the earlier epoch whose weak cell it
  re-scanned. `weakIn` is exactly those epochs, in `comparedWith`'s order.

`RescanGate.mask(xyz)` says, per point in the **tileset frame** -- the frame the root
transform places, which is the site frame whenever the run had one; the frame
`splat_render.load_tileset` returns, every tile's own origin composed -- whether the point
is in a target cell. A point outside the window is in none.

`RescanGateError` is every refusal: a payload that is not `rescan-gate-v1` or does not add
up (`from_payload`); a tileset whose own confidence grid disagrees with the targets
(`check_tileset`: they are not at its finest level, or were computed for another epoch, or
another revision of it, so this tileset is not the one they gate); and, when a layer is published, evidence without the
gate, for another epoch than the asset's, for targets the API no longer gives, or a layer
holding a gaussian outside them (`verify_layer`, over `layer_positions`: every tile's
gaussians, parents as well as leaves).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cell_grid

__all__ = [
    "INSET_M",
    "MAX_CELLS",
    "POLICY",
    "RescanGate",
    "RescanGateError",
    "layer_positions",
    "verify_layer",
]

POLICY = "rescan-gate-v1"
#: The API compares at a level no finer than B's finest, which the pipeline caps at 2^22
#: cells; four times that is not a window the API wrote.
MAX_CELLS = 1 << 24
#: A lifted gaussian is kept only when its centre is at least this far inside target cells
#: in x and y (`RescanGate.inside`), so the packed layer still has it inside once stored.
#: First principles, not a measurement: the layer's PLY holds float32 (half a spacing is at
#: most 2^-12 m below 8,192 m from the frame origin), SPZ rounds to half a quantum (2^-13 m
#: at 12 fractional bits) and decodes tile-local float32 (2^-13 m under 2,048 m): 2^-11 m
#: in all, half of this. A layer that still lands outside is refused at publish, not kept.
INSET_M = math.ldexp(1.0, -10)

_KEYS = (
    "policy",
    "siteId",
    "epochId",
    "weakIn",
    "comparedWith",
    "frame",
    "cellM",
    "level",
    "i0",
    "j0",
    "ni",
    "nj",
    "rowBytes",
    "bitOrder",
    "mask",
    "cells",
    "satisfiedBy",
    "reason",
)


class RescanGateError(ValueError):
    """A fill-targets answer, a tileset or a layer the gate refuses; the message says why."""


def _int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RescanGateError(f"fill targets: {key} must be an integer, not {value!r}")
    return value


def _uuid(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise RescanGateError(f"fill targets: {what} must be a UUID string, not {value!r}")
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise RescanGateError(f"fill targets: {what} {value!r} is not a UUID") from error


def _uuids(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RescanGateError(f"fill targets: {what} must be a list of UUIDs, not {value!r}")
    out = tuple(_uuid(v, what) for v in value)
    if len(set(out)) != len(out):
        raise RescanGateError(f"fill targets: {what} names an epoch twice")
    return out


def _satisfied(encoded: object, cells: int, compared: int) -> tuple[int, ...]:
    """`satisfiedBy` read: one index into `comparedWith` per target."""
    if not isinstance(encoded, str):
        raise RescanGateError("fill targets: satisfiedBy must be a base64 string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RescanGateError(f"fill targets: satisfiedBy is not base64: {error}") from error
    if len(data) != cells:
        raise RescanGateError(
            f"fill targets: satisfiedBy holds {len(data)} indices, not one per target ({cells})"
        )
    beyond = [index for index in data if index >= compared]
    if beyond:
        raise RescanGateError(
            f"fill targets: satisfiedBy names index {beyond[0]}, but only {compared} earlier "
            "epochs were compared with"
        )
    return tuple(data)


def _points(xyz: Any) -> np.ndarray:
    array = np.asarray(xyz, dtype=np.float64)
    points = array.reshape(-1, array.shape[-1]) if array.size else np.zeros((0, 3))
    if points.shape[1] < 2:
        raise RescanGateError("the gate needs at least x and y per point")
    return points


@dataclass(frozen=True)
class RescanGate:
    """One epoch's fill targets: `targets[j - j0, i - i0]` is True for a target cell."""

    site_id: str
    epoch_id: str
    weak_in: tuple[str, ...]
    compared_with: tuple[str, ...]
    level: int
    i0: int
    j0: int
    targets: np.ndarray
    #: sha256 of the packed rows exactly as served (`mask`, base64-decoded).
    mask_sha256: str
    reason: str = ""
    #: Per target, in the mask's order, the index into `compared_with` of the earlier epoch
    #: whose weak cell it re-scanned (`satisfiedBy`).
    satisfied_by: tuple[int, ...] = ()

    @property
    def ni(self) -> int:
        return int(self.targets.shape[1])

    @property
    def nj(self) -> int:
        return int(self.targets.shape[0])

    @property
    def cells(self) -> int:
        return int(np.count_nonzero(self.targets))

    @property
    def cell_m(self) -> float:
        return cell_grid.Lattice(self.level).cell_m

    def rescanned(self) -> dict[tuple[int, int], str]:
        """Per target cell `(i, j)`, the earlier epoch whose weak cell it re-scanned."""
        jj, ii = np.nonzero(self.targets)  # row-major: rows south to north, `i` ascending
        return {
            (self.i0 + int(di), self.j0 + int(dj)): self.compared_with[index]
            for dj, di, index in zip(jj, ii, self.satisfied_by, strict=True)
        }

    @classmethod
    def from_json(cls, path: Path) -> RescanGate:
        """The gate in a file holding the API's `FillTargetsRead` JSON."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RescanGateError(f"fill targets {path}: {error}") from error
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: object) -> RescanGate:
        """The gate in a decoded `FillTargetsRead`, every field checked against the others:
        the policy, frame and bit order this reader knows; a level on the lattice whose cell
        is `cellM` exactly; `rowBytes` and the mask's length for `ni x nj`; zero padding
        bits; `cells` the mask's count; the epochs named consistently with it."""
        if not isinstance(payload, Mapping):
            raise RescanGateError("fill targets must be a JSON object (FillTargetsRead)")
        missing = [key for key in _KEYS if key not in payload]
        if missing:
            raise RescanGateError(f"fill targets: missing {', '.join(missing)}")
        if payload["policy"] != POLICY:
            raise RescanGateError(
                f"fill targets: policy {payload['policy']!r} is not {POLICY!r}, the only one "
                "this gate reads"
            )
        if payload["frame"] != cell_grid.FRAME:
            raise RescanGateError(f"fill targets: frame {payload['frame']!r} is not 'tileset'")
        if payload["bitOrder"] != "little":
            raise RescanGateError(
                f"fill targets: bitOrder {payload['bitOrder']!r} is not 'little' (least "
                "significant bit first)"
            )
        level = _int(payload, "level")
        try:
            lattice = cell_grid.Lattice(level)
        except cell_grid.CellGridError as error:
            raise RescanGateError(f"fill targets: {error}") from error
        cell_m = payload["cellM"]
        if isinstance(cell_m, bool) or not isinstance(cell_m, int | float):
            raise RescanGateError(f"fill targets: cellM must be a number, not {cell_m!r}")
        if float(cell_m) != lattice.cell_m:
            raise RescanGateError(
                f"fill targets: cellM {cell_m} is not level {level}'s {lattice.cell_m} m"
            )
        i0, j0 = _int(payload, "i0"), _int(payload, "j0")
        ni, nj = _int(payload, "ni"), _int(payload, "nj")
        if ni < 0 or nj < 0 or ni * nj > MAX_CELLS:
            raise RescanGateError(f"fill targets: a {ni} x {nj} window is not one the API sends")
        row_bytes = _int(payload, "rowBytes")
        if row_bytes != (ni + 7) // 8:
            raise RescanGateError(f"fill targets: rowBytes {row_bytes} is not ceil({ni} / 8)")
        encoded = payload["mask"]
        if not isinstance(encoded, str):
            raise RescanGateError("fill targets: mask must be a base64 string")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise RescanGateError(f"fill targets: mask is not base64: {error}") from error
        if len(data) != row_bytes * nj:
            raise RescanGateError(
                f"fill targets: mask holds {len(data)} bytes, not {nj} rows of {row_bytes}"
            )
        rows = np.frombuffer(data, dtype=np.uint8).reshape(nj, row_bytes)
        bits = np.unpackbits(rows, axis=1, bitorder="little")
        if bits[:, ni:].any():
            raise RescanGateError("fill targets: bits past ni are set (the window is wrong)")
        targets = bits[:, :ni].astype(bool)
        cells = _int(payload, "cells")
        if cells != int(np.count_nonzero(targets)):
            raise RescanGateError(
                f"fill targets: cells says {cells}, the mask holds {int(targets.sum())}"
            )
        epoch_id = _uuid(payload["epochId"], "epochId")
        weak_in = _uuids(payload["weakIn"], "weakIn")
        compared_with = _uuids(payload["comparedWith"], "comparedWith")
        if not compared_with:
            raise RescanGateError(
                "fill targets: compared with no earlier epoch, so nothing was re-scanned"
            )
        if epoch_id in compared_with:
            raise RescanGateError("fill targets: the epoch is compared with itself")
        if not set(weak_in) <= set(compared_with):
            raise RescanGateError("fill targets: weakIn names an epoch it was not compared with")
        if bool(weak_in) != (cells > 0):
            raise RescanGateError(
                "fill targets: weakIn and cells disagree (a target is weak in an earlier "
                "epoch by definition)"
            )
        satisfied = _satisfied(payload["satisfiedBy"], cells, len(compared_with))
        named = tuple(compared_with[k] for k in sorted(set(satisfied)))
        if weak_in != named:
            raise RescanGateError(
                f"fill targets: weakIn {list(weak_in)} is not the epochs satisfiedBy names "
                f"{list(named)}"
            )
        reason = payload["reason"]
        return cls(
            site_id=_uuid(payload["siteId"], "siteId"),
            epoch_id=epoch_id,
            weak_in=weak_in,
            compared_with=compared_with,
            level=level,
            i0=i0,
            j0=j0,
            targets=targets,
            mask_sha256=hashlib.sha256(data).hexdigest(),
            reason=reason if isinstance(reason, str) else "",
            satisfied_by=satisfied,
        )

    def mask(self, xyz: Any) -> np.ndarray:
        """Per point (x east, y north in the tileset frame; z ignored, cells are columns):
        is it in a target cell? `(n,)` bool; False outside the window and for a point that
        is not finite."""
        points = _points(xyz)
        i, j = cell_grid.Lattice(self.level).index(points)
        inside = (i >= self.i0) & (i < self.i0 + self.ni) & (j >= self.j0) & (j < self.j0 + self.nj)
        out = np.zeros(points.shape[0], dtype=bool)
        out[inside] = self.targets[j[inside] - self.j0, i[inside] - self.i0]
        return out

    def inside(self, xyz: Any, inset_m: float = INSET_M) -> np.ndarray:
        """`mask`, held `inset_m` inside: the point moved by +/-`inset_m` in x and in y is
        still in a target cell every way (a point near the edge between two target cells
        is kept; one near the edge of the targets is not)."""
        points = _points(xyz)[:, :2]
        out = np.ones(points.shape[0], dtype=bool)
        for dx in (-inset_m, inset_m):
            for dy in (-inset_m, inset_m):
                out &= self.mask(points + [dx, dy])
        return out

    def check_tileset(self, tileset: Path) -> str:
        """Refuses unless these can be this tileset's targets: it carries a confidence grid
        (`extras.confidence` -- only a measured epoch has fill targets) whose finest level is
        the gate's and whose plane there has the gate's window, and every target cell is weak and covered there
        (the API takes B's targets from B's own grid, so a genuine answer always is). Returns
        the grid's sha256, which the layer's evidence records (`confidenceSha256`).

        What this cannot tell apart: an earlier epoch A of the same site whose grid has B's
        window (every target is weak in A too, by definition). `verify_layer` closes that at
        publish, against the sha256 of the grid the scan's asset carries now."""
        try:
            grids = cell_grid.from_tileset(Path(tileset))
        except (OSError, cell_grid.CellGridError) as error:
            raise RescanGateError(
                f"{tileset}: its confidence grid cannot be read: {error}"
            ) from error
        grid = grids.get("confidence")
        if grid is None:
            raise RescanGateError(
                f"{tileset} carries no confidence grid (extras.confidence): it was never "
                "measured, so it is no re-scan and has no cells to fill"
            )
        try:
            plane = cell_grid.plane_at(grid, self.level)
        except (KeyError, cell_grid.CellGridError) as error:
            raise RescanGateError(f"{tileset}: {error}") from error
        if self.level != grid.level:
            raise RescanGateError(
                f"the fill targets are at level {self.level}, but {tileset}'s finest is "
                f"{grid.level}: a re-scan's targets are its own finest cells (#59), never a "
                "coarsened parent of cells it saw well"
            )
        window = (plane.i0, plane.j0, plane.ni, plane.nj)
        if window != (self.i0, self.j0, self.ni, self.nj):
            raise RescanGateError(
                f"the fill targets' window {(self.i0, self.j0, self.ni, self.nj)} at level "
                f"{self.level} is not {tileset}'s confidence window {window}: they were "
                f"computed for another epoch than this tileset's (targets for {self.epoch_id})"
            )
        score, reasons = grid.channel("score"), grid.channel("reasons")
        weak = cell_grid.weak_mask(plane, score=score, reasons=reasons)
        covered = cell_grid.covered_mask(plane, score=score, reasons=reasons)
        stray = int(np.count_nonzero(self.targets & ~(weak & covered)))
        if stray:
            raise RescanGateError(
                f"{stray} of the {self.cells} target cells are not weak in {tileset}'s own "
                f"confidence grid: the targets for {self.epoch_id} are another revision's"
            )
        # `decode` checked the planes against it, so it names exactly this grid.
        document = json.loads(Path(tileset).read_text(encoding="utf-8"))
        return str(document["root"]["extras"]["confidence"]["sha256"])

    def evidence(self, confidence_sha256: str | None = None) -> dict[str, object]:
        """What the inferred layer's `extras.evidence.rescanGate` says it was gated by;
        `confidence_sha256`, the measured tileset's grid it was checked against
        (`check_tileset`)."""
        out: dict[str, object] = {
            "policy": POLICY,
            "siteId": self.site_id,
            "epochId": self.epoch_id,
            "weakIn": list(self.weak_in),
            "comparedWith": list(self.compared_with),
            "level": self.level,
            "cellM": self.cell_m,
            "i0": self.i0,
            "j0": self.j0,
            "ni": self.ni,
            "nj": self.nj,
            "cells": self.cells,
            "maskSha256": self.mask_sha256,
        }
        if confidence_sha256 is not None:
            out["confidenceSha256"] = confidence_sha256
        return out


#: What a layer's `rescanGate` and the targets fetched at publish must agree on.
_SAME = ("policy", "siteId", "epochId", "level", "i0", "j0", "ni", "nj", "cells", "maskSha256")


def verify_layer(
    evidence: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
    tileset: Mapping[str, Any],
    current: Callable[[str, str], object],
    positions: Any,
) -> RescanGate:
    """What publish-fill.yml asks before an inferred layer goes beside a scan's tiles:

    * its evidence carries `rescanGate` (it was filled through the gate);
    * the scan's asset is a current revision of that very epoch, on that site, and its
      current `tileset` (the document) carries the very confidence grid the fill was
      checked against (`confidenceSha256`): the layer was made on these tiles' epoch;
    * `current(site_id, epoch_id)` -- the API's fill targets for that epoch, fetched now,
      only once the checks above hold -- are the targets it was filled for (same window
      and the same mask bytes): targets that changed since the fill ran (a re-run of the
      epoch, an earlier epoch measured since) are a refusal;
    * every gaussian of the layer (`positions`, in its tileset frame) is in a target cell.

    Returns the current gate; anything else is a `RescanGateError`."""
    gate = evidence.get("rescanGate")
    if not isinstance(gate, Mapping):
        raise RescanGateError(
            "the layer's evidence carries no rescanGate: it was not filled through the "
            "fill gate (teacher_fill.py fill --still-weak), so it is not published"
        )
    if gate.get("policy") != POLICY:
        raise RescanGateError(
            f"the layer's rescanGate policy {gate.get('policy')!r} is not {POLICY!r}"
        )
    epoch = asset.get("epochId")
    if not epoch:
        raise RescanGateError(
            "the scan's asset is a revision of no epoch (seeded or legacy): nothing was "
            "measured, so no re-scan gate can hold for it"
        )
    if asset.get("validTo"):
        raise RescanGateError("the scan's asset is a superseded revision; name its current one")
    if str(epoch) != gate.get("epochId"):
        raise RescanGateError(
            f"the layer was gated by epoch {gate.get('epochId')}'s targets, but the scan's "
            f"asset is a revision of epoch {epoch}"
        )
    if str(asset.get("siteId")) != gate.get("siteId"):
        raise RescanGateError(
            f"the layer was gated on site {gate.get('siteId')}, the asset is on "
            f"{asset.get('siteId')}"
        )
    extras = (tileset.get("root") or {}).get("extras") or {}
    grid = extras.get("confidence") if isinstance(extras, Mapping) else None
    sha = grid.get("sha256") if isinstance(grid, Mapping) else None
    if not sha:
        raise RescanGateError("the scan's current tileset carries no confidence grid")
    if gate.get("confidenceSha256") != sha:
        raise RescanGateError(
            "the layer was filled on tiles whose confidence grid is not the one the scan's "
            "current tileset carries: another epoch's or another revision's tiles"
        )
    fresh = RescanGate.from_payload(current(str(asset.get("siteId")), str(epoch)))
    now = fresh.evidence()
    changed = [key for key in _SAME if now[key] != gate.get(key)]
    if changed:
        raise RescanGateError(
            f"the fill targets changed since the layer was filled ({', '.join(changed)} "
            "differ): fill it again against the current targets"
        )
    points = _points(positions)
    outside = int(np.count_nonzero(~fresh.mask(points)))
    if outside:
        raise RescanGateError(
            f"{outside} of the layer's {points.shape[0]} gaussians are outside the cells a "
            "re-scan left weak"
        )
    if points.shape[0] == 0:
        raise RescanGateError("the layer holds no gaussians")
    return fresh


def layer_positions(tileset: Path) -> np.ndarray:
    """Every gaussian of **every tile** a layer's `tileset.json` names -- parents, drawn at
    coarse detail, as well as leaves; `content` and 3D Tiles 1.1 `contents`; a GLB two
    tiles name, at both -- in its tileset frame, each tile's origin composed
    (`splat_tiles.tile_placements`): what publish holds to the targets (`verify_layer`).
    `splat_render.load_tileset` reads the leaves alone, which is what a renderer needs and
    not what is published. A tile uri that is not beside the tileset is a
    `RescanGateError`."""
    from rig_tiles import tile_positions
    from splat_tiles import SplatFormatError, tile_placements

    path = Path(tileset)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        placements = tile_placements(document)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SplatFormatError) as error:
        raise RescanGateError(f"{path}: {error}") from error
    folder = path.parent.resolve()
    parts: list[np.ndarray] = []
    # Every place a tile is drawn: two tiles naming one GLB draw it at both.
    for uri, offset in placements:
        target = (folder / uri).resolve()
        if not target.is_relative_to(folder):
            raise RescanGateError(f"{path}: tile {uri!r} is not beside the layer's tileset")
        try:
            parts.append(tile_positions(target, offset))
        except (OSError, ValueError, KeyError) as error:
            raise RescanGateError(f"{path}: tile {uri!r} cannot be read: {error}") from error
    return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=np.float32)
