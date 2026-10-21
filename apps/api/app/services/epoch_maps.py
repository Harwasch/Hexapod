"""An epoch revision's per-cell maps (`epoch_maps`, ADR 0016): recording and reading them.

`register` records the confidence grid a run shipped beside its tiles as the map of the
asset it published (`record_map`); an epoch then reads the map of its *current* revision
(`confidence_of`), so a re-run's new map replaces the old one in what the epoch says while
the superseded revision keeps its own.

What a run says about its grid is checked before it is believed (`confidence_descriptor`):
the container and version this API can read, a file name beside the tiles and nothing
else, a cell on the site lattice and the summary keys every reader relies on. A descriptor
that fails is logged and not recorded -- the epoch then reads as not measured, which is
what the API can vouch for -- and never fails the registration of tiles that are fine.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urljoin

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset, Epoch, EpochMap
from app.models.enums import EpochMapKind
from app.schemas.confidence import AlignmentCapRead, EpochConfidenceRead

log = logging.getLogger(__name__)

FORMAT = "hexapod.cellgrid"
VERSION = 1
#: The finest cell of the site lattice (tools/captures/cell_grid.py `LATTICE_BASE_M`).
LATTICE_BASE_M = 0.25
#: Levels a plane may be at (`cell_grid.MAX_LEVEL`).
MAX_LEVEL = 30
#: A grid's descriptor is a few kilobytes (eight levels and a summary); past this it is
#: not one, and it is not stored.
MAX_DESCRIPTOR_BYTES = 64 * 1024
#: A file beside `tileset.json`: one plain segment, the packer's `confidence.bin`.
_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SUMMARY_COUNTS = ("cells", "scored", "weak", "unseen")


class DescriptorError(ValueError):
    """A run's grid descriptor that this API cannot vouch for, and why."""


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DescriptorError(f"{what} must be an integer, not {value!r}")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise DescriptorError(f"{what} must be a finite number, not {value!r}")
    return float(value)


def confidence_descriptor(raw: object) -> dict[str, Any]:
    """`raw` (registration.json's `confidence`, the tileset's `extras.confidence`) when it
    is a confidence grid this API can read; `DescriptorError` saying why not otherwise."""
    if not isinstance(raw, Mapping):
        raise DescriptorError("a cell grid descriptor is a JSON object")
    if len(json.dumps(raw)) > MAX_DESCRIPTOR_BYTES:
        raise DescriptorError(f"the descriptor is over {MAX_DESCRIPTOR_BYTES} bytes")
    if raw.get("format") != FORMAT or raw.get("version") != VERSION:
        raise DescriptorError(
            f"not a {FORMAT} v{VERSION}: {raw.get('format')!r} v{raw.get('version')!r}"
        )
    if raw.get("kind") != EpochMapKind.CONFIDENCE.value or raw.get("frame") != "tileset":
        raise DescriptorError(
            f"not a confidence grid in the tileset frame: kind {raw.get('kind')!r}, "
            f"frame {raw.get('frame')!r}"
        )
    uri = raw.get("uri")
    if not isinstance(uri, str) or not _FILE.fullmatch(uri) or uri in (".", ".."):
        raise DescriptorError(f"uri {uri!r} is not a file beside the tileset")
    lattice = raw.get("lattice")
    if not isinstance(lattice, Mapping) or lattice.get("baseM") != LATTICE_BASE_M:
        raise DescriptorError(f"not on the site lattice ({LATTICE_BASE_M} m): {lattice!r}")
    level = _int(lattice.get("level"), "lattice.level")
    if not 0 <= level <= MAX_LEVEL:
        raise DescriptorError(f"lattice level {level} is outside 0..{MAX_LEVEL}")
    cell = _number(raw.get("cellM"), "cellM")
    if cell != math.ldexp(LATTICE_BASE_M, level):
        raise DescriptorError(f"cellM {cell} is not level {level}'s cell")
    levels = raw.get("levels")
    if not isinstance(levels, list) or not levels:
        raise DescriptorError("levels must be a non-empty list")
    if not isinstance(raw.get("sha256"), str):
        raise DescriptorError("no sha256 of the planes")
    summary = raw.get("summary")
    if not isinstance(summary, Mapping):
        raise DescriptorError("no summary")
    for key in _SUMMARY_COUNTS:
        _int(summary.get(key), f"summary.{key}")
    _number(summary.get("weakShare"), "summary.weakShare")
    reasons = summary.get("reasons")
    if not isinstance(reasons, Mapping) or not all(
        isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool)
        for name, count in reasons.items()
    ):
        raise DescriptorError("summary.reasons must map each reason to a count")
    cap = raw.get("alignmentCap")
    if not isinstance(cap, Mapping):
        raise DescriptorError("no alignmentCap")
    _number(cap.get("uncertaintyM"), "alignmentCap.uncertaintyM")
    _number(cap.get("factor"), "alignmentCap.factor")
    _int(cap.get("priorEpochs"), "alignmentCap.priorEpochs")
    if not isinstance(raw.get("policy"), str):
        raise DescriptorError("no policy")
    return dict(raw)


def map_url(tileset_url: str, descriptor: Mapping[str, Any]) -> str:
    """The grid's URL: its `uri` beside `tileset_url`."""
    return urljoin(tileset_url, str(descriptor["uri"]))


def record_map(
    db: Session,
    *,
    epoch: Epoch,
    asset: Asset,
    job_id: uuid.UUID | None,
    descriptor: object,
    tileset_url: str,
) -> EpochMap | None:
    """Make `descriptor` the confidence map of `asset` (a revision of `epoch`), or none.

    The map follows the run that last published the asset's tiles: recorded when it has
    none, replaced when the run says something else (a single bucket rewrites a run's keys
    in place, so the same URL can hold a new grid), removed when the run measured nothing
    (`descriptor` None). A descriptor this API cannot vouch for is logged and removes the
    map too: an epoch reads as not measured rather than as a grid nobody can read. In the
    caller's transaction; not committed.
    """
    existing = db.scalars(
        select(EpochMap).where(
            EpochMap.asset_id == asset.id, EpochMap.kind == EpochMapKind.CONFIDENCE.value
        )
    ).first()
    checked: dict[str, Any] | None = None
    if descriptor is not None:
        try:
            checked = confidence_descriptor(descriptor)
        except DescriptorError as error:
            log.warning(
                "register: epoch %s: the run's confidence grid is not recorded: %s",
                epoch.id,
                error,
            )
    if checked is None:
        if existing is not None:
            db.delete(existing)
        return None
    url = map_url(tileset_url, checked)
    if existing is not None and existing.descriptor == checked and existing.url == url:
        return existing
    row = existing or EpochMap(
        epoch_id=epoch.id, asset_id=asset.id, kind=EpochMapKind.CONFIDENCE.value
    )
    row.job_id = job_id
    row.format = FORMAT
    row.version = VERSION
    row.url = url
    row.cell_m = float(checked["cellM"])
    row.level = int(checked["lattice"]["level"])
    row.descriptor = checked
    row.summary = dict(checked["summary"])
    if existing is None:
        db.add(row)
        epoch.maps.append(row)
    return row


def map_of(
    maps: Iterable[EpochMap], asset_id: uuid.UUID | None, kind: EpochMapKind
) -> EpochMap | None:
    """The `kind` map of revision `asset_id`, from already-loaded rows."""
    if asset_id is None:
        return None
    return next((m for m in maps if m.asset_id == asset_id and m.kind == kind.value), None)


def confidence_to_read(row: EpochMap) -> EpochConfidenceRead:
    descriptor, summary = row.descriptor, row.summary
    cap = descriptor["alignmentCap"]
    return EpochConfidenceRead(
        status="measured",
        asset_id=row.asset_id,
        url=row.url,
        cell_m=row.cell_m,
        level=row.level,
        policy=str(descriptor["policy"]),
        weak_share=float(summary["weakShare"]),
        weak_cells=int(summary["weak"]),
        scored_cells=int(summary["scored"]),
        unseen_cells=int(summary["unseen"]),
        reasons={str(k): int(v) for k, v in dict(summary["reasons"]).items()},
        alignment_cap=AlignmentCapRead(
            uncertainty_m=float(cap["uncertaintyM"]),
            factor=float(cap["factor"]),
            prior_epochs=int(cap["priorEpochs"]),
            source=str(cap.get("source") or "unknown"),
        ),
    )


def maps_of_asset(db: Session, asset_id: uuid.UUID) -> list[EpochMap]:
    return list(db.scalars(select(EpochMap).where(EpochMap.asset_id == asset_id)))


def carry_map(
    db: Session, *, epoch: Epoch, source: Asset, target: Asset, tileset_url: str
) -> EpochMap | None:
    """`source`'s confidence map as `target`'s, beside `tileset_url`: a republish of the
    same tiles carried the grid into a new generation (`app/worker/carry.py`) although
    the run measured none itself, so the grid is still true of what the revision shows."""
    found = map_of(maps_of_asset(db, source.id), source.id, EpochMapKind.CONFIDENCE)
    if found is None:
        return None
    return record_map(
        db,
        epoch=epoch,
        asset=target,
        job_id=found.job_id,
        descriptor=found.descriptor,
        tileset_url=tileset_url,
    )


def repoint(db: Session, asset: Asset, tileset_url: str, files: Iterable[str]) -> None:
    """An attach moved `asset` to a new generation at `tileset_url` holding `files`: each
    map follows it when its file was copied there, and goes when it was not. Not
    committed: the attach's transaction."""
    present = set(files)
    for row in maps_of_asset(db, asset.id):
        uri = str(row.descriptor.get("uri", ""))
        if uri in present:
            row.url = map_url(tileset_url, row.descriptor)
        else:
            db.delete(row)
