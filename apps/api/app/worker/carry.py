"""Carrying a scan's sidecars into the generation a republish cuts -- when they still hold.

A worker publish copies a run's own outputs into a generation of its own
(`app/worker/publish.py`). The run's outputs are its tiles, `collision.bin` and
`viewcones.bin`; what was attached beside the live tiles since -- objects from a
segmentation, an inferred fill, a backfilled grid, the streamed LOD, a plant rig
(`app/services/sidecars.py`) -- is not among them. Repointing the asset at the new
generation as it stands made all of that vanish from the live site.

So before the copy, the live generation is read and each sidecar kind in it is decided:

* **the run brings its own** (`collision`, `viewCones` from the packer): the new one wins
  and the old is not carried -- "superseded", which clears any flag for the kind;
* **bound to the splats' positions** (`POSITIONS`: instances, skin, rig): carried when
  every tile of the new tileset has a position checksum the kind's binding lists -- the
  very test the viewer applies before it draws a tile with them. The checksums are
  computed from the new tiles, the run's own (from the workdir where it still has them,
  else from the private bucket), with the function the binding was written with
  (`app/worker/positions.py`), and compared with what the live `instances.json`,
  `skin.json`, `rig.json` and `plants.json` list. A re-pack that kept every position
  (another spherical-harmonics degree) keeps them; a new reconstruction drops them. The
  same tiles keep them without hashing anything;
* **bound to the tiles' bytes** (`TILES`: collision, view cones, sog, a split): carried
  only when the new tiles are the live tiles (`sidecars.tiles_fingerprint`: the same tree
  and every tile's size and ETag). A retried register or a republish of the same bytes
  keeps them; anything else drops them;
* **keyed by instance ids** (`INSTANCES`: materials, telemetry): carried exactly when
  `instances` is;
* **in the scan's frame only** (`FRAME`: inferred layers): carried.

When the tiles are the same, everything the live generation has beyond them is carried,
known kind or not: nothing about the scan changed. When they differ, a root extras key no
kind recognises cannot be judged, so it is dropped too. Every dropped kind becomes a flag
on the asset ("Objects need re-segmenting", `assets.sidecar_flags`) and a warning in the
log, with the reason.

The plan is made with no transaction open, like the publish it feeds; `register` then
takes the asset's row lock and repoints only if the asset still points at the generation
the plan read (`CarryPlan.based_on`). An attach that landed in between makes `register`
raise `registration.LiveMoved`, and the runner plans and publishes again on top of it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services import sidecars
from app.services.sidecars import Dependence
from app.storage import ObjectStorage, ObjectSummary
from app.worker import positions
from app.worker.publish import Publisher, PublishError

log = logging.getLogger("app.worker")

#: Why a kind bound to the splats is dropped.
NEW_TILES = "the run published new tiles, and it was bound to the previous ones ({basis})"
NEW_POSITIONS = (
    "the run published new tiles whose positions are not all ones it was bound to ({basis})"
)
SINGLE_BUCKET = (
    "this deployment publishes from one bucket, where a run's tileset is its own keys and "
    "cannot hold what was attached beside the previous one"
)

#: Which `POSITIONS` kinds still hold for the new tiles, of the ones asked about.
PositionCheck = Callable[[Collection[str]], frozenset[str]]


@dataclass(frozen=True)
class Dropped:
    """A sidecar kind the new generation will not have, and why."""

    kind: str
    action: str
    reason: str


@dataclass(frozen=True)
class CarryPlan:
    """What a republish carries from the live generation, and what it drops.

    `based_on` is the live URL the plan read (None where the asset had none), which
    `register` compares with the asset under its row lock before repointing it.
    `objects` are `(key in the live generation, path beside the new tileset.json)`;
    `document` is the new `tileset.json` with the carried extras, or None to publish the
    run's own unchanged.
    """

    based_on: str | None
    objects: tuple[tuple[str, str], ...] = ()
    document: bytes | None = None
    carried: tuple[str, ...] = ()
    dropped: tuple[Dropped, ...] = ()
    superseded: tuple[str, ...] = ()
    #: Every kind the run's own tileset has (the packer's `collision`, `viewCones`): a flag
    #: for any of them is cleared, whether or not the live generation had one.
    provided: tuple[str, ...] = ()
    #: Lines for the generation's hash: the carried objects' bytes and the new extras, so
    #: a publish that carries something never lands on a generation that did not.
    lines: tuple[str, ...] = field(default=())

    @property
    def keep_rig(self) -> bool:
        return "rig" in self.carried


def plan_carry(
    publisher: Publisher,
    *,
    live_url: str | None,
    tiles_prefix: str,
    entry: str,
    tiles_dir: Path | None = None,
) -> CarryPlan:
    """Decide what of the live generation at `live_url` goes into the run's new one.

    `tiles_prefix` and `entry` are the run's tileset directory and `tileset.json` in the
    private bucket; `tiles_dir`, where the worker still has it, the same directory on disk,
    which the position checksums are read from rather than downloaded (a tile missing there
    is read from the bucket). A live URL outside the bucket a browser reads (an ion asset, a
    seeded `sites/` scan) has nothing here to carry. A store that cannot be read raises
    `PublishError`, which withholds the publish: dropping every sidecar because R2 had a
    bad minute would take objects off a site that has nothing wrong with it.
    """
    if live_url is None:
        return CarryPlan(based_on=None)
    source = publisher.public if publisher.splits_buckets else publisher.private
    try:
        here = sidecars.locate(live_url, source)
        if here is None:
            return CarryPlan(based_on=live_url)
        live_objects = sidecars.list_directory(source, here.directory)
        if here.entry not in live_objects:
            return CarryPlan(based_on=live_url)
        try:
            live = sidecars.parse_tileset(source.get_object(here.key))
        except sidecars.TilesetError:
            log.warning("carry: %s is not a tileset; nothing of it is carried", live_url)
            return CarryPlan(based_on=live_url)
        directory = tiles_prefix.rstrip("/") + "/"
        new_objects = sidecars.list_directory(publisher.private, directory)
        unreadable = None
        try:
            new = sidecars.parse_tileset(publisher.private.get_object(entry))
        except sidecars.TilesetError as error:
            # Published as it is (the copy does not read it), but nothing can be merged
            # into it or compared with it.
            new, unreadable = {"root": {}}, f"the run's tileset.json could not be read ({error})"
    except PublishError:
        raise
    except Exception as error:
        raise PublishError(f"could not read what {live_url} carries: {error}") from error
    return _decide(
        publisher,
        live_url=live_url,
        live=live,
        live_objects=live_objects,
        live_entry=here.entry,
        new=new,
        new_objects=new_objects,
        new_entry=entry[len(directory) :],
        unreadable=unreadable,
        held=_position_check(
            source,
            live_objects,
            publisher.private,
            new,
            new_objects,
            tiles_dir=tiles_dir,
            live_url=live_url,
        ),
    )


def _position_check(
    source: ObjectStorage,
    live_objects: Mapping[str, ObjectSummary],
    private: ObjectStorage,
    new: Mapping[str, Any],
    new_objects: Mapping[str, ObjectSummary],
    *,
    tiles_dir: Path | None,
    live_url: str,
) -> PositionCheck:
    """The check `_decide` runs on the `POSITIONS` kinds it would otherwise drop.

    Each kind's binding files are read from the live generation and every tile of the new
    tileset is hashed (`positions.tile_checksum`), stopping as soon as no kind is left that
    lists every tile so far: a new reconstruction is ruled out by its first tile. A tile
    that is not a splat GLB this can read, or one the tileset names and the run does not
    have, vouches for nothing. A store that cannot be read raises `PublishError`, as every
    read of the plan does.
    """

    def check(names: Collection[str]) -> frozenset[str]:
        try:
            return held(names)
        except Exception as error:
            raise PublishError(f"could not read what {live_url} carries: {error}") from error

    def held(names: Collection[str]) -> frozenset[str]:
        accepted: dict[str, frozenset[str]] = {}
        for name in names:
            kind = sidecars.KINDS_BY_NAME[name]
            documents = {
                file: source.get_object(live_objects[file].key)
                for file, _ in kind.checksums
                if file in live_objects
            }
            listed = sidecars.bound_checksums(kind, documents)
            if listed:
                accepted[name] = listed
        uris = sorted(sidecars.content_uris(new))
        if not accepted or not uris:
            return frozenset()
        remaining = set(accepted)
        for count, uri in enumerate(uris, start=1):
            data = _tile_bytes(uri, private, new_objects, tiles_dir=tiles_dir)
            if data is None:
                log.info("carry: %s is not among the run's tiles; no binding holds", uri)
                return frozenset()
            try:
                checksum = positions.tile_checksum(data)
            except positions.TileFormatError as error:
                log.info("carry: %s: no position checksum (%s); no binding holds", uri, error)
                return frozenset()
            remaining = {name for name in remaining if checksum in accepted[name]}
            if not remaining:
                log.info(
                    "carry: %s (%s, tile %d of %d) is in no binding from %s",
                    uri,
                    checksum,
                    count,
                    len(uris),
                    live_url,
                )
                return frozenset()
        log.info(
            "carry: every one of the run's %d tiles has positions %s binds",
            len(uris),
            sorted(remaining),
        )
        return frozenset(remaining)

    return check


def _tile_bytes(
    uri: str,
    private: ObjectStorage,
    new_objects: Mapping[str, ObjectSummary],
    *,
    tiles_dir: Path | None,
) -> bytes | None:
    """One of the run's tiles: from disk where the worker has it, else from the bucket."""
    if tiles_dir is not None and ".." not in uri.split("/"):
        local = tiles_dir / uri
        if local.is_file():
            return local.read_bytes()
    found = new_objects.get(uri)
    return private.get_object(found.key) if found is not None else None


def _decide(
    publisher: Publisher,
    *,
    live_url: str,
    live: dict[str, Any],
    live_objects: Mapping[str, ObjectSummary],
    live_entry: str,
    new: dict[str, Any],
    new_objects: Mapping[str, ObjectSummary],
    new_entry: str,
    unreadable: str | None = None,
    held: PositionCheck | None = None,
) -> CarryPlan:
    live_extras, new_extras = sidecars.root_extras(live), sidecars.root_extras(new)
    live_rels = set(live_objects) - {live_entry} - sidecars.content_uris(live)
    new_rels = set(new_objects) - {new_entry} - sidecars.content_uris(new)
    live_kinds = sidecars.discover(live_extras, live_rels)
    new_kinds = sidecars.discover(new_extras, new_rels)
    fingerprint = sidecars.tiles_fingerprint(live, live_objects)
    same = (
        unreadable is None
        and fingerprint is not None
        and fingerprint == sidecars.tiles_fingerprint(new, new_objects)
    )
    owners = sidecars.assign(live_extras, live_rels)
    # Not the same bytes: a kind bound to positions may still hold, which only hashing the
    # new tiles can tell -- so they are hashed only when such a kind would otherwise go.
    positional = [
        name
        for name in live_kinds
        if name not in new_kinds and sidecars.KINDS_BY_NAME[name].depends is Dependence.POSITIONS
    ]
    kept_by_positions = (
        held(positional)
        if held is not None
        and positional
        and not same
        and unreadable is None
        and publisher.splits_buckets
        else frozenset()
    )

    carry_all = same and publisher.splits_buckets
    carried: list[str] = []
    dropped: list[Dropped] = []
    superseded: list[str] = []
    # Kinds that follow another are decided after it.
    ordered = sorted(live_kinds, key=lambda n: sidecars.KINDS_BY_NAME[n].follows is not None)
    for name in ordered:
        kind = sidecars.KINDS_BY_NAME[name]
        if name in new_kinds:
            superseded.append(name)
            continue
        reason = (
            None
            if carry_all
            else unreadable
            or _why_not(
                kind,
                same=same,
                positioned=kept_by_positions,
                carried=carried,
                split=publisher.splits_buckets,
            )
        )
        if reason is None:
            carried.append(name)
        else:
            dropped.append(Dropped(kind=name, action=kind.action, reason=reason))

    unknown = [key for key in sidecars.unknown_extras(live_extras) if key not in new_extras]
    if carry_all:
        # The same tiles: everything beside them is still true, whatever it is -- except
        # what the run made again itself, whose files and keys are the run's.
        replaced = {k for name in superseded for k in sidecars.KINDS_BY_NAME[name].extras}
        files = sorted(rel for rel in live_rels if rel not in new_objects)
        extras = {
            k: v
            for k, v in live_extras.items()
            if k not in new_extras and k not in replaced and k not in sidecars.TILESET_EXTRAS
        }
        carried += unknown
    else:
        files = sorted(
            rel for rel in live_rels if owners.get(rel) in carried and rel not in new_objects
        )
        extras = _carried_extras(live_extras, new_extras, carried, set(files))
        for key in unknown:
            dropped.append(
                Dropped(
                    kind=key,
                    action=f"Unrecognised sidecar extras.{key} needs re-attaching",
                    reason=unreadable
                    or (
                        "the run published new tiles, and nothing here knows what it depends on"
                        if publisher.splits_buckets
                        else SINGLE_BUCKET
                    ),
                )
            )
        stray = sorted(rel for rel in live_rels if rel not in owners and rel not in new_objects)
        if stray:
            log.warning("carry: %s: not carried, belonging to no sidecar kind: %s", live_url, stray)

    document = None
    if extras:
        document = sidecars.serialize(sidecars.with_extras(new, {**new_extras, **extras}))
    objects = tuple((live_objects[rel].key, rel) for rel in files)
    lines = tuple(
        f"carry\t{rel}\t{live_objects[rel].size}\t{live_objects[rel].etag}" for rel in files
    ) + ((f"tileset\t{hashlib.sha256(document).hexdigest()}",) if document is not None else ())
    for name in carried:
        log.info("carry: %s carried from %s", name, live_url)
    for name in superseded:
        log.info("carry: %s not carried from %s: the run made its own", name, live_url)
    for gone in dropped:
        log.warning(
            "carry: %s dropped from %s: %s (%s)", gone.kind, live_url, gone.reason, gone.action
        )
    return CarryPlan(
        based_on=live_url,
        objects=objects,
        document=document,
        carried=tuple(carried),
        dropped=tuple(dropped),
        superseded=tuple(superseded),
        provided=tuple(sorted(new_kinds)),
        lines=lines,
    )


def _why_not(
    kind: sidecars.SidecarKind,
    *,
    same: bool,
    positioned: Collection[str],
    carried: list[str],
    split: bool,
) -> str | None:
    """None to carry `kind`, else the reason it is dropped."""
    if not split:
        return SINGLE_BUCKET
    if kind.depends is Dependence.FRAME:
        return None
    if kind.depends is Dependence.INSTANCES:
        follows = kind.follows or "instances"
        if follows in carried:
            return None
        return f"it names {follows} ids, and {follows} was not carried"
    if same:
        return None
    if kind.depends is Dependence.POSITIONS:
        return None if kind.name in positioned else NEW_POSITIONS.format(basis=kind.basis)
    return NEW_TILES.format(basis=kind.basis)


def _carried_extras(
    live: Mapping[str, Any], new: Mapping[str, Any], carried: list[str], files: set[str]
) -> dict[str, Any]:
    """The root extras of the carried kinds, where the new tileset has none of its own.

    A list of layers keeps only the entries whose files were carried: a split's fills are
    declared in `inferredLayers` but live under `fills/`, and go with the split.
    """
    out: dict[str, Any] = {}
    for name in carried:
        for key in sidecars.KINDS_BY_NAME[name].extras:
            if key not in live or key in new:
                continue
            value = live[key]
            if isinstance(value, list):
                value = [
                    entry
                    for entry in value
                    if all(
                        uri.removeprefix("./") in files
                        for uri in sidecars.referenced_uris(key, [entry])
                    )
                ]
                if not value:
                    continue
            out[key] = value
    return out
