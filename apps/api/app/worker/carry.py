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
* **bound to the splats** (`TILES`: instances, skin, collision, view cones, sog, rig):
  carried only when the new tiles are the live tiles (`sidecars.tiles_fingerprint`: the
  same tree and every tile's size and ETag). A retried register or a republish of the same
  bytes keeps them; a new reconstruction drops them;
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.services import sidecars
from app.services.sidecars import Dependence
from app.storage import ObjectSummary
from app.worker.publish import Publisher, PublishError

log = logging.getLogger("app.worker")

#: Why a kind bound to the splats is dropped.
NEW_TILES = "the run published new tiles, and it was bound to the previous ones ({basis})"
SINGLE_BUCKET = (
    "this deployment publishes from one bucket, where a run's tileset is its own keys and "
    "cannot hold what was attached beside the previous one"
)


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
    publisher: Publisher, *, live_url: str | None, tiles_prefix: str, entry: str
) -> CarryPlan:
    """Decide what of the live generation at `live_url` goes into the run's new one.

    `tiles_prefix` and `entry` are the run's tileset directory and `tileset.json` in the
    private bucket. A live URL outside the bucket a browser reads (an ion asset, a seeded
    `sites/` scan) has nothing here to carry. A store that cannot be read raises
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
    )


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
            or _why_not(kind, same=same, carried=carried, split=publisher.splits_buckets)
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
    kind: sidecars.SidecarKind, *, same: bool, carried: list[str], split: bool
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
    return None if same else NEW_TILES.format(basis=kind.basis)


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
