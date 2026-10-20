"""Attaching sidecars to an asset: one publisher, and every attach a new generation.

`POST /assets/{id}/sidecars`. A caller -- a GitHub workflow, after a segmentation, a
collision backfill, an inferred fill, a streamed-LOD build or a plant rig -- stages its
files in the private bucket under `staging/assets/<asset id>/<token>/`, laid out as they
should sit beside `tileset.json`, and asks for them to be attached. The asset's tileset is
never written in place. Instead, holding the asset's row lock:

1. its current directory in the public bucket is listed -- a generation
   (`runs/<job>/p<generation>/...`) or a legacy prefix published before generations
   existed (`runs/<job>/package/splat/`, the spool, pumpkin and camp scans);
2. `basedOn`, the tileset the caller computed its files against, is checked to hold the
   same tiles as the current one (`sidecars.tiles_fingerprint`), so nothing bound to old
   splats lands on new ones;
3. the staged files are checked: under the prefix, plain paths, allowed extensions and
   sizes, never `tileset.json` or one of the scan's tiles;
4. a new generation is written: every current object (tiles, and every sidecar the
   current generation already has) copied server side, the staged files beside them, and
   last -- once every copy has returned -- `tileset.json` with the merged root extras. All
   of it immutable: nothing writes a generation twice, so a browser may keep it a year.
   What the request replaces is not copied (`_plan`): the old files of a kind it stages
   (the whole file set, or the directory unit), and every kind keyed by the ids of a kind
   it replaces -- the skin, materials and telemetry, when it replaces `instances` --
   unless it sends those too;
5. the new generation is listed back and refused unless its tiles are byte-identical to
   the current ones (`sidecars.tiles_fingerprint`);
6. the asset's URL moves to the new `tileset.json` -- the one exception ADR 0013 makes to
   an asset's URL never being rewritten, because the tiles are the same bytes and the old
   generation stays, immutable, in storage -- with the URL it left appended to its
   `url_history` (when, and why: `sidecar-attach`); each kind dropped that way is flagged
   on the asset (`sidecar_flags`, as a republish flags what it cannot carry), and the
   transaction commits.

A second attach on the same asset waits on the row lock and then builds on the first's
generation, so neither loses the other's files; the worker's register takes the same lock
before it repoints (`app/worker/registration.py`). The lock is held across object-store
copies -- seconds for a scan of a thousand tiles, eight copies at a time -- with
`idle_in_transaction_session_timeout` set for this transaction alone, so a process that
freezes holding it is cut off by the database rather than holding it until TCP gives up.

A failure anywhere before the commit leaves the asset exactly as it was: the half-written
generation is keys nothing points at. The staged files are deleted after a success; a
failed attach leaves them for the caller to retry, and a lifecycle rule on `staging/`
removes what nobody retries (docs/DEPLOYMENT.md).
"""

from __future__ import annotations

import json
import logging
import posixpath
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import HttpUrl, TypeAdapter, ValidationError
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.models import Asset
from app.schemas.sidecar import SidecarAttach, SidecarAttachment
from app.services import epoch_maps, epochs, fill_gate, sidecars
from app.services.assets import asset_to_read
from app.services.errors import ConflictError, InvalidInputError, NotFoundError
from app.services.published import IMMUTABLE_CACHE
from app.storage import ObjectStorage, ObjectSummary, StorageUnavailableError
from app.storage.parallel import TRANSFER_WORKERS, each

log = logging.getLogger("twin.sidecars")

#: How long an attach waits for another attach (or a worker's register) to let go of the
#: asset before answering 409. An attach of a large scan holds it for seconds.
LOCK_WAIT = "60s"
#: How long this transaction may sit idle -- copying -- before the database ends it and
#: releases the lock. Far longer than a copy takes; short of a frozen process's hours.
HOLD_LIMIT = "600s"
#: Postgres's SQLSTATE for `lock_timeout`.
LOCK_NOT_AVAILABLE = "55P03"

#: What each 409 means, as the Problem's `code` -- what tools/captures/attach_sidecars.py
#: branches on, rather than on the words of the detail:
#: `basedOn` no longer holds the asset's tiles; compute the files again on its current ones.
TILES_CHANGED = "tiles_changed"
#: another attach, or a worker's register, held the asset past `LOCK_WAIT`; retry.
BUSY = "busy"
#: the asset or its directory cannot take an attach at all; retrying will not help.
NOT_ATTACHABLE = "not_attachable"

#: Why an attach drops a kind keyed by another's ids: a republish's rule (`carry._why_not`,
#: "it names instances ids, and instances was not carried"), for what an attach did to it.
LEADER_REPLACED = (
    "it names {follows} ids, and an attach replaced {follows}: it was bound to the previous "
    "ones ({basis})"
)
LEADER_REMOVED = "it names {follows} ids, and an attach removed {follows} ({basis})"

#: Why an attach moved the asset off a URL, as its `url_history` records it.
REASON = "sidecar-attach"

_URL = TypeAdapter(HttpUrl)


@dataclass(frozen=True)
class _Directory:
    """One generation (or legacy prefix) as listed: its objects and its tileset."""

    location: sidecars.TilesetLocation
    objects: dict[str, ObjectSummary]
    document: dict[str, Any]


def attach_sidecars(
    db: Session,
    asset_id: uuid.UUID,
    payload: SidecarAttach,
    *,
    staging: ObjectStorage,
    public: ObjectStorage,
) -> SidecarAttachment:
    """Attach the staged files to the asset's tileset in a new generation; see the module."""
    if not (staging.available and public.available):
        raise StorageUnavailableError(
            "Attaching sidecars needs object storage. Set OBJECT_STORAGE_* (see .env.example)."
        )
    # Everything that can be refused without storage or a lock, first.
    prefix = _checked(sidecars.staging_prefix, asset_id, payload.staging_prefix)
    expected = (
        None
        if payload.files is None
        else {_checked(sidecars.check_sidecar_name, rel) for rel in payload.files}
    )
    patch = _checked_patch(payload.extras)
    rig_set = "rig_url" in payload.model_fields_set
    if rig_set and payload.rig_url is not None:
        _checked(sidecars.check_relative, payload.rig_url)

    asset = _lock(db, asset_id)
    try:
        result, staged_keys = _attach(
            db, asset, payload, prefix, expected, patch, rig_set, staging=staging, public=public
        )
    except BaseException:
        db.rollback()
        raise
    _forget(staging, staged_keys)
    return result


def _attach(
    db: Session,
    asset: Asset,
    payload: SidecarAttach,
    prefix: str,
    expected: set[str] | None,
    patch: dict[str, Any],
    rig_set: bool,
    *,
    staging: ObjectStorage,
    public: ObjectStorage,
) -> tuple[SidecarAttachment, list[str]]:
    if epochs.is_superseded(asset):
        # History (decision D3): a re-run replaced this revision, and attaching to it would
        # cut a generation nothing shows.
        live = epochs.current_asset(db, asset.epoch_id) if asset.epoch_id else None
        raise ConflictError(
            f"asset {asset.id} is a superseded revision of its epoch; attach to its current "
            f"asset {live.id if live is not None else '(none)'}",
            code=NOT_ATTACHABLE,
        )
    source = asset.source if isinstance(asset.source, dict) else {}
    url = source.get("url")
    if source.get("type") != "3d-tiles-url" or not isinstance(url, str):
        raise ConflictError(
            f"asset {asset.id} is not a 3D Tiles URL, so it has no sidecars", code=NOT_ATTACHABLE
        )
    here = sidecars.locate(url, public)
    if here is None:
        raise ConflictError(
            f"asset {asset.id}'s tileset ({url}) is not a run's tileset in the public bucket "
            "(runs/<job>/...), so there is no generation to cut from it",
            code=NOT_ATTACHABLE,
        )
    current = _read(public, here)
    _check_base(public, payload.based_on, url, current)
    staged = _staged(staging, prefix, expected)
    if not staged and not patch and not rig_set:
        raise InvalidInputError(f"nothing to attach: {prefix} is empty and no extras were given")

    old_extras = sidecars.root_extras(current.document)
    _check_fills(patch, old_extras, asset)
    extras = sidecars.merge_extras(old_extras, patch)
    tiles = sidecars.content_uris(current.document)
    owners = sidecars.assign(extras, staged)
    for rel in staged:
        if rel == here.entry:
            raise InvalidInputError(
                f"{here.entry} is written by the attach, from `extras`; do not stage it"
            )
        if rel in tiles:
            raise InvalidInputError(
                f"{rel} is one of the scan's tiles; a sidecar cannot replace it"
            )
        kind = sidecars.KINDS_BY_NAME.get(owners.get(rel, ""))
        if kind is not None and not kind.attachable:
            raise InvalidInputError(f"{rel} is part of a {kind.name} sidecar, which {kind.refusal}")

    # What the request replaces, and what was keyed by that and goes with it (`_plan`).
    plan = _plan(current, here.entry, tiles, old_extras, staged, owners, patch)
    for name in plan.dropped:
        for key in sidecars.KINDS_BY_NAME[name].extras:
            extras.pop(key, None)
    kept = plan.kept
    final = set(kept) | set(staged) | {here.entry}
    for key, value in patch.items():
        for uri in sidecars.referenced_uris(key, value):
            rel = uri.removeprefix("./")
            try:
                sidecars.check_relative(rel)
            except sidecars.SidecarPathError as error:
                raise InvalidInputError(f"extras.{key}: {error}") from error
            if rel not in final:
                raise InvalidInputError(
                    f"extras.{key} names {uri}, which is neither staged nor already beside "
                    "the tileset"
                )
    if rig_set and payload.rig_url is not None and payload.rig_url not in final:
        raise InvalidInputError(
            f"rigUrl {payload.rig_url} is neither staged nor beside the tileset"
        )
    _verify_fills(
        db,
        asset,
        _fills_judged(patch, old_extras),
        extras=extras,
        document=current.document,
        staged=staged,
        kept=kept,
        staging=staging,
        public=public,
    )
    too_big = sorted(rel for rel, item in kept.items() if item.size > sidecars.MAX_COPY_BYTES)
    if too_big:
        raise ConflictError(
            f"{too_big[0]} is too large for one server-side copy", code=NOT_ATTACHABLE
        )

    # What the new generation's tiles must be, to the byte (ADR 0013's exception).
    tiles_now = sidecars.tiles_fingerprint(current.document, current.objects)
    if tiles_now is None:
        raise ConflictError(
            f"the store cannot vouch for the bytes of {url}'s tiles (an object without an "
            "ETag), so a new generation could not be shown to hold them unchanged",
            code=NOT_ATTACHABLE,
        )
    body = sidecars.serialize(sidecars.with_extras(current.document, extras))
    generation = sidecars.generation_from(
        [f"from\t{here.key}"]
        + [f"cur\t{rel}\t{item.size}\t{item.etag}" for rel, item in kept.items()]
        + [f"new\t{rel}\t{item.size}\t{item.etag}" for rel, item in staged.items()]
        + [f"extras\t{json.dumps(patch, sort_keys=True)}"]
        + ([f"rig\t{payload.rig_url}"] if rig_set else [])
    )
    if generation == here.generation:
        generation = sidecars.fresh_generation()
    target = here.at(generation)

    copies = [(public.bucket, item.key, rel) for rel, item in sorted(kept.items())] + [
        (staging.bucket, item.key, rel) for rel, item in sorted(staged.items())
    ]

    def copy(entry: tuple[str, str, str]) -> object:
        bucket, key, rel = entry
        return public.copy_object(
            bucket,
            key,
            target.directory + rel,
            content_type=sidecars.content_type_for(rel),
            cache_control=IMMUTABLE_CACHE,
        )

    each(copy, copies, workers=TRANSFER_WORKERS)
    # The barrier: the root goes up only once everything it can name is there.
    public.put_object(target.key, body, "application/json", cache_control=IMMUTABLE_CACHE)
    new_url = public.public_url(target.key)
    _check_identical(public, target, tiles_now, url)

    attached = sorted(sidecars.discover({k: v for k, v in patch.items() if v is not None}, staged))
    present = sidecars.discover(extras, final - tiles - {here.entry})
    carried = sorted(set(present) - set(attached))
    cleared = set(attached) | ({"rig"} if rig_set and payload.rig_url is not None else set())
    removed = sorted(plan.had_files - final)
    now = datetime.now(tz=UTC)
    flags = [
        sidecars.flag(
            name,
            action=sidecars.KINDS_BY_NAME[name].action,
            reason=reason,
            job_id=None,
            at=now,
        )
        for name, reason in plan.dropped.items()
    ]

    asset.source = {**dict(asset.source), "url": new_url}
    asset.url_history = [
        *(asset.url_history or []),
        {"url": url, "movedTo": new_url, "replacedAt": now.isoformat(), "reason": REASON},
    ]
    if rig_set:
        render = dict(asset.render_config)
        if payload.rig_url is None:
            render.pop("rigUrl", None)
        else:
            render["rigUrl"] = payload.rig_url
        asset.render_config = render
    asset.sidecar_flags = sidecars.updated_flags(asset.sidecar_flags, clear=cleared, add=flags)
    # The revision's confidence map is a file of the generation: it moves with it.
    epoch_maps.repoint(db, asset, new_url, final)
    db.commit()
    db.refresh(asset)
    log.info(
        "sidecars: asset %s moved from %s to generation %s (%d copied, %d staged; attached %s, "
        "carried %s, dropped %s, removed %s)",
        asset.id,
        url,
        generation,
        len(kept),
        len(staged),
        attached,
        carried,
        sorted(plan.dropped),
        removed,
    )
    for name, reason in plan.dropped.items():
        log.warning("sidecars: %s dropped from asset %s: %s", name, asset.id, reason)
    return (
        SidecarAttachment(
            url=new_url,
            previous_url=url,
            generation=generation,
            copied=len(kept),
            staged=sorted(staged),
            attached=attached,
            carried=carried,
            dropped=sorted(plan.dropped),
            removed=removed,
            extras=sorted(extras),
            asset=asset_to_read(asset),
        ),
        [item.key for item in staged.values()],
    )


@dataclass(frozen=True)
class _Plan:
    """What of the current generation goes into the new one."""

    #: The current objects copied forward: the tiles, and every file nothing replaced.
    kept: dict[str, ObjectSummary]
    #: Kinds the current generation has that the new one drops, each with the reason.
    dropped: dict[str, str]
    #: Every sidecar file of the current generation, which `removed` is reckoned from.
    had_files: set[str]


def _plan(
    current: _Directory,
    entry: str,
    tiles: set[str],
    old_extras: dict[str, Any],
    staged: dict[str, ObjectSummary],
    owners: dict[str, str],
    patch: dict[str, Any],
) -> _Plan:
    """What the request replaces, and so what of the current generation is not copied.

    * A staged file replaces the same path. One under a directory kind replaces its whole
      unit (one fill's `inferred/<name>/`, all of `sog/`), so a smaller re-run leaves no
      stale chunk; one of any other kind replaces the kind's whole file set, so a new
      `instances.json` takes the old `instances.emb` -- whose rows were the old ids -- with
      it. A sibling meant to stay is staged again: one tool writes a kind's files together
      from the same inputs, and nothing here can tell that an old one matches a new one.
    * A kind's root extras key set to null, with none of its files staged, removes the
      kind: its files go with the key.
    * A kind keyed by another's ids (`follows`: the skin, materials and telemetry by
      `instances.json`'s) holds only where that kind does, the rule a republish applies
      (`carry._why_not`). When the request replaces the kind it follows -- stages a file of
      it, or sets or removes its key -- it is dropped, files and key, and flagged on the
      asset; unless the request sends it too (stages a file of it, or sets or removes its
      key), which is the caller saying what it is bound to now.
    """
    rels = set(current.objects) - tiles - {entry}
    had_owners = sidecars.assign(old_extras, rels)
    had = sidecars.discover(old_extras, rels)
    by_key = sidecars.KINDS_BY_EXTRAS
    staged_kinds = {owners[rel] for rel in staged if rel in owners}
    patched = {by_key[key].name for key in patch if key in by_key}
    removing = {by_key[key].name for key, v in patch.items() if v is None and key in by_key}
    removing -= staged_kinds
    replacing = staged_kinds | patched

    dropped: dict[str, str] = {}
    for name in sidecars.followers(replacing):
        if name not in had or name in replacing:
            continue
        kind = sidecars.KINDS_BY_NAME[name]
        template = LEADER_REMOVED if kind.follows in removing else LEADER_REPLACED
        dropped[name] = template.format(follows=kind.follows, basis=kind.basis)

    whole = set(dropped) | removing
    whole |= {name for name in staged_kinds if not sidecars.KINDS_BY_NAME[name].dirs}
    units = set()
    for rel in staged:
        owner = sidecars.KINDS_BY_NAME.get(owners.get(rel, ""))
        units.add(owner.unit(rel) if owner is not None else rel)
    kept = {
        rel: item
        for rel, item in current.objects.items()
        if rel != entry
        and (rel in tiles or not (_replaced(rel, units) or had_owners.get(rel, "") in whole))
    }
    return _Plan(kept=kept, dropped=dropped, had_files=rels)


def _lock(db: Session, asset_id: uuid.UUID) -> Asset:
    """The asset, locked for this transaction; 409 after `LOCK_WAIT` of somebody else's."""
    try:
        db.execute(text(f"SET LOCAL lock_timeout = '{LOCK_WAIT}'"))
        db.execute(text(f"SET LOCAL idle_in_transaction_session_timeout = '{HOLD_LIMIT}'"))
        asset = db.execute(
            select(Asset)
            .where(Asset.id == asset_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
    except OperationalError as error:
        db.rollback()
        if getattr(error.orig, "sqlstate", None) == LOCK_NOT_AVAILABLE:
            raise ConflictError(
                f"another attach or publish of asset {asset_id} is in progress; retry shortly",
                code=BUSY,
            ) from error
        raise
    if asset is None:
        db.rollback()
        raise NotFoundError("asset", asset_id)
    return asset


def _read(public: ObjectStorage, location: sidecars.TilesetLocation) -> _Directory:
    try:
        objects = sidecars.list_directory(public, location.directory)
    except sidecars.TooManyObjects as error:
        raise ConflictError(str(error), code=NOT_ATTACHABLE) from error
    if location.entry not in objects:
        raise ConflictError(f"{location.key} is not in the public bucket", code=NOT_ATTACHABLE)
    try:
        document = sidecars.parse_tileset(public.get_object(location.key))
    except sidecars.TilesetError as error:
        raise ConflictError(
            f"{location.key} is not a tileset: {error}", code=NOT_ATTACHABLE
        ) from error
    return _Directory(location=location, objects=objects, document=document)


def _check_base(public: ObjectStorage, based_on: str, url: str, current: _Directory) -> None:
    """Refuse unless `based_on` holds the tiles the asset holds now.

    The same URL is the common case. Another URL is fine when it holds the same tiles --
    an attach that landed first cut a generation from them, or the caller read a legacy
    URL the asset has since moved off -- and a 409 when a republish replaced them, because
    every kind bound to the splats (`instances.json`, `collision.bin`, `sog/`) was computed
    on the old ones.
    """
    if _same_url(based_on, url):
        return
    then = sidecars.locate(based_on, public)
    previous = None
    if then is not None:
        try:
            previous = _read(public, then)
        except ConflictError:
            previous = None
    now_fp = sidecars.tiles_fingerprint(current.document, current.objects)
    then_fp = sidecars.tiles_fingerprint(previous.document, previous.objects) if previous else None
    if now_fp is None or now_fp != then_fp:
        raise ConflictError(
            f"the asset's tiles are no longer the ones at {based_on}: it points at {url}, "
            "which a republish wrote. Compute the sidecars against that tileset and stage "
            "them again.",
            code=TILES_CHANGED,
        )


def _check_identical(
    public: ObjectStorage, target: sidecars.TilesetLocation, tiles_now: str, url: str
) -> None:
    """Refuse unless the generation just written holds the current tiles, to the byte.

    ADR 0013 never rewrites an asset's URL but for an attach, and an attach only because
    it adds files beside the very same tiles. So the new generation is listed back, as it
    now is in the bucket, and its tiles' fingerprint (`sidecars.tiles_fingerprint`: the
    tree, and every tile's size and ETag) must be the current one's before the asset may
    move to it. What was written stays as keys nothing points at, like any failed attach.
    """
    written = _read(public, target)
    if sidecars.tiles_fingerprint(written.document, written.objects) != tiles_now:
        raise ConflictError(
            f"the new generation at {target.key} does not hold the tiles of {url} unchanged "
            "(not the tiles' bytes, or not their tree), so the asset stays where it is",
            code=NOT_ATTACHABLE,
        )


def _staged(
    staging: ObjectStorage, prefix: str, expected: set[str] | None
) -> dict[str, ObjectSummary]:
    try:
        found = sidecars.list_directory(staging, prefix, limit=sidecars.MAX_SIDECAR_FILES)
    except sidecars.TooManyObjects as error:
        raise InvalidInputError(str(error)) from error
    total = 0
    for rel, item in found.items():
        if not item.key.startswith(prefix):  # pragma: no cover - a listing by this prefix
            raise InvalidInputError(f"{item.key} is not under {prefix}")
        _checked(sidecars.check_sidecar_name, rel)
        if item.size > sidecars.MAX_SIDECAR_BYTES:
            raise InvalidInputError(
                f"{rel} is {item.size:,} bytes; a sidecar may be at most "
                f"{sidecars.MAX_SIDECAR_BYTES:,}"
            )
        total += item.size
    if total > sidecars.MAX_ATTACH_BYTES:
        raise InvalidInputError(
            f"{len(found)} staged files hold {total:,} bytes; one attach may hold at most "
            f"{sidecars.MAX_ATTACH_BYTES:,}"
        )
    if expected is not None and expected != set(found):
        missing, unexpected = sorted(expected - set(found)), sorted(set(found) - expected)
        raise InvalidInputError(
            f"{prefix} does not hold exactly `files`: missing {missing}, unexpected {unexpected}"
        )
    return found


def _replaced(rel: str, units: set[str]) -> bool:
    return rel in units or any(unit.endswith("/") and rel.startswith(unit) for unit in units)


def _fills_judged(patch: dict[str, Any], current: dict[str, Any]) -> list[Any]:
    """The inferred layers the request adds or changes: `inferredLayers`, and the layers of a
    bake-off's fill variant (`extras.variants.fill`, apps/web lib/variants.ts) -- the client
    sends the whole `variants` value, so the entries it already had are not judged again."""
    layers = list(patch.get("inferredLayers") or []) if "inferredLayers" in patch else []
    variants = patch.get("variants")
    if isinstance(variants, dict) and isinstance(variants.get("fill"), list):
        had = (current.get("variants") or {}).get("fill") or []
        for variant in variants["fill"]:
            if variant in had or not isinstance(variant, dict):
                continue
            listed = variant.get("inferredLayers")
            layers += listed if isinstance(listed, list) else [None]
    return layers


def _declared_fills(extras: dict[str, Any]) -> list[Any]:
    """Every inferred layer a root's extras declare: `inferredLayers`, and the layers of each
    fill variant (`variants.fill[].inferredLayers`)."""
    layers = list(extras.get("inferredLayers") or [])
    variants = extras.get("variants")
    if isinstance(variants, dict) and isinstance(variants.get("fill"), list):
        for variant in variants["fill"]:
            listed = variant.get("inferredLayers") if isinstance(variant, dict) else None
            layers += listed if isinstance(listed, list) else []
    return layers


def _touched_fills(
    layers: list[Any], staged: dict[str, ObjectSummary], read: Callable[[str], bytes | None]
) -> list[Any]:
    """The `layers` the request stages a file of: the layer's tileset, a file in its folder,
    or -- for a layer beside `tileset.json` itself, with no folder -- a tile it names."""
    touched: list[Any] = []
    for entry in layers:
        uri = entry.get("uri") if isinstance(entry, dict) else None
        if not isinstance(uri, str):
            continue
        rel = uri.removeprefix("./")
        folder = posixpath.dirname(rel)
        if rel in staged or (folder and any(name.startswith(folder + "/") for name in staged)):
            touched.append(entry)
            continue
        if folder:
            continue
        data = read(rel)
        try:
            document = json.loads(data) if data is not None else None
        except ValueError:
            document = None
        tiles = sidecars.content_uris(document) if isinstance(document, dict) else set()
        if tiles & set(staged):
            touched.append(entry)
    return touched


def _check_fills(patch: dict[str, Any], current: dict[str, Any], asset: Asset) -> None:
    """The fill gate's cheap half, before anything is read: every layer the request adds or
    changes (`_fills_judged`) carries a `rescan-gate-v1` gate against the confidence grid
    the asset's tiles carry now, for its epoch, at a plain path (`fill_gate_problem`). A
    422 that says why. Its other half, `_verify_fills`, admits a layer."""
    layers = _fills_judged(patch, current)
    if not layers:
        return
    sha = sidecars.grid_sha256(current)
    epoch = str(asset.epoch_id) if asset.epoch_id is not None else None
    for entry in layers:
        why = sidecars.fill_gate_problem(entry, grid_sha=sha, epoch_id=epoch)
        if why is not None:
            uri = entry.get("uri") if isinstance(entry, dict) else None
            raise InvalidInputError(f"extras.inferredLayers {uri!r} is refused: {why}")


def _verify_fills(
    db: Session,
    asset: Asset,
    layers: list[Any],
    *,
    extras: dict[str, Any],
    document: dict[str, Any],
    staged: dict[str, ObjectSummary],
    kept: dict[str, ObjectSummary],
    staging: ObjectStorage,
    public: ObjectStorage,
) -> None:
    """The fill gate's second barrier (ADR 0016 § Fill; the first is publish-fill.yml), as
    the API computes it (`fill_gate`): the targets recomputed from the stored maps, every
    gaussian of every tile of each layer held to them, read from the new generation's
    files -- staged, or kept from the current one.

    Judged: every layer the request declares anew (`layers`), and every layer the new
    generation declares (`extras`) any of whose files the request stages -- an empty
    `extras` that replaces a layer's tiles, or an unchanged variant entry sent again with
    new files, changes what the viewer shows as surely as a new declaration does. A
    split's own hole fill (`fills/`) goes with its split and is not judged."""

    def read(rel: str) -> bytes | None:
        if rel in staged:
            return staging.get_object(staged[rel].key)
        if rel in kept:
            return public.get_object(kept[rel].key)
        return None

    judged: list[dict[str, Any]] = []
    for entry in [*layers, *_touched_fills(_declared_fills(extras), staged, read)]:
        if not isinstance(entry, dict) or entry in judged:
            continue
        if str(entry.get("uri") or "").removeprefix("./").startswith("fills/"):
            continue
        judged.append(entry)
    sha = sidecars.grid_sha256(extras)
    epoch = str(asset.epoch_id) if asset.epoch_id is not None else None
    for entry in judged:
        why = sidecars.fill_gate_problem(entry, grid_sha=sha, epoch_id=epoch)
        if why is not None:
            raise InvalidInputError(f"inferred layer {entry.get('uri')!r} is refused: {why}")

    root = document.get("root")
    transform = root.get("transform") if isinstance(root, dict) else None
    fill_gate.verify(db, public, asset, judged, scan_transform=transform, read=read)


def _checked_patch(extras: dict[str, Any]) -> dict[str, Any]:
    for key in extras:
        _checked(sidecars.check_extras_key, key)
    size = len(json.dumps(extras, separators=(",", ":")))
    if size > sidecars.MAX_EXTRAS_BYTES:
        raise InvalidInputError(
            f"extras is {size:,} bytes; root extras are a table of contents, at most "
            f"{sidecars.MAX_EXTRAS_BYTES:,}"
        )
    return dict(extras)


def _checked[**P, R](check: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """`check(...)`, with a path the request may not use turned into a 422."""
    try:
        return check(*args, **kwargs)
    except sidecars.SidecarPathError as error:
        raise InvalidInputError(str(error)) from error


def _same_url(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return str(_URL.validate_python(a)) == str(_URL.validate_python(b))
    except ValidationError:
        return False


def _forget(staging: ObjectStorage, keys: list[str]) -> None:
    """Delete what was staged, now that it is published. Best effort: a key left behind
    is private and expires under the bucket's lifecycle rule for `staging/`."""

    def one(key: str) -> None:
        try:
            staging.delete_object(key)
        except Exception:
            log.warning("sidecars: could not delete the staged %s", key, exc_info=True)

    each(one, keys, workers=TRANSFER_WORKERS)
