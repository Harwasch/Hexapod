"""Sidecars: what is published beside a scan's tiles that is not the tiles.

A scan's directory in the public bucket is its `tileset.json`, the tiles it names, and a
growing set of files that other steps add beside them and declare on the root tile's
`extras` -- the segmentation's `instances.json`, a backfilled `collision.bin`, an inferred
fill under `inferred/<name>/`, PlayCanvas's streamed level of detail under `sog/`, a plant
rig. The web reads each one off `root.extras` (apps/web/src/lib/instances.ts, collision.ts,
inferred.ts, ...) or, for `sog/` and the rig, from a path beside the tileset.

Those files used to be added **in place** by GitHub workflows that read the live
`tileset.json`, injected a key, and wrote it back. That breaks three ways: two workflows
running at once drop each other's key; inside a published generation, which is served
`immutable` for a year (`app/services/published.py`), a rewritten sidecar is stale in every
cache that has it; and a worker republish of the scan cuts a generation from the run's own
outputs, which have none of them, so objects, collision, fill and streamed LOD vanish from
the live site.

So there is one publisher. A sidecar is attached through the API
(`app/services/attach.py`), which cuts a **new** generation -- the current tiles and every
sidecar they already have, copied server side, the new files beside them, and a
`tileset.json` with the merged extras written last -- and repoints the asset under a row
lock. And a worker republish carries each sidecar of the live generation into the new one
only when it can still be true of the new splats (`app/worker/carry.py`). This module is
what the two agree on: which files are sidecars of which kind, what each kind depends on,
what a staged file may be called, and when two tilesets hold the same tiles. It does no
I/O beyond listing a directory, and imports nothing from `app.worker`, so the API can use
it (as it uses `published.py`).

**What a kind depends on** decides whether it survives new tiles, and it was read off the
code that writes each one (docs/SCENE_OBJECTS.md, section 8, has the table):

* `TILES` -- bound to the exact splats in the tiles. `instances.json`, `skin.json` and the
  rig's `plants.json` key every tile by the FNV-1a checksum of its decoded positions
  (`synthetic_tree.checksum_positions`, mirrored by `packages/world`'s `checksumPositions`)
  and run-length encode ids in that tile's own gaussian order; the viewer refuses a tile
  whose checksum the binding does not list. `collision.bin` and `viewcones.bin` are grids
  computed from the kept splats; `sog/` is the leaves' gaussians themselves, re-encoded. A
  new reconstruction moves every position, so these are carried only when the new tiles are
  the same tiles (`tiles_fingerprint`).
* `INSTANCES` -- keyed by `instances.json` ids (`materials.json`, `telemetry.json`). A
  re-segmentation renumbers them, so they go wherever `instances` goes.
* `FRAME` -- placed in the tileset's local frame with no splat indices or checksums: an
  inferred layer is a tileset of its own whose root transform is the scan's
  (publish-fill.yml checks exactly that). It is carried whatever the splats are.
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import re
import secrets
import uuid
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.services.published import GENERATION_LENGTH, is_published, published_key, unpublished
from app.storage import ObjectStorage
from app.storage.base import ObjectSummary


class Dependence(StrEnum):
    """What a sidecar kind is a function of; see the module docstring."""

    TILES = "tiles"
    INSTANCES = "instances"
    FRAME = "frame"


@dataclass(frozen=True)
class SidecarKind:
    """One kind of file published beside the tiles, and what it depends on.

    `files` are names beside `tileset.json`; `dirs` are directories beside it whose whole
    contents belong to the kind. A staged file under a directory kind replaces the old
    contents of its *unit* -- the first `unit_depth` segments of its path: one fill's
    `inferred/<name>/`, or all of `sog/` -- so a smaller re-run leaves no stale chunk
    behind. `action` is what the asset's flag says when the kind is dropped.
    """

    name: str
    extras: tuple[str, ...]
    files: tuple[str, ...] = ()
    dirs: tuple[str, ...] = ()
    unit_depth: int = 1
    depends: Dependence = Dependence.TILES
    #: The kind `INSTANCES` kinds follow, by name.
    follows: str | None = None
    basis: str = ""
    action: str = ""
    #: False for a kind that cannot be attached beside unchanged tiles.
    attachable: bool = True

    def owns(self, rel: str) -> bool:
        return rel in self.files or any(rel.startswith(d) for d in self.dirs)

    def unit(self, rel: str) -> str:
        """What a staged `rel` of this kind replaces: itself, or its directory unit."""
        for directory in self.dirs:
            if rel.startswith(directory):
                parts = rel.split("/")
                if len(parts) > self.unit_depth:
                    return "/".join(parts[: self.unit_depth]) + "/"
        return rel


INSTANCES = SidecarKind(
    name="instances",
    extras=("instances",),
    files=("instances.json", "instances.emb"),
    depends=Dependence.TILES,
    basis=(
        "`tiles` maps each tile's position checksum (FNV-1a of its decoded float32 positions) "
        "to run-length instance ids in that tile's gaussian order (segment_scene.tile_binding, "
        "rebind_instances.py); instances.emb rows are instance ids"
    ),
    action="Objects need re-segmenting",
)
SKIN = SidecarKind(
    name="skin",
    extras=("skin",),
    files=("skin.json", "skin.bin"),
    depends=Dependence.TILES,
    basis=(
        "`tiles` maps tile checksums to skin runs and to rows of skin.bin in each tile's "
        "gaussian order; each skin moves an instances.json id"
    ),
    action="Skins need refitting",
)
MATERIALS = SidecarKind(
    name="materials",
    extras=("materials",),
    files=("materials.json",),
    depends=Dependence.INSTANCES,
    follows="instances",
    basis="one record per instances.json id",
    action="Materials need re-pointing at the new objects",
)
TELEMETRY = SidecarKind(
    name="telemetry",
    extras=("telemetry",),
    files=("telemetry.json",),
    depends=Dependence.INSTANCES,
    follows="instances",
    basis="pose streams bound to instances.json ids",
    action="Telemetry bindings need re-pointing",
)
OBJECTS = SidecarKind(
    name="objects",
    extras=("objects", "split"),
    dirs=("objects/", "fills/"),
    unit_depth=2,
    depends=Dependence.TILES,
    basis=(
        "split_objects.py rewrites the scan's tiles without the objects and binds the "
        "object tiles in instances.json"
    ),
    action="Split objects need re-splitting",
    # A split rewrites the scan's own tiles: it is a new tileset, not files beside one.
    attachable=False,
)
COLLISION = SidecarKind(
    name="collision",
    extras=("collision",),
    files=("collision.bin",),
    depends=Dependence.TILES,
    basis=(
        "splat_tiles.collision_grid: the solid cells of the kept splats (opacity_min) in the "
        "tileset's local ENU frame"
    ),
    action="Collision needs a backfill",
)
VIEW_CONES = SidecarKind(
    name="viewCones",
    extras=("viewCones",),
    files=("viewcones.bin",),
    depends=Dependence.TILES,
    basis="view_cones: per cell of the splats' grid, the directions it was seen from",
    action="View cones need rebuilding",
)
INFERRED = SidecarKind(
    name="inferredLayers",
    extras=("inferredLayers",),
    dirs=("inferred/",),
    unit_depth=2,
    depends=Dependence.FRAME,
    basis=(
        "a tileset of its own in the scan's local ENU frame (its root transform is the "
        "scan's, publish-fill.yml checks it), with no splat indices or checksums"
    ),
    action="Inferred fill needs re-linking",
)
NATIVE_LOD = SidecarKind(
    name="nativeLod",
    extras=("nativeLod",),
    dirs=("sog/",),
    depends=Dependence.TILES,
    basis=(
        "splat-transform's merge of every leaf tile's SPZ, Morton-ordered and decimated: "
        "the old splats themselves, re-encoded"
    ),
    action="Streamed LOD needs a backfill",
)
RIG = SidecarKind(
    name="rig",
    extras=(),
    files=("rig.json", "motion.json", "plants.json"),
    depends=Dependence.TILES,
    basis=(
        "scene_plants: rig.json is stamped with the tiles' checksums (rig_tiles.stamp) and "
        "plants.json binds per tile checksum; the asset's renderConfig.rigUrl points at it"
    ),
    action="Plants need re-rigging",
)

#: Every kind, in the order a file is assigned to one: the first that owns it.
KINDS: tuple[SidecarKind, ...] = (
    INSTANCES,
    SKIN,
    MATERIALS,
    TELEMETRY,
    OBJECTS,
    COLLISION,
    VIEW_CONES,
    INFERRED,
    NATIVE_LOD,
    RIG,
)
KINDS_BY_NAME: dict[str, SidecarKind] = {kind.name: kind for kind in KINDS}
KINDS_BY_EXTRAS: dict[str, SidecarKind] = {key: kind for kind in KINDS for key in kind.extras}

#: Root extras that belong to the tileset itself rather than to a sidecar: the packer's
#: gaussian count on the root tile (tools/captures/splat_tiles.py). Never patched, never
#: carried: the new tileset has its own.
TILESET_EXTRAS = frozenset({"gaussians"})

# --- what a staged file may be ------------------------------------------------------

#: Where a caller stages the files it attaches, in the private bucket:
#: `staging/assets/<asset id>/<token>/<path beside tileset.json>`. Private, so nothing is
#: world-readable before it has been checked, and per asset so a request can only take
#: files staged for the asset it names.
STAGING_ROOT = "staging/assets"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
#: One segment of a path beside `tileset.json`: plain, and never `.`, `..` or hidden.
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_DEPTH = 6
MAX_PATH = 512
#: A root extras key a request may set.
_EXTRAS_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")

OCTET = "application/octet-stream"
#: The extensions a sidecar may have, and the type each is published with. Every one is
#: in the tile proxy's own list (`CONTENT_TYPES` in functions/r2/[[path]].js), which 404s
#: anything else; tests/test_sidecar_attach.py holds the two together.
SIDECAR_TYPES: dict[str, str] = {
    "json": "application/json",
    "glb": "model/gltf-binary",
    "bin": OCTET,
    "emb": OCTET,
    "f32": OCTET,
    "u8": OCTET,
    "webp": "image/webp",
}
#: The type an object copied into a generation is labelled with, by extension: the proxy's
#: table again. Stated rather than guessed, because `mimetypes` reads the host's own tables
#: and calls `.emb` `chemical/x-embl-dl-nucleotide` on some of them.
COPY_TYPES: dict[str, str] = {
    **SIDECAR_TYPES,
    "b3dm": OCTET,
    "pnts": OCTET,
    "spz": OCTET,
    "ply": OCTET,
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
}

#: Ceilings on one attach. A sidecar is small next to its tiles -- the camp's
#: instances.emb is ~15 MB, a SOG chunk a few -- so these are refusals of a mistake, not
#: budgets.
MAX_SIDECAR_FILES = 5_000
MAX_SIDECAR_BYTES = 1024**3
MAX_ATTACH_BYTES = 8 * 1024**3
MAX_EXTRAS_BYTES = 256 * 1024
#: Objects in one scan's directory. Matches `publish.MAX_PUBLISHED_OBJECTS`: a refusal,
#: never a truncation, because a generation missing tiles is a site with holes.
MAX_DIRECTORY_OBJECTS = 20_000
#: One `CopyObject`'s ceiling on S3 and R2.
MAX_COPY_BYTES = 5 * 1024**3


class SidecarPathError(ValueError):
    """A path, prefix or key that a request may not use, and why."""


class TooManyObjects(RuntimeError):  # noqa: N818 - a refusal, named for what it found
    """A directory with more objects than one generation may hold."""


def check_relative(rel: str) -> str:
    """`rel` if it is a plain path beside `tileset.json`, else `SidecarPathError`.

    Every segment is letters, digits, `.`, `_` or `-`, starting with a letter or digit, so
    nothing can climb out of the directory (`..`), hide (`.x`) or be empty (`a//b`).
    """
    if not rel or len(rel) > MAX_PATH:
        raise SidecarPathError(f"{rel!r} is not a path beside tileset.json")
    parts = rel.split("/")
    if len(parts) > MAX_DEPTH or not all(_SEGMENT.fullmatch(part) for part in parts):
        raise SidecarPathError(
            f"{rel!r} is not a plain path beside tileset.json (letters, digits, '.', '_' "
            f"and '-' in at most {MAX_DEPTH} segments, none starting with '.')"
        )
    return rel


def extension(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def check_sidecar_name(rel: str) -> str:
    """`rel` if a staged file may be called that: a plain path with an allowed extension."""
    check_relative(rel)
    if extension(rel) not in SIDECAR_TYPES:
        raise SidecarPathError(
            f"{rel!r}: a sidecar's extension must be one of {sorted(SIDECAR_TYPES)}"
        )
    return rel


def content_type_for(rel: str) -> str:
    return COPY_TYPES.get(extension(rel), OCTET)


def staging_prefix(asset_id: uuid.UUID, prefix: str) -> str:
    """The request's staging prefix, if it is `staging/assets/<this asset>/<token>/`."""
    expected = f"{STAGING_ROOT}/{asset_id}/"
    token = prefix.removeprefix(expected).removesuffix("/")
    if not prefix.startswith(expected) or not prefix.endswith("/") or not _TOKEN.fullmatch(token):
        raise SidecarPathError(
            f"stagingPrefix must be {expected}<token>/, a token of letters, digits, '.', "
            f"'_' and '-' (a workflow run id, say); got {prefix!r}"
        )
    return prefix


def check_extras_key(key: str) -> str:
    if not _EXTRAS_KEY.fullmatch(key):
        raise SidecarPathError(f"{key!r} is not a root extras key")
    if key in TILESET_EXTRAS:
        raise SidecarPathError(f"extras.{key} belongs to the tileset, not to a sidecar")
    kind = KINDS_BY_EXTRAS.get(key)
    if kind is not None and not kind.attachable:
        raise SidecarPathError(
            f"extras.{key} is a {kind.name} sidecar, which rewrites the scan's tiles and "
            "cannot be attached beside them"
        )
    return key


# --- tilesets ------------------------------------------------------------------------


class TilesetError(ValueError):
    """A `tileset.json` that is not one."""


def parse_tileset(data: bytes) -> dict[str, Any]:
    try:
        document = json.loads(data)
    except ValueError as error:
        raise TilesetError(f"not JSON: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("root"), dict):
        raise TilesetError("no root tile")
    return document


def root_extras(document: Mapping[str, Any]) -> dict[str, Any]:
    root = document.get("root")
    extras = root.get("extras") if isinstance(root, dict) else None
    return dict(extras) if isinstance(extras, dict) else {}


def with_extras(document: Mapping[str, Any], extras: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of `document` whose root extras are `extras` (no `extras` key when empty)."""
    out = copy.deepcopy(dict(document))
    root = dict(out["root"])
    if extras:
        root["extras"] = dict(extras)
    else:
        root.pop("extras", None)
    out["root"] = root
    return out


def serialize(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


def merge_extras(extras: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """`extras` with each key of `patch` set, or removed where its value is null.

    A JSON merge patch (RFC 7386) one level deep -- a key is replaced whole -- with one
    exception: a list of `{uri, ...}` entries (`inferredLayers`) merged into a list of them
    is merged **by uri**. An entry replaces the one with its uri and every other is kept,
    so a workflow sends only its own layer, and two fills attached one after the other
    both stay declared; had each sent the whole list it read, the second would have
    dropped the first's entry. An empty list replaces the list, as any other value does.
    """
    out = dict(extras)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif value and _uri_list(value) and _uri_list(out.get(key)):
            mine = {entry["uri"]: entry for entry in value}
            kept = [mine.pop(entry["uri"], entry) for entry in out[key]]
            out[key] = kept + [entry for entry in value if entry["uri"] in mine]
        else:
            out[key] = value
    return out


def _uri_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(entry, dict) and isinstance(entry.get("uri"), str) for entry in value
    )


def content_uris(document: Mapping[str, Any]) -> set[str]:
    """Every tile content a tileset names (`content.uri` and 3D Tiles 1.1 `contents`)."""
    found: set[str] = set()
    stack: list[Any] = [document.get("root")]
    while stack:
        tile = stack.pop()
        if not isinstance(tile, dict):
            continue
        contents = [tile.get("content"), *(tile.get("contents") or [])]
        for content in contents:
            uri = content.get("uri") if isinstance(content, dict) else None
            if isinstance(uri, str) and uri:
                found.add(uri.removeprefix("./"))
        children = tile.get("children")
        if isinstance(children, list):
            stack.extend(children)
    return found


def referenced_uris(key: str, value: Any) -> list[str]:
    """The files one root extras value names: `{uri}`, a list of `{uri}` (inferred layers,
    split objects), or -- for `nativeLod` alone -- a bare uri."""
    if isinstance(value, str):
        return [value] if key == "nativeLod" else []
    if isinstance(value, dict):
        uri = value.get("uri")
        return [uri] if isinstance(uri, str) else []
    if isinstance(value, list):
        return [
            entry["uri"]
            for entry in value
            if isinstance(entry, dict) and isinstance(entry.get("uri"), str)
        ]
    return []


def tiles_fingerprint(
    document: Mapping[str, Any], objects: Mapping[str, ObjectSummary]
) -> str | None:
    """Who these tiles are, or None where the store cannot vouch for them.

    The tileset without its root extras -- the tree, its bounds and errors, every content
    uri -- and each content object's size and ETag. Two tilesets with the same fingerprint
    hold the same bytes in the same tiles, so whatever was bound to one is bound to the
    other: a chain of attaches copies the tiles unchanged and keeps it; a new
    reconstruction, a Refine or a re-pack changes it. It is a sufficient test, not a
    necessary one -- tiles re-encoded with the very same positions would fail it -- and
    that is the safe side: a kind dropped by mistake is flagged and rebuilt, while one kept
    by mistake hides the wrong splats. An object without an ETag cannot vouch for its
    bytes, so neither can the fingerprint.
    """
    skeleton = copy.deepcopy(dict(document))
    skeleton["root"] = {k: v for k, v in dict(skeleton["root"]).items() if k != "extras"}
    lines = [json.dumps(skeleton, sort_keys=True, separators=(",", ":"))]
    for uri in sorted(content_uris(document)):
        found = objects.get(uri)
        if found is None or not found.etag:
            return None
        lines.append(f"{uri}\t{found.size}\t{found.etag}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def assign(extras: Mapping[str, Any], rels: Collection[str]) -> dict[str, str]:
    """Each sidecar file among `rels`, by the kind it belongs to.

    A file is the first kind's whose names or directories own it (`KINDS` order); a file
    no pattern owns belongs to the kind whose extras name it, or name a tileset in its
    directory (an inferred layer's tiles beside its own `tileset.json`). Everything else
    -- the tiles, the root `tileset.json` -- is not a sidecar and is absent here.
    """
    named: list[tuple[str, SidecarKind]] = []
    for key, value in extras.items():
        kind = KINDS_BY_EXTRAS.get(key)
        if kind is None:
            continue
        for uri in referenced_uris(key, value):
            named.append((uri.removeprefix("./"), kind))
    out: dict[str, str] = {}
    for rel in rels:
        owner = next((kind for kind in KINDS if kind.owns(rel)), None)
        if owner is None:
            owner = next(
                (
                    kind
                    for uri, kind in named
                    if rel == uri or ("/" in uri and rel.startswith(posixpath.dirname(uri) + "/"))
                ),
                None,
            )
        if owner is not None:
            out[rel] = owner.name
    return out


def discover(extras: Mapping[str, Any], rels: Collection[str]) -> dict[str, tuple[str, ...]]:
    """The sidecar kinds a generation has, each with its files.

    A kind is present when the root declares it or any of its files is there: `sog/` and
    the rig are found beside the tileset without a key (the web probes for one and reads
    the other's path off the asset), and a declared kind whose files are missing is still
    a kind the viewer will look for.
    """
    owned = assign(extras, rels)
    found: dict[str, tuple[str, ...]] = {}
    for kind in KINDS:
        files = tuple(sorted(rel for rel, name in owned.items() if name == kind.name))
        if files or any(_declared(extras.get(key)) for key in kind.extras):
            found[kind.name] = files
    return found


def _declared(value: Any) -> bool:
    """Whether an extras value declares something: `nativeLod: false` says there is none."""
    return value is not None and value is not False and value != [] and value != {}


def unknown_extras(extras: Mapping[str, Any]) -> list[str]:
    """Root extras keys that are neither a known sidecar's nor the tileset's own."""
    return sorted(k for k in extras if k not in KINDS_BY_EXTRAS and k not in TILESET_EXTRAS)


# --- where a tileset is -------------------------------------------------------------


@dataclass(frozen=True)
class TilesetLocation:
    """A run's tileset in a bucket, published into a generation or not.

    `own_key` is the key without a generation, `runs/<job>/<rest>`: what `published_key`
    puts a generation into. A legacy tileset -- published before generations existed, as
    the spool, pumpkin and camp scans were (infra/modal/segment.py `SCANS`) -- is its own
    key, and `generation` is None.
    """

    key: str
    own_key: str
    generation: str | None

    @property
    def directory(self) -> str:
        return self.key.rsplit("/", 1)[0] + "/"

    @property
    def entry(self) -> str:
        return self.key.rsplit("/", 1)[1]

    def at(self, generation: str) -> TilesetLocation:
        return TilesetLocation(
            key=published_key(self.own_key, generation),
            own_key=self.own_key,
            generation=generation,
        )


def locate(url: str, storage: ObjectStorage) -> TilesetLocation | None:
    """Where the tileset at `url` is in `storage`, or None if it is not a run's tileset
    there (an ion asset, a seeded `sites/` scan, another host)."""
    base = storage.public_url("")
    if not url.startswith(base):
        return None
    key = url[len(base) :]
    if not key.startswith("runs/") or any(c in key for c in "?#") or not key.endswith(".json"):
        return None
    own = unpublished(key)
    try:
        published_key(own, "0" * GENERATION_LENGTH)
    except ValueError:
        return None
    generation = key.split("/")[2][1:] if is_published(key) else None
    return TilesetLocation(key=key, own_key=own, generation=generation)


def list_directory(
    storage: ObjectStorage, directory: str, *, limit: int = MAX_DIRECTORY_OBJECTS
) -> dict[str, ObjectSummary]:
    """Every object under `directory` (which ends in `/`), by its path relative to it,
    paged to the end. A directory marker (a zero-byte key ending in `/`) is skipped."""
    found: dict[str, ObjectSummary] = {}
    token: str | None = None
    while True:
        page = storage.list_objects(directory, continuation_token=token)
        for item in page.objects:
            rel = item.key[len(directory) :]
            if rel and not rel.endswith("/"):
                found[rel] = item
        if len(found) > limit:
            raise TooManyObjects(f"{directory} holds more than {limit} objects")
        token = page.next_continuation_token
        if token is None:
            return found


def generation_from(lines: Iterable[str]) -> str:
    """A generation for what is being written: the hash of `lines` where every line can
    vouch for its bytes, a random one where any cannot (an empty ETag). The same inputs
    land on the same keys, so a retried attach overwrites its own half-copy with
    identical bytes; anything else gets keys nobody has fetched."""
    collected = list(lines)
    if not collected or any(line.endswith("\t") for line in collected):
        return secrets.token_hex(GENERATION_LENGTH // 2)
    digest = hashlib.sha256("\n".join(sorted(collected)).encode("utf-8")).hexdigest()
    return digest[:GENERATION_LENGTH]


def fresh_generation() -> str:
    return secrets.token_hex(GENERATION_LENGTH // 2)


# --- the asset's flags --------------------------------------------------------------


def flag(
    kind: str, *, action: str, reason: str, job_id: uuid.UUID | None, at: datetime
) -> dict[str, Any]:
    """One entry of `assets.sidecar_flags`, in the API's own camel case."""
    return {
        "kind": kind,
        "action": action,
        "reason": reason,
        "jobId": str(job_id) if job_id is not None else None,
        "flaggedAt": at.isoformat(),
    }


def updated_flags(
    existing: Iterable[Mapping[str, Any]] | None,
    *,
    clear: Collection[str] = (),
    add: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """The asset's flags with the kinds in `clear` and in `add` removed, then `add`
    appended: one flag per kind, the newest reason winning."""
    added = [dict(entry) for entry in add]
    gone = set(clear) | {str(entry["kind"]) for entry in added}
    kept = [dict(entry) for entry in existing or () if str(entry.get("kind")) not in gone]
    return kept + added
