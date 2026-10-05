"""The motion-skins bake-off: one scan's skins by several methods and handle policies, written
as variants beside its tiles (docs/SCENE_OBJECTS.md §9).

Each variant is a `skin.json` + `skin.bin` in today's format (hexapod.skin v1, `skin_scene.py`)
under `variants/skins/<name>/`, declared on the measured tileset's root as one entry of
`extras.variants.skins` (`{name, label, about, skin}`, the path relative to `tileset.json`).
Today's `extras.skin` is left alone: a viewer that never picks a variant sees what it saw.

Subcommands::

    skin_variants.py fetch URL DIR [--instances] [--only 1,2]   # a published scan's tiles
    skin_variants.py build DIR [--only 1,2 | --whole] [--variants a,b] [--narrow]
    skin_variants.py register TILESET_JSON VARIANTS_JSON         # the merged extras.variants
    skin_variants.py scan NAME WORK [--variants a,b]             # a bake-off scan, end to end
    skin_variants.py publish OUT                                 # what `scan` prepared

`fetch` downloads `tileset.json` and the tiles (with `--only`, just the tiles whose bounds
meet those instances'), and with `--instances` the scan's `instances.json` (and
`materials.json` when declared). `build` reads them, fits every variant over the chosen
objects (`--only`: those instances and everything below them, whatever their behaviour; or
`--whole`: the scan is one object -- the Minnetonka tree -- and its instance is made here),
and writes `DIR/variants/skins/<name>/`, `DIR/variants.json` (the entries to register) and
`DIR/skins_report.json` (per variant and object: handles, class, the wind's anchored modes and
their frequencies at the prior, fit seconds, bytes). `register` prints the root's
`extras.variants` with these entries added or replaced by name, every other system and
variant kept.

`scan` and `publish` are what .github/workflows/publish-skins.yml runs, one bake-off scan
(`BAKEOFF`) at a time. `scan` finds the scan's **current** tileset (a run's scan through the
API, `attach_sidecars.resolve_scan`, by its legacy URL in infra/modal/segment.py `SCANS`; the
Minnetonka tree at its site), fetches and builds, and lays out `WORK/out/<scan>/` as it will
sit beside `tileset.json` (`variants/skins/<name>/skin.json`, `skin.bin`), with `publish.json`
(where it goes) and the report. `publish` merges this run's entries into the
`extras.variants` the tileset declares **now** (re-read at publish time, so a variant another
bake-off registered meanwhile is kept) and:

* a run's scan: writes `attach.json` and attaches through the API (`attach_sidecars.attach`:
  the files staged in the private bucket, a new generation cut, nothing written to the public
  bucket or a `tileset.json` by hand);
* a site (the Minnetonka tree, published by app.seed.publish under `sites/`, which the attach
  refuses: it is not a run's tileset): uploads the files beside its `tileset.json` in the
  public bucket and rewrites that `tileset.json` with only `extras.variants` changed, as
  minnetonka-tree.yml's publish writes the site.

The variants (`VARIANTS`):

| name               | method                         | handles                          |
| ------------------ | ------------------------------ | -------------------------------- |
| `freeform`         | FreeForm/RKPM over the splats  | size (today's rule)              |
| `freeform-stiff`   | FreeForm/RKPM over the splats  | stiffness-aware                  |
| `pinned-stiff`     | FreeForm/RKPM, base pinned     | stiffness-aware                  |
| `tetfem-stiff`     | linear FEM, filled volume, base held | stiffness-aware            |
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import skin_methods
import skin_scene
import skin_wind as sw

#: The bucket answers this agent (infra/modal/segment.py `USER_AGENT`).
USER_AGENT = "curl/8.5.0 (hexapod-segment)"
#: Where variants sit beside `tileset.json`.
VARIANT_ROOT = "variants/skins"


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    policy: str  # "size" | "stiffness"
    label: str
    about: str


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "freeform",
        "freeform",
        "size",
        "FreeForm · size rule",
        "Today's method: FreeForm/RKPM skinning eigenmodes over the splat centres (a shell), "
        "8 to 16 handles by size alone; the wind keeps the mode mixes that leave the base still.",
    ),
    Variant(
        "freeform-stiff",
        "freeform",
        "stiffness",
        "FreeForm · stiffness rule",
        "The same modes, with handles by stiffness class: rigid things none (they move only as "
        "a whole), firm things like pumpkins 4, plants by size, big trees 32.",
    ),
    Variant(
        "pinned-stiff",
        "pinned",
        "stiffness",
        "Pinned FreeForm · stiffness rule",
        "FreeForm/RKPM modes solved with the base held still, so every handle is a bending mode "
        "of a rooted object; handles by stiffness class.",
    ),
    Variant(
        "tetfem-stiff",
        "tetfem",
        "stiffness",
        "Volume FEM · stiffness rule",
        "Linear finite elements on tetrahedra filling the object's occupancy (its unseen inside "
        "taken as solid), base held, splats embedded barycentrically; handles by stiffness "
        "class.",
    ),
)
VARIANTS_BY_NAME = {v.name: v for v in VARIANTS}

METHOD_DOCS = {
    "freeform": skin_scene.DEFAULT_METHOD,
    "pinned": {
        "name": "simplicits-rkpm-pinned",
        "source": "NVIDIA Kaolin (Apache-2.0), vendored in tools/captures/kaolin_rkpm.py; the "
        "base band (skin_scene.anchor_mask) held by a penalty",
        "material": {"uniform": True, "poisson": skin_scene.POISSON},
    },
    "tetfem": {
        "name": "tet-fem-skinning-eigenmodes",
        "source": "tools/captures/skin_methods.py: P1 FEM on a Kuhn tetrahedral mesh of the "
        "voxelised, closed and filled splat occupancy; SciPy eigsh",
        "material": {"uniform": True, "poisson": skin_scene.POISSON},
    },
}


# ----------------------------------------------------------------------------------- fetch


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def _boxes(tileset: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Every content tile's uri and its box (min, max) in the tileset's frame."""
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    def walk(tile: dict) -> None:
        uri = tile.get("content", {}).get("uri")
        box = tile.get("boundingVolume", {}).get("box")
        if uri and box:
            c = np.array(box[:3], np.float64)
            axes = np.array(box[3:12], np.float64).reshape(3, 3)
            half = np.abs(axes).sum(0)
            out.append((uri, c - half, c + half))
        elif uri:
            out.append((uri, np.full(3, -np.inf), np.full(3, np.inf)))
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    return out


def fetch(
    url: str,
    out: Path,
    *,
    instances: bool = False,
    only: Sequence[int] | None = None,
    margin: float = 0.5,
) -> list[str]:
    """`tileset.json` and its tiles (only those meeting `only`'s bounds, when given) into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    base = url.rsplit("/", 1)[0]
    tileset = json.loads(_get(url))
    (out / "tileset.json").write_text(json.dumps(tileset), encoding="utf-8")
    doc = None
    if instances:
        raw = _get(f"{base}/instances.json")
        (out / "instances.json").write_bytes(raw)
        doc = json.loads(raw)
        materials = (tileset["root"].get("extras") or {}).get("materials")
        if isinstance(materials, dict) and materials.get("uri"):
            (out / "materials.json").write_bytes(_get(f"{base}/{materials['uri']}"))
    wanted: list[tuple[np.ndarray, np.ndarray]] | None = None
    if only and doc is not None:
        by_id = {int(i["id"]): i for i in doc["instances"]}
        wanted = []
        for k in only:
            b = by_id[int(k)]["bounds"]
            wanted.append((np.array(b["min"]) - margin, np.array(b["max"]) + margin))
    uris = []
    for uri, low, high in _boxes(tileset):
        if wanted is not None and not any(
            (low <= hi).all() and (high >= lo).all() for lo, hi in wanted
        ):
            continue
        target = out / uri
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(_get(f"{base}/{uri}"))
        uris.append(uri)
    (out / "fetched.json").write_text(json.dumps({"url": url, "tiles": uris}), encoding="utf-8")
    return uris


# ----------------------------------------------------------------------------------- build


def read_scan(
    directory: Path,
    *,
    whole: bool = False,
    label: str = "tree",
    instances: Path | None = None,
    materials: Path | None = None,
) -> tuple[list[skin_scene.TileSplats], dict, dict[int, dict]]:
    """The fetched tiles (only those `fetch` wrote), the instances document (made here for a
    `whole` scan: every splat one object) and the materials by instance."""
    from rig_tiles import tile_positions
    from synthetic_tree import checksum_positions

    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    fetched = directory / "fetched.json"
    uris = (
        json.loads(fetched.read_text(encoding="utf-8"))["tiles"]
        if fetched.exists()
        else [u for u, _, _ in _boxes(tileset)]
    )
    leaves = skin_scene._leaf_uris(tileset)
    tiles: list[skin_scene.TileSplats] = []
    if whole:
        bounds_low, bounds_high = np.full(3, np.inf), np.full(3, -np.inf)
        tile_runs: dict[str, list[int]] = {}
        for uri in uris:
            positions = tile_positions(directory / uri)
            checksum = checksum_positions(positions)
            ids = np.ones(len(positions), np.int64)
            tiles.append(skin_scene.TileSplats(uri, checksum, positions, ids, uri in leaves))
            tile_runs[checksum] = [1, len(positions)]
            if uri in leaves and len(positions):
                bounds_low = np.minimum(bounds_low, positions.min(0))
                bounds_high = np.maximum(bounds_high, positions.max(0))
        doc = {
            "format": "hexapod.instances",
            "version": 1,
            "instances": [
                {
                    "id": 1,
                    "parent": None,
                    "level": 0,
                    "splats": int(sum(len(t.ids) for t in tiles if t.leaf)),
                    "bounds": {"min": bounds_low.tolist(), "max": bounds_high.tolist()},
                    "tags": [{"label": label, "score": 1.0}],
                    "category": "trees",
                    "behaviour": "in-place",
                    # A whole-scan object of an isolated tree (real_tree.py): foliage.
                    "properties": {"vegetation": 1.0, "elastic": 0.7, "rigid": 0.2},
                }
            ],
            "tiles": tile_runs,
        }
    else:
        doc = json.loads((instances or directory / "instances.json").read_text(encoding="utf-8"))
        for uri in uris:
            positions = tile_positions(directory / uri)
            checksum = checksum_positions(positions)
            runs = doc["tiles"].get(checksum)
            if runs is None:
                raise SystemExit(
                    f"{uri} ({checksum}) is not in instances.json: segment these tiles"
                )
            ids = skin_scene.decode_runs(runs)
            if ids.size != positions.shape[0]:
                raise SystemExit(f"{uri}: {positions.shape[0]} gaussians, {ids.size} ids")
            tiles.append(skin_scene.TileSplats(uri, checksum, positions, ids, uri in leaves))
    table: dict[int, dict] = {}
    path = materials or directory / "materials.json"
    if path.exists():
        for record in json.loads(path.read_text(encoding="utf-8")).get("materials", []):
            table[int(record["instance"])] = record
    return tiles, doc, table


def policy_of(
    name: str, materials: Mapping[int, Mapping], *, wide: bool
) -> skin_methods.HandlePolicy:
    if name == "size":
        return skin_methods.size_policy()
    if name == "stiffness":
        return skin_methods.stiffness_policy(materials, wide=wide)
    raise ValueError(f"unknown handle policy {name!r}")


def wind_summary(entry: dict, traits: Mapping | None) -> dict:
    """What the wind makes of a skin at its property prior: how many anchored modes it keeps
    and the lowest ones' frequencies (Hz)."""
    m = int(entry["handles"])
    if m <= 1:
        return {"anchoredModes": 0, "lowestHz": []}
    dynamics = sw.SkinDynamics.from_skin(entry)
    material = sw.material_prior((traits or {}).get("properties"), "in-place")
    model = sw.skin_wind_model(dynamics, material)
    if model is None:
        return {"anchoredModes": 0, "lowestHz": []}
    hz = sorted(float(w) / (2 * np.pi) for w in model.omega)
    return {"anchoredModes": len(hz), "lowestHz": [round(f, 3) for f in hz[:3]]}


def build_variants(
    directory: Path,
    names: Sequence[str],
    *,
    only: Sequence[int] | None = None,
    whole: bool = False,
    wide: bool = True,
    log: bool = True,
    instances: Path | None = None,
    materials: Path | None = None,
    out: Path | None = None,
    flat: bool = False,
) -> tuple[list[dict], dict]:
    """Every named variant into `directory/variants/skins/<name>/`; the entries to register and
    the report."""
    tiles, doc, table = read_scan(directory, whole=whole, instances=instances, materials=materials)
    root = out or directory
    chosen = [1] if whole else list(only or [])
    if not chosen:
        raise SystemExit("choose the objects to skin: --only ID,... (or --whole)")
    owner = skin_scene.owners_of(doc["instances"], chosen)
    by_id = {int(i["id"]): i for i in doc["instances"]}
    entries: list[dict] = []
    report: dict = {"objects": {str(k): skin_methods.traits_of(by_id[k]) for k in chosen}}
    for name in names:
        variant = VARIANTS_BY_NAME[name]
        started = time.perf_counter()
        policy = policy_of(variant.policy, table, wide=wide)
        built = skin_scene.build(
            tiles,
            doc["instances"],
            owner=owner,
            fit=skin_methods.fitter(variant.method, policy, doc["instances"]),
            method={**METHOD_DOCS[variant.method], "handlePolicy": policy.about},
            listed="skinned",
            log=log,
        )
        target = root if flat else root / VARIANT_ROOT / name
        skin_scene.write_skin(target, built)
        seconds = time.perf_counter() - started
        rows = []
        for entry, skin in zip(built.document["skins"], built.skins, strict=True):
            rows.append(
                {
                    "instance": entry["instance"],
                    "label": entry.get("traits", {}).get("label"),
                    "class": entry.get("class"),
                    "handles": entry["handles"],
                    "splats": entry["splats"],
                    "fitSeconds": round(skin.seconds, 2),
                    **wind_summary(entry, entry.get("traits")),
                }
            )
        report[name] = {
            "seconds": round(seconds, 1),
            "skinBin": len(built.blob),
            "skinJson": (target / "skin.json").stat().st_size,
            "rowBytes": built.document["weights"]["rowBytes"],
            "rows": built.document["weights"]["rows"],
            "clipped": built.clipped,
            "skins": rows,
        }
        entries.append(
            {
                "name": variant.name,
                "label": variant.label,
                "about": variant.about,
                "skin": f"{VARIANT_ROOT}/{variant.name}/skin.json",
            }
        )
        if log:
            print(f"{name}: {len(built.skins)} skins, {len(built.blob):,} B, {seconds:.1f} s")
    if flat:
        return entries, report
    # A run of some variants keeps what an earlier run built of the others (by name).
    listed = root / "variants.json"
    if listed.exists():
        now = {e["name"] for e in entries}
        earlier = json.loads(listed.read_text(encoding="utf-8")).get("skins", [])
        kept = [e for e in earlier if e["name"] not in now and (root / e["skin"]).exists()]
        order = [v.name for v in VARIANTS]
        entries = sorted(kept + entries, key=lambda e: order.index(e["name"]))
        old_report = root / "skins_report.json"
        if old_report.exists():
            previous = json.loads(old_report.read_text(encoding="utf-8"))
            report = {
                **{e["name"]: previous[e["name"]] for e in kept if e["name"] in previous},
                **report,
            }
    listed.write_text(json.dumps({"skins": entries}, indent=1) + "\n", encoding="utf-8")
    (root / "skins_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return entries, report


# ------------------------------------------------------------------------------ the scans

#: The bake-off's scans and the objects skinned in each: the spool (1); both pumpkins (2, 3);
#: the camp's vegetation -- shrubs and small trees in two clusters (26, 56, 67, 122 at the
#: north fort; 93, 103, 148 by the east huts), a big shrub (16), a trunk (91) and a pine (276);
#: the Minnetonka tree whole (its scan is the one tree, and has no instances.json).
BAKEOFF: dict[str, dict] = {
    "spool": {"only": (1,)},
    "pumpkin": {"only": (2, 3)},
    "camp": {"only": (16, 26, 56, 67, 93, 103, 122, 148, 276, 91)},
    "minnetonka-tree": {"whole": True, "site": "minnetonka-tree"},
}
#: The public bucket's base (infra/modal/segment.py `PUBLIC` without `/runs`).
PUBLIC_BASE = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev"


def _segment_scans() -> dict[str, str]:
    """infra/modal/segment.py `SCANS`: each run scan's legacy tileset URL."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "infra" / "modal" / "segment.py"
    spec = importlib.util.spec_from_file_location("segment_app", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot read {path}")
    module = importlib.util.module_from_spec(spec)
    try:  # the app module imports modal; only its SCANS table is read here
        import modal  # noqa: F401
    except ImportError:
        sys.modules["modal"] = _ModalStub()  # type: ignore[assignment]
    spec.loader.exec_module(module)
    return dict(module.SCANS)


class _ModalStub:
    """Enough of `modal` for infra/modal/segment.py to import without it (only `SCANS` is
    read): every attribute and call is another stub."""

    def __getattr__(self, name: str) -> object:
        return _ModalStub()

    def __call__(self, *args: object, **kwargs: object) -> object:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _ModalStub()


def locate(name: str) -> dict:
    """Where a bake-off scan is now: `{kind, url, assetId?}` -- `attach` for a run's scan (its
    asset's current tileset), `site` for a site under `sites/`."""
    spec = BAKEOFF[name]
    if spec.get("site"):
        return {"kind": "site", "url": f"{PUBLIC_BASE}/sites/{spec['site']}/splat/tileset.json"}
    import attach_sidecars

    asset = attach_sidecars.resolve_scan(name, legacy_url=_segment_scans()[name])
    return {"kind": "attach", "url": asset["url"], "assetId": asset["assetId"]}


def prepare_scan(name: str, work: Path, names: Sequence[str]) -> Path:
    """Fetches, builds and lays out bake-off scan `name` under `work`; returns the publish
    directory (`work/out/<name>`)."""
    import shutil

    spec = BAKEOFF[name]
    where = locate(name)
    scan_dir = work / "scan" / name
    whole = bool(spec.get("whole"))
    fetch(where["url"], scan_dir, instances=not whole, only=None if whole else spec["only"])
    entries, report = build_variants(
        scan_dir, names, only=None if whole else spec["only"], whole=whole
    )
    out = work / "out" / name
    if out.exists():
        shutil.rmtree(out)
    for entry in entries:
        source = scan_dir / Path(entry["skin"]).parent
        target = out / Path(entry["skin"]).parent
        target.mkdir(parents=True, exist_ok=True)
        for file in ("skin.json", "skin.bin"):
            shutil.copy(source / file, target / file)
    (out.parent / f"{name}.publish.json").write_text(
        json.dumps({**where, "scan": name, "variants": {"skins": entries}}, indent=1) + "\n",
        encoding="utf-8",
    )
    (out.parent / f"{name}.report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    return out


def publish(out: Path) -> dict:
    """Registers and publishes what `prepare_scan` laid out in `out` (see the module comment)."""
    import attach_sidecars

    plan = json.loads((out.parent / f"{out.name}.publish.json").read_text(encoding="utf-8"))
    entries = plan["variants"]
    if plan["kind"] == "attach":
        now = attach_sidecars.resolve_asset(plan["assetId"])
        current = json.loads(_get(now["url"]))
        variants = merged_variants(current["root"].get("extras") or {}, entries)
        # basedOn: the tiles the skins were bound to; the API accepts it while the asset's
        # tiles are those (another attach since only copied them).
        attach_sidecars.write_manifest(
            out, asset_id=plan["assetId"], based_on=plan["url"], extras={"variants": variants}
        )
        return attach_sidecars.attach(out)
    return publish_site(out, plan["url"], entries)


def publish_site(out: Path, url: str, entries: Mapping[str, Sequence[Mapping]]) -> dict:
    """The files beside a site's `tileset.json` in the public bucket, and that `tileset.json`
    rewritten with the merged `extras.variants` (nothing else of it changed)."""
    import os

    import boto3

    bucket = os.environ.get("OBJECT_STORAGE_PUBLIC_BUCKET")
    if not bucket:
        raise SystemExit("missing OBJECT_STORAGE_PUBLIC_BUCKET: the site's public bucket")
    prefix = url.removeprefix(PUBLIC_BASE + "/").rsplit("/", 1)[0] + "/"
    if not prefix.startswith("sites/"):
        raise SystemExit(f"{url} is not a site's tileset under {PUBLIC_BASE}/sites/")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
    )
    files = sorted(path for path in out.rglob("*") if path.is_file())
    types = {".json": "application/json", ".bin": "application/octet-stream"}
    for path in files:
        rel = path.relative_to(out).as_posix()
        if not rel.startswith(VARIANT_ROOT + "/") or path.suffix not in types:
            raise SystemExit(f"{rel} is not a skin variant's file")
        s3.upload_file(
            str(path), bucket, prefix + rel, ExtraArgs={"ContentType": types[path.suffix]}
        )
        print(f"uploaded {bucket}/{prefix}{rel}")
    # Last, once every file it names is there: the tileset with the variants declared.
    key = prefix + "tileset.json"
    tileset = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    extras = tileset["root"].setdefault("extras", {})
    extras["variants"] = merged_variants(extras, entries)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(tileset, separators=(",", ":")).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"declared {len(entries.get('skins', []))} skin variants in {bucket}/{key}")
    return {"url": url, "files": len(files), "variants": extras["variants"]}


# -------------------------------------------------------------------------------- register


def merged_variants(extras: Mapping, entries: Mapping[str, Sequence[Mapping]]) -> dict:
    """`extras.variants` with `entries` (by system) added, each replacing the entry of its name;
    every other system and variant kept, in order."""
    current = extras.get("variants")
    out: dict = {k: list(v) for k, v in current.items()} if isinstance(current, dict) else {}
    for system, mine in entries.items():
        kept = [e for e in out.get(system, []) if isinstance(e, dict)]
        names = {e["name"] for e in mine}
        replaced = [
            next(m for m in mine if m["name"] == e.get("name")) if e.get("name") in names else e
            for e in kept
        ]
        seen = {e.get("name") for e in replaced}
        out[system] = replaced + [dict(m) for m in mine if m["name"] not in seen]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("fetch", help="a published scan's tileset and tiles")
    f.add_argument("url")
    f.add_argument("directory", type=Path)
    f.add_argument("--instances", action="store_true", help="also instances.json (+ materials)")
    f.add_argument("--only", default=None, help="just the tiles meeting these instances")
    b = commands.add_parser("build", help="fit and write the variants")
    b.add_argument("directory", type=Path)
    b.add_argument("--only", default=None, help="comma-separated instance ids to skin")
    b.add_argument("--whole", action="store_true", help="the scan is one object")
    b.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    b.add_argument("--narrow", action="store_true", help="no rows of two texels (trees at 16)")
    b.add_argument("--instances", type=Path, help="instances.json (default: beside the tiles)")
    b.add_argument("--materials", type=Path, help="materials.json (default: beside the tiles)")
    b.add_argument("--out", type=Path, help="where variants/ goes (default: the directory)")
    b.add_argument("--flat", action="store_true", help="one variant, its files right in --out")
    r = commands.add_parser("register", help="print the merged extras.variants")
    r.add_argument("tileset", type=Path)
    r.add_argument("variants", type=Path)
    sc = commands.add_parser("scan", help="a bake-off scan: locate, fetch, build, lay out")
    sc.add_argument("name", choices=sorted(BAKEOFF))
    sc.add_argument("work", type=Path)
    sc.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    pu = commands.add_parser("publish", help="register and publish what `scan` laid out")
    pu.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    ids = None if getattr(args, "only", None) is None else [int(v) for v in args.only.split(",")]
    if args.command == "fetch":
        uris = fetch(args.url, args.directory, instances=args.instances, only=ids)
        print(f"{len(uris)} tiles -> {args.directory}")
    elif args.command == "build":
        names = [n for n in args.variants.split(",") if n]
        unknown = [n for n in names if n not in VARIANTS_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown variants {unknown}; one of {sorted(VARIANTS_BY_NAME)}")
        if args.flat and len(names) != 1:
            raise SystemExit("--flat writes one variant: name it with --variants")
        build_variants(
            args.directory,
            names,
            only=ids,
            whole=args.whole,
            wide=not args.narrow,
            instances=args.instances,
            materials=args.materials,
            out=args.out,
            flat=args.flat,
        )
    elif args.command == "scan":
        names = [n for n in args.variants.split(",") if n]
        unknown = [n for n in names if n not in VARIANTS_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown variants {unknown}; one of {sorted(VARIANTS_BY_NAME)}")
        out = prepare_scan(args.name, args.work, names)
        print(f"{args.name}: laid out in {out}")
    elif args.command == "publish":
        publish(args.out)
    else:
        tileset = json.loads(args.tileset.read_text(encoding="utf-8"))
        entries = json.loads(args.variants.read_text(encoding="utf-8"))
        extras = tileset.get("root", {}).get("extras") or {}
        json.dump(merged_variants(extras, entries), sys.stdout, indent=1)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
