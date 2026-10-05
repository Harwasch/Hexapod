"""Publish inferred layers as bake-off variants beside a scan's tiles: `variants/fill/<name>/`,
declared in the measured root's `extras.variants.fill` (bake-off conventions).

Today's layers stay what they are (`extras.inferredLayers`); a variant is another layer the
viewer offers on the same scan. A variant entry:

    {"name": "vace-1-3b", "label": "Wan2.1-VACE 1.3B", "about": "...",
     "inferredLayers": [{"uri": "variants/fill/vace-1-3b/tileset.json", "evidence": {...}}]}

The API replaces a root extras key whole (apps/api app/services/sidecars.py `merge_extras`),
so `extras.variants` is sent whole: the asset's *current* variants with this run's entries
added or put in place of the ones with their names -- never another system's entries, never
another variant. `refresh` re-reads the current tileset just before the attach, so a variant
another bake-off published in between is kept.

    publish_variants.py build --scan spool --fill <artifact dir> --out <dir> \\
        --jobs gen-spool-wan2-1-vace-1-3b,gen-spool-inpaint-lama
    publish_variants.py refresh <dir>/spool      # just before attach_sidecars.py attach

Standard library and `attach_sidecars` only (a workflow runs it with the captures project).
"""

from __future__ import annotations

import argparse
import json
import struct
import tarfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import attach_sidecars

SYSTEM = "fill"
#: Each generator's layer folder (`generative_fill.slug` of its filler name) and the variant
#: it is published as.
FILL_VARIANTS: dict[str, dict[str, str]] = {
    "wan2-1-vace-1-3b": {
        "name": "vace-1-3b",
        "label": "Wan2.1-VACE 1.3B",
        "about": (
            "A video model walks a camera from a real photo towards what the scan missed, is "
            "given every pixel the scan knows and paints only the rest; the views are lifted "
            "into gaussians and anything a real camera saw through is removed."
        ),
    },
    "wan2-2-ti2v-5b": {
        "name": "wan22-5b",
        "label": "Wan 2.2 TI2V-5B",
        "about": (
            "The same camera walk with Wan 2.2: the known pixels are held fixed while the "
            "model draws the unknown ones; lifted into gaussians and carved by real sight lines."
        ),
    },
    "cosmos-predict2-2b": {
        "name": "cosmos-p2-2b",
        "label": "Cosmos-Predict2 2B",
        "about": (
            "The same camera walk with NVIDIA's Cosmos world model, started from the real "
            "photo and kept to the known pixels at every step; lifted and carved the same way."
        ),
    },
    "inpaint-lama": {
        "name": "lama-baseline",
        "label": "Per-view LaMa (baseline)",
        "about": (
            "Today's approach for holes: each view painted on its own by an image inpainter, "
            "lifted and carved the same way. The baseline the video models are judged against."
        ),
    },
}
USER_AGENT = "curl/8.5.0 (hexapod-publish-variants)"


def register(variants: Mapping[str, Any] | None, system: str, entry: Mapping[str, Any]) -> dict:
    """`variants` with `entry` added to `variants[system]`, or put in place of the entry with
    its `name`; every other system and every other variant kept, in order."""
    out = {k: list(v) if isinstance(v, list) else v for k, v in dict(variants or {}).items()}
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("a variant needs a name")
    current = [v for v in out.get(system, []) if isinstance(v, dict)]
    if any(v.get("name") == name for v in current):
        current = [dict(entry) if v.get("name") == name else v for v in current]
    else:
        current.append(dict(entry))
    out[system] = current
    return out


def _check_layer(folder: Path) -> dict[str, Any]:
    """The layer's own tileset, checked whole (as publish-fill.yml checks today's layers):
    every tile a binary glTF that is all there, every sidecar present, nothing unnamed, and
    `extras.evidence` saying `inferred`. Its evidence."""
    import rig_tiles

    document = json.loads((folder / "tileset.json").read_text(encoding="utf-8"))
    extras = document["root"].get("extras", {})
    evidence = extras.get("evidence") or {}
    if evidence.get("kind") != "inferred":
        raise SystemExit(f"{folder}: the layer's root carries no inferred evidence")
    uris = rig_tiles.tile_uris(document)
    if not uris:
        raise SystemExit(f"{folder}: the layer has no tiles")
    for uri in uris:
        path = folder / uri
        if "/" in uri or ".." in uri or not path.is_file():
            raise SystemExit(f"{folder}: tile {uri} is missing")
        data = path.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", data)
        if magic != b"glTF" or version != 2 or length != len(data):
            raise SystemExit(f"{folder}: tile {uri} is not a whole binary glTF")
    sidecars = [v["uri"] for v in extras.values() if isinstance(v, dict) and "uri" in v]
    for uri in sidecars:
        if not (folder / uri).is_file():
            raise SystemExit(f"{folder}: sidecar {uri} is missing")
    unnamed = sorted({p.name for p in folder.iterdir()} - {"tileset.json", *uris, *sidecars})
    if unnamed:
        raise SystemExit(f"{folder}: files the layer does not name: {unnamed}")
    return {
        "evidence": evidence,
        "transform": document["root"].get("transform"),
        "tiles": len(uris),
    }


def _extract(archive: Path, dest: Path) -> None:
    """`inferred.tar.gz` (one folder, `inferred/`, of plain files) into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if member.isdir():
                continue
            if not member.isfile() or len(parts) != 2 or parts[0] != "inferred":
                raise SystemExit(f"unexpected {member.name!r} in {archive}")
            source = tar.extractfile(member)
            assert source is not None
            (dest / parts[1]).write_bytes(source.read())


def entries_path(out: Path) -> Path:
    return out.parent / f"{out.name}.variant-entries.json"


def variant_for(job: str, scan: str) -> tuple[str, dict[str, str]]:
    """The variant a fill.yml job folder (`<kind>-<scan>-<layer slug>`) publishes as."""
    for kind in ("gen", "holdout"):
        prefix = f"{kind}-{scan}-"
        if job.startswith(prefix):
            slug = job[len(prefix) :]
            if kind == "holdout":
                raise SystemExit(f"{job}: a held-out run is a check, not a layer to publish")
            if slug not in FILL_VARIANTS:
                raise SystemExit(f"{job}: no variant for layer {slug!r} ({sorted(FILL_VARIANTS)})")
            return slug, FILL_VARIANTS[slug]
    raise SystemExit(f"{job} is not a gen-{scan}-<layer> folder")


def build(
    scan: str,
    fill: Path,
    out: Path,
    jobs: Sequence[str],
    *,
    asset: Mapping[str, str],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Every job's layer under `out/variants/fill/<name>/`, checked against the scan's
    `current` tileset (same frame), and `attach.json` with the merged variants. Returns the
    manifest."""
    entries = []
    for job in jobs:
        _, meta = variant_for(job, scan)
        dest = out / "variants" / SYSTEM / meta["name"]
        _extract(fill / job / "inferred.tar.gz", dest)
        checked = _check_layer(dest)
        if checked["transform"] != current["root"]["transform"]:
            raise SystemExit(f"{job}: the layer is not in the scan's current tileset's frame")
        entries.append(
            {
                "name": meta["name"],
                "label": meta["label"],
                "about": meta["about"],
                "inferredLayers": [
                    {
                        "uri": f"variants/{SYSTEM}/{meta['name']}/tileset.json",
                        "evidence": checked["evidence"],
                    }
                ],
            }
        )
    # This run's entries, for `refresh`: beside the folder, not in it (it is not a sidecar).
    entries_path(out).write_text(json.dumps(entries, indent=1), encoding="utf-8")
    variants = current["root"].get("extras", {}).get("variants")
    for entry in entries:
        variants = register(variants, SYSTEM, entry)
    return attach_sidecars.write_manifest(
        out, asset_id=asset["assetId"], based_on=asset["url"], extras={"variants": variants}
    )


def refresh(out: Path, *, asset: Mapping[str, str], current: Mapping[str, Any]) -> dict:
    """`attach.json` re-merged onto the scan's tileset as it is now (another variant may have
    been published since `build`), and based on its current URL."""
    manifest = attach_sidecars.read_manifest(out)
    entries = json.loads(entries_path(out).read_text(encoding="utf-8"))
    variants = current["root"].get("extras", {}).get("variants")
    for entry in entries:
        variants = register(variants, SYSTEM, entry)
    manifest["extras"] = {"variants": variants}
    manifest["basedOn"] = asset["url"]
    (out / attach_sidecars.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def _resolve(scan: str, legacy: str | None, get: Callable[[str], Any]) -> tuple[dict, dict]:
    asset = attach_sidecars.resolve_scan(scan, legacy_url=legacy)
    return asset, get(asset["url"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--scan", required=True)
    b.add_argument("--legacy-url", help="the scan's run URL, to find its asset by")
    b.add_argument("--fill", type=Path, required=True, help="the fill.yml artifact's folder")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--jobs", required=True, help="comma-separated gen-<scan>-<layer> folders")
    r = sub.add_parser("refresh")
    r.add_argument("dir", type=Path, help="the folder build wrote (holding attach.json)")
    args = parser.parse_args(argv)
    if args.command == "build":
        asset, current = _resolve(args.scan, args.legacy_url, fetch_json)
        jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
        manifest = build(args.scan, args.fill, args.out, jobs, asset=asset, current=current)
        fill = manifest["extras"]["variants"].get(SYSTEM, [])
        print(f"## Fill variants: {args.scan}\n")
        print(f"- asset `{asset['assetId']}`: {asset['url']}")
        for entry in fill:
            ev = entry.get("inferredLayers", [{}])[0].get("evidence", {})
            print(
                f"- `{entry['name']}` ({entry.get('label')}): {ev.get('filler')}, "
                f"{ev.get('views')} views, {ev.get('gaussians')} gaussians, "
                f"mean confidence {ev.get('meanConfidence')}"
            )
        print(f"- other systems kept: {sorted(set(manifest['extras']['variants']) - {SYSTEM})}")
    else:
        # The asset build resolved, at its current URL (an attach since may have moved it).
        asset = attach_sidecars.resolve_asset(attach_sidecars.read_manifest(args.dir)["assetId"])
        manifest = refresh(args.dir, asset=asset, current=fetch_json(asset["url"]))
        print(json.dumps({"basedOn": manifest["basedOn"], "variants": manifest["extras"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
