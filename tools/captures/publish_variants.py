"""Publish inferred layers as bake-off variants beside a scan's tiles: `variants/fill/<name>/`,
declared in the measured root's `extras.variants.fill` (bake-off conventions); and withdraw
variants that are no longer wanted.

Today's layers stay what they are (`extras.inferredLayers`); a variant is another layer the
viewer offers on the same scan. A variant entry:

    {"name": "anchor-refs", "label": "...", "about": "...", "look": "...",
     "inferredLayers": [{"uri": "variants/fill/anchor-refs/tileset.json", "evidence": {...}}]}

**Only through the shared helpers.** The request this writes (`attach.json`) never carries
an `extras.variants` value: it names the entries to register (`register`) or the names to
withdraw (`withdraw`), and `attach_sidecars.attach` merges them, at the request, into the
tileset as it is then (`with_variant` / `without_variant`), so every other system's entries
and every other variant -- including one attached by another bake-off meanwhile -- stay.
Before writing, `preflight` merges the same changes into the tileset as it is now and refuses
any result that would drop another system's entry or leave no variants at all.

    publish_variants.py build --scan spool --fill <artifact dir> --out <dir> \\
        --jobs anchor-spool-anchor-refs,anchor-spool-anchor-norefs
    publish_variants.py withdraw --scan spool --out <dir> --names wan22-5b,cosmos-p2-2b

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
#: Where to look, per scan, in a variant's `look` line.
LOOK_WHERE = {
    "spool": "orbit to look down on the spool's top",
    "pumpkin": "orbit low, under the pumpkins, where they meet the straw",
}
LOOK = "Set Inferred to Highlight; {where}. Purple is generated."
#: Each layer folder (round 1: `generative_fill.slug` of its filler name; round 2:
#: `anchor_fill.LAYERS`) and the variant it is published as.
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
    "anchor-refs": {
        "name": "anchor-refs",
        "label": "Anchors + photos, then view by view (Qwen-Image-Edit 2511)",
        "about": (
            "Finds what the photos saw badly or not at all. Where they saw a surface from one "
            "side only (the spool's top), the best photos are first warped onto that surface "
            "as opaque gaussians, so it is solid from every side. Then a few anchor views are "
            "filled by an image editor given the render and two real photos of that spot, and "
            "each other view gets only what is still missing, placed in 3D at depth that meets "
            "the scan."
        ),
    },
    "anchor-norefs": {
        "name": "anchor-norefs",
        "label": "Anchors without photos, then view by view (Qwen-Image-Edit 2511)",
        "about": (
            "The same, but no real photo anywhere: no surface is warped from the photos, and "
            "the editor sees only the render. The test of what the photos add."
        ),
    },
    "anchor-vace": {
        "name": "anchor-vace",
        "label": "Anchors + photos, then all views at once (Wan2.1-VACE 14B)",
        "about": (
            "The same photo-guided anchors, then a video model fills every other view in one "
            "pass, all the views as frames of one clip with the real photos and anchors, so "
            "they agree with each other."
        ),
    },
    "anchor-splash": {
        "name": "splash-refs",
        "label": "Splat-repair anchors + photos, then view by view (Gaussian-Splash LoRA)",
        "about": (
            "The photo arm with other anchors: an editor adapter trained to repair splat "
            "renders is given the render with everything the scan does not know blacked "
            "out and a real photo of the spot, and repaints the blanks. Only seeds whose "
            "known pixels and camera still match the render are kept."
        ),
        "look": (
            "Set Inferred to Highlight; orbit to look down on the spool's top and low under "
            "its flange. Purple is generated."
        ),
    },
    "anchor-splash-asis": {
        "name": "splash-asis",
        "label": "Splat-repair anchors as designed, aligned, + photos (Gaussian-Splash LoRA)",
        "about": (
            "The splat-repair adapter run as its own workflow runs it: the plain render and "
            "the whole real photo, no mask. It redraws the whole frame, so each output whose "
            "camera is within 5 degrees is aligned back onto the render (homography, then "
            "dense flow) and only the holes are taken from it. Then view by view with photos."
        ),
        "look": (
            "Set Inferred to Highlight; orbit to look down on the spool's top, then low "
            "round its rim. Purple is generated."
        ),
    },
    "anchor-splash-locked": {
        "name": "splash-locked",
        "label": "Splat-repair anchors, camera locked, + photos (Gaussian-Splash LoRA)",
        "about": (
            "The splat-repair adapter given the plain render, but every pixel the scan knows "
            "is locked to the render at each denoising step, so the camera cannot move and "
            "only the holes and see-through parts are made, from a real photo of the spot. "
            "Then view by view with photos."
        ),
        "look": (
            "Set Inferred to Highlight; orbit to look down on the spool's top, then low "
            "round its rim. Purple is generated."
        ),
    },
    "anchor-splash-region": {
        "name": "splash-region",
        "label": "Splat-repair anchors repainting the whole top, camera locked, + photos",
        "about": (
            "The splat-repair adapter given the plain render, with the whole soft top of the "
            "spool to repaint at once and everything else locked to the render, so the camera "
            "holds and the top is painted coherently from a real photo of it. Seeds that move "
            "the camera, change the scan around the top or change the wood's tone are dropped. "
            "Then view by view with photos."
        ),
        "look": (
            "Set Inferred to Highlight; look down on the spool's top: sharper wood grain, bolts "
            "and holes painted by the adapter where the scan was soft. Purple is generated."
        ),
    },
    "anchor-splash-footprint": {
        "name": "splash-footprint",
        "label": "Splash LoRA top (detail invented)",
        "about": (
            "The top's wood detail is painted by an image model (a splat-repair LoRA), not "
            "measured. Only the top face's own footprint was repainted, with its outline and "
            "everything outside locked to the scan, so the camera and the top's shape hold. "
            "But the positions of its bolts, planks and holes are invented and do not match "
            "the real spool: against held-out real photos it scores worse than the photo-only "
            "fill (-0.63 dB, +0.027 LPIPS). Published for judging by eye."
        ),
        "look": (
            "Look down on the top: sharp wood painted by an image model. Bolt and plank "
            "positions are made up and differ from the real spool; compare with "
            "'Anchors + photos'."
        ),
    },
    "anchor-splash-voids": {
        "name": "splash-voids",
        "label": "Splat-repair anchors on the true holes, + photos (Gaussian-Splash LoRA)",
        "about": (
            "The splat-repair adapter shown the render as it was trained on: only the true "
            "holes blacked out as smooth shapes, the smeared and see-through parts left for it "
            "to repair. Everything away from the holes is locked to the render, so the camera "
            "holds. Then view by view with photos."
        ),
        "look": (
            "Set Inferred to Highlight; orbit to look down on the spool's top, then low "
            "round its rim. Purple is generated."
        ),
    },
    "anchor-refs-alpha": {
        "name": "anchor-refs-alpha",
        "label": "Anchors + photos, distilled to be solid (coverage term)",
        "about": (
            "The photo arm, with its gaussians also trained to cover the rebuilt surface fully "
            "from 17 directions around it, so it is not see-through from any side."
        ),
    },
    "anchor-refs-full": {
        "name": "anchor-refs-solid",
        "label": "Anchors + photos, distilled to be solid (all terms)",
        "about": (
            "The photo arm, trained to cover the rebuilt surface from 17 directions, to sit at "
            "its depth and lie along it, with no needle-shaped gaussians, and with random "
            "gaussians dropped while training so no single one carries a pixel."
        ),
    },
    "anchor-norefs-alpha": {
        "name": "anchor-norefs-alpha",
        "label": "Anchors without photos, distilled to be solid (coverage term)",
        "about": "The no-photo control, trained to cover the fitted surface from 17 directions.",
    },
    "anchor-norefs-full": {
        "name": "anchor-norefs-solid",
        "label": "Anchors without photos, distilled to be solid (all terms)",
        "about": (
            "The no-photo control, with all the solidity terms: coverage, depth, lying along "
            "the surface, no needles, random dropout."
        ),
    },
    "anchor-refs+anchor-shape": {
        "name": "refs-shape",
        "label": "Anchors + photos, and the shape under the flange",
        "about": (
            "The photo arm's top, and under the top flange, which no camera saw, the drum and "
            "the flange's underside continued from the measured surfaces: never where a camera "
            "saw through, coloured from the nearest wood in shade. Two layers."
        ),
        "look": (
            "Set Inferred to Highlight; orbit low, under the spool's top flange. "
            "Purple is generated."
        ),
    },
}
#: Where each layer of a variant of several goes, under the variant's folder.
LAYER_DIRS = {"anchor-refs": "top", "anchor-shape": "shape"}
#: Job kinds whose folders hold layers to publish; the others are checks.
PUBLISHED_KINDS = ("gen", "anchor")
CHECK_KINDS = ("holdout", "leaveout")
USER_AGENT = "curl/8.5.0 (hexapod-publish-variants)"


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
    supersedes = extras.get("supersedes")
    return {
        "evidence": evidence,
        "transform": document["root"].get("transform"),
        "tiles": len(uris),
        "supersedes": supersedes.get("uri") if isinstance(supersedes, dict) else None,
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


def variant_for(job: str, scan: str) -> tuple[str, dict[str, str]]:
    """The variant a fill.yml job folder (`<kind>-<scan>-<layer>`) publishes as; several
    folders joined by `+` publish as one variant of several layers."""
    slugs = [variant_for_layer(part, scan) for part in job.split("+")]
    key = "+".join(slugs)
    if key not in FILL_VARIANTS:
        raise SystemExit(f"{job}: no variant for {key!r} ({sorted(FILL_VARIANTS)})")
    return key, FILL_VARIANTS[key]


def variant_for_layer(job: str, scan: str) -> str:
    """The layer of one job folder (`<kind>-<scan>-<layer>`): a variant's, or one layer of a
    variant of several (`LAYER_DIRS`)."""
    for kind in (*PUBLISHED_KINDS, *CHECK_KINDS):
        prefix = f"{kind}-{scan}-"
        if job.startswith(prefix):
            slug = job[len(prefix) :]
            if kind in CHECK_KINDS:
                raise SystemExit(f"{job}: a held-out run is a check, not a layer to publish")
            known = {*FILL_VARIANTS, *LAYER_DIRS}
            if slug not in known:
                raise SystemExit(f"{job}: no variant for layer {slug!r} ({sorted(known)})")
            return slug
    kinds = "|".join(PUBLISHED_KINDS)
    raise SystemExit(f"{job} is not a ({kinds})-{scan}-<layer> folder")


def entry_for(
    meta: Mapping[str, str],
    scan: str,
    evidence: Mapping[str, Any],
    supersedes: str | None = None,
    layers: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """A fill variant's entry. `supersedes`: the layer's own `supersedes.json` (relative to
    the layer), the measured splats its rebuilt surface replaces: published as
    `variants/fill/<name>/<file>`, so a viewer that reads it hides them while the variant is
    shown (older viewers ignore the field: `lib/variants.ts` reads only what it knows)."""
    entry: dict[str, Any] = {"name": meta["name"], "label": meta["label"], "about": meta["about"]}
    if meta.get("look"):
        entry["look"] = meta["look"]
    elif scan in LOOK_WHERE and meta["name"].startswith("anchor-"):
        entry["look"] = LOOK.format(where=LOOK_WHERE[scan])
    base = f"variants/{SYSTEM}/{meta['name']}"
    entry["inferredLayers"] = (
        [{"uri": f"{base}/{sub}/tileset.json", "evidence": dict(ev)} for sub, ev in layers]
        if layers
        else [{"uri": f"{base}/tileset.json", "evidence": dict(evidence)}]
    )
    if supersedes:
        entry["supersedes"] = f"variants/{SYSTEM}/{meta['name']}/{supersedes}"
    return entry


def preflight(
    current: Any, changes: Sequence[tuple[str, str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The merge `attach` will make, made now on the tileset as it is (for review), and a
    summary. Refuses a result with no variants at all (the attach would then send none: never
    a `variants: null`), one that drops or changes another system's entries, or one that
    drops a fill variant this run did not name."""
    before = {k: list(v) for k, v in dict(current or {}).items() if isinstance(v, list)}
    after = attach_sidecars.changed_variants(before, changes)
    if after is None:
        raise SystemExit("the merge would leave no variants at all: refused")
    for system, entries in before.items():
        if system == SYSTEM:
            continue
        if after.get(system) != entries:
            raise SystemExit(f"the merge would change the {system!r} variants: refused")
    named = {str(v["name"] if isinstance(v, dict) else v) for _, s, v in changes if s == SYSTEM}
    kept = {e.get("name") for e in after.get(SYSTEM, []) if isinstance(e, dict)}
    lost = {e.get("name") for e in before.get(SYSTEM, []) if isinstance(e, dict)} - kept - named
    if lost:
        raise SystemExit(f"the merge would drop fill variants {sorted(lost)}: refused")
    summary = {
        system: [e.get("name") for e in entries if isinstance(e, dict)]
        for system, entries in after.items()
    }
    return after, summary


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
    `current` tileset (same frame), and `attach.json` registering their entries (merged at
    the request by the shared attach). Returns the manifest."""
    register = []
    for job in jobs:
        _, meta = variant_for(job, scan)
        dest = out / "variants" / SYSTEM / meta["name"]
        parts = job.split("+")
        layers: list[tuple[str, Any]] = []
        supersedes = None
        for part in parts:
            sub = LAYER_DIRS[variant_for_layer(part, scan)] if len(parts) > 1 else ""
            folder = dest / sub if sub else dest
            _extract(fill / part / "inferred.tar.gz", folder)
            checked = _check_layer(folder)
            if checked["transform"] != current["root"]["transform"]:
                raise SystemExit(f"{part}: the layer is not in the scan's current tileset's frame")
            layers.append((sub, checked["evidence"]))
            if checked.get("supersedes") and supersedes is None:
                supersedes = f"{sub}/{checked['supersedes']}" if sub else checked["supersedes"]
        entry = entry_for(meta, scan, layers[0][1], supersedes, layers if len(parts) > 1 else None)
        register.append((SYSTEM, entry))
    changes = [("with", s, e) for s, e in register]
    _, summary = preflight(current["root"].get("extras", {}).get("variants"), changes)
    manifest = attach_sidecars.write_manifest(
        out, asset_id=asset["assetId"], based_on=asset["url"], extras={}, register=register
    )
    manifest["preflight"] = summary
    return manifest


def withdraw(
    scan: str,
    out: Path,
    names: Sequence[str],
    *,
    asset: Mapping[str, str],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """`attach.json` (and no files) withdrawing the fill variants `names`: the attach takes
    them off the tileset as it is at the request (`without_variant`); every other entry
    stays. A name the tileset does not declare is refused (nothing to withdraw)."""
    declared = {
        e.get("name")
        for e in (current["root"].get("extras", {}).get("variants") or {}).get(SYSTEM, [])
        if isinstance(e, dict)
    }
    missing = sorted(set(names) - declared)
    if missing:
        raise SystemExit(f"{scan} declares no fill variants {missing}: nothing to withdraw")
    changes = [("without", SYSTEM, n) for n in names]
    _, summary = preflight(current["root"].get("extras", {}).get("variants"), changes)
    out.mkdir(parents=True, exist_ok=True)
    manifest = attach_sidecars.write_manifest(
        out,
        asset_id=asset["assetId"],
        based_on=asset["url"],
        extras={},
        withdraw=[(SYSTEM, n) for n in names],
    )
    manifest["preflight"] = summary
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
    b = sub.add_parser("build", help="lay out and register layers as fill variants")
    b.add_argument("--scan", required=True)
    b.add_argument("--legacy-url", help="the scan's run URL, to find its asset by")
    b.add_argument("--fill", type=Path, required=True, help="the fill.yml artifact's folder")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--jobs", required=True, help="comma-separated <kind>-<scan>-<layer> folders")
    w = sub.add_parser("withdraw", help="withdraw fill variants by name")
    w.add_argument("--scan", required=True)
    w.add_argument("--legacy-url")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--names", required=True, help="comma-separated variant names")
    args = parser.parse_args(argv)
    asset, current = _resolve(args.scan, args.legacy_url, fetch_json)
    if args.command == "build":
        jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
        manifest = build(args.scan, args.fill, args.out, jobs, asset=asset, current=current)
        print(f"## Fill variants: {args.scan}\n")
        print(f"- asset `{asset['assetId']}`: {asset['url']}")
        for item in manifest["register"]:
            entry = item["entry"]
            ev = entry["inferredLayers"][0]["evidence"]
            print(
                f"- register `{entry['name']}` ({entry['label']}): {ev.get('filler')}, "
                f"{ev.get('views')} views, {ev.get('gaussians')} gaussians, "
                f"mean confidence {ev.get('meanConfidence')}"
            )
    else:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        manifest = withdraw(args.scan, args.out, names, asset=asset, current=current)
        print(f"## Withdraw fill variants: {args.scan}\n")
        print(f"- asset `{asset['assetId']}`: {asset['url']}")
        print(f"- withdraw {names}")
    print(f"- after the merge (as of now): {json.dumps(manifest['preflight'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
