"""Fetch an uploaded capture's splat and its published tiles, for the scene step.

``scene_plants.py`` needs two things a capture already has once ``splat-ingest`` (or
``photo-reconstruct``) has run: its ``canonical.ply`` -- metric, east/north/up -- and the
level-of-detail tileset the ``package`` stage packed from it, so the rig it writes is stamped
with the checksums of *those* tiles. The API serves both, and this fetches them:

* ``GET /api/v1/captures/{id}`` -- the site it registered, and how its scale was resolved;
* ``GET /api/v1/captures/{id}/splat.ply`` -- a redirect to a signed URL of ``canonical.ply``;
* ``GET /api/v1/assets?siteId=...`` -- the site's gaussian-splat asset, whose ``source.url``
  is its ``tileset.json``; every tile it names is fetched beside it.

It refuses a capture whose scale was never resolved (``scaleSource: unresolved``) unless told
the splat is metric anyway (``--assume-metric``): every threshold of the scene step is in
metres -- the FAO's 0.5 m and 5 m, a crown width from a height -- and a splat in unknown units
would be classified by numbers that mean nothing.

Usage::

    uv run python fetch_capture.py <capture-id> work/ [--api https://twin-api.fly.dev]
    uv run python scene_plants.py work/canonical.ply --tiles work/tiles --out work/living

Nothing here needs a credential: the capture, its splat and its published tiles are public
reads. Publishing the result beside the tiles does (``.github/workflows/living-plants.yml``).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_API = "https://twin-api.fly.dev"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
#: Scale sources that give metres (apps/api app/models/enums.py ``ScaleSource``).
#: ``camera-height-estimate`` is deliberately absent: it is a ±20%-or-worse guess from how
#: high a phone is usually held, and the scene step's thresholds would classify by it.
METRIC_SCALE_SOURCES = ("arkit", "exif-gps", "manual")
#: Cloudflare (the public bucket's r2.dev host) answers 403 to Python's default
#: ``Python-urllib/3.x`` agent, so every request names this tool instead.
USER_AGENT = "hexapod-fetch-capture/1"


def _open(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _get_json(url: str) -> object:
    with _open(url, 60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(url, 600) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    return path.stat().st_size


def tile_uris(tileset: dict) -> list[str]:
    """Every content uri of a tileset, depth first (as rig_tiles.tile_uris)."""
    uris: list[str] = []

    def walk(tile: dict) -> None:
        uri = tile.get("content", {}).get("uri")
        if uri:
            uris.append(uri)
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    return uris


def check_scale(capture: dict, assume_metric: bool) -> str:
    """The capture's scale source, or a refusal when it is not known to be metres."""
    source = str(capture.get("scaleSource") or "unresolved")
    if source not in METRIC_SCALE_SOURCES and not assume_metric:
        raise SystemExit(
            f"the capture's scale source is {source!r}: its splat is not known to be in metres, "
            "and the scene step's thresholds are. Pass --assume-metric if you know it is."
        )
    return source


def splat_asset(assets: list, site_id: str) -> dict:
    """The site's gaussian-splat asset: the one with a tileset to bind."""
    for asset in assets:
        if asset.get("representation") == "gaussian-splat" and asset.get("siteId") == site_id:
            return asset
    raise SystemExit(f"site {site_id} has no gaussian-splat asset: has the capture finished?")


def fetch(
    capture_id: str, into: Path, *, api: str = DEFAULT_API, assume_metric: bool = False
) -> dict:
    if not UUID.match(capture_id):
        raise SystemExit(f"{capture_id!r} is not a capture id")
    base = api.rstrip("/")
    capture = _get_json(f"{base}/api/v1/captures/{capture_id}")
    if not isinstance(capture, dict):
        raise SystemExit("the API did not return a capture")
    scale = check_scale(capture, assume_metric)
    site_id = capture.get("siteId")
    if not site_id:
        raise SystemExit("the capture has no site yet: it has not finished a run")
    assets = _get_json(f"{base}/api/v1/assets?siteId={urllib.parse.quote(str(site_id))}")
    asset = splat_asset(assets if isinstance(assets, list) else [], str(site_id))
    tileset_url = str(asset.get("source", {}).get("url") or "")
    if not tileset_url.startswith("https://") and not tileset_url.startswith("http://"):
        raise SystemExit(f"the splat asset's tileset is not a URL: {tileset_url!r}")
    into.mkdir(parents=True, exist_ok=True)
    ply_bytes = _download(f"{base}/api/v1/captures/{capture_id}/splat.ply", into / "canonical.ply")
    tiles = into / "tiles"
    _download(tileset_url, tiles / "tileset.json")
    tileset = json.loads((tiles / "tileset.json").read_text(encoding="utf-8"))
    tile_bytes = 0
    for uri in tile_uris(tileset):
        if "/" in uri or ".." in uri:
            raise SystemExit(f"tile uri {uri!r} is not a file beside tileset.json")
        tile_bytes += _download(urllib.parse.urljoin(tileset_url, uri), tiles / uri)
    summary = {
        "capture": capture_id,
        "slug": capture.get("slug"),
        "siteId": site_id,
        "scaleSource": scale,
        "assetId": asset.get("id"),
        "tilesetUrl": tileset_url,
        "plyBytes": ply_bytes,
        "tiles": len(tile_uris(tileset)),
        "tileBytes": tile_bytes,
    }
    (into / "capture.json").write_text(json.dumps({"capture": capture, "asset": asset}, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("capture_id")
    parser.add_argument("into", type=Path)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--assume-metric", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            fetch(args.capture_id, args.into, api=args.api, assume_metric=args.assume_metric),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
