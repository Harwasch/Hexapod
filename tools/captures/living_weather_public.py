"""Read the public R2 catalog and audit all discoverable splat scans, serially.

Only public read URLs; artifacts live in the caller's cache/notes directories.
Minnetonka and synthetic-tree reuse existing bytes. No catalogue/API writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from living_weather_detail import check, leaf_uris

BUCKET = "https://pub-67ae49c6d86140a89f7ae818c1b02e99.r2.dev/"


def download(url: str, dest: Path, optional: bool = False) -> bool:
    if dest.exists():
        return True
    if not url.startswith(BUCKET):
        raise ValueError("Only the public scan bucket is allowed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(Request(url, headers={"User-Agent": "curl/8.5.0"}), timeout=120) as response:
            dest.write_bytes(response.read())
        return True
    except HTTPError as e:
        if optional and e.code == 404:
            return False
        raise


def run(cache: Path, out: Path, minnetonka: Path, synthetic: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    catalog = cache / "public-catalog.json"
    download(BUCKET + "catalog.json", catalog)
    results = []
    for site in json.loads(catalog.read_text())["sites"]:
        assets = [a for a in site.get("assets", []) if a.get("representation") == "gaussian-splat"]
        if not assets:
            results.append(
                {
                    "slug": site["slug"],
                    "name": site["name"],
                    "score": None,
                    "tier": "off",
                    "reason": "No public scan asset (withdrawn)",
                }
            )
        for asset in assets:
            url = asset["source"].get("url", "")
            if site["slug"] == "minnetonka-tree":
                source = minnetonka
            elif site["slug"] == "synthetic-tree":
                source = synthetic
            else:
                source = cache / "scans" / site["slug"]
                manifest = source / "splat/tileset.json"
                download(url, manifest)
                doc = json.loads(manifest.read_text())
                for uri in leaf_uris(doc):
                    if Path(uri).is_absolute() or ".." in Path(uri).parts:
                        raise ValueError("Unsafe tile URI")
                    download(urljoin(url, uri), source / "splat" / uri)
                instances = doc["root"].get("extras", {}).get("instances")
                if isinstance(instances, dict):
                    instances = instances.get("uri") or instances.get("url")
                if isinstance(instances, str):
                    download(
                        urljoin(url, instances), source / "splat/instances.json", optional=True
                    )
                # A fitted rig is optional; absent evidence leaves Tier A available.
                download(
                    urljoin(url, "../source/rig.json"), source / "source/rig.json", optional=True
                )
            result = check(source, ground_override=0 if site["slug"] == "minnetonka-tree" else None)
            result.update(
                {
                    "slug": site["slug"],
                    "name": site["name"],
                    "url": url,
                    "sourceHost": urlparse(url).hostname,
                }
            )
            results.append(result)
            (out / f"{site['slug']}-detail.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            (out / "public-scan-detail.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--minnetonka", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path)
    args = parser.parse_args()
    run(args.cache, args.out, args.minnetonka, args.synthetic)
