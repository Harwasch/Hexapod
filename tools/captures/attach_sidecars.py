"""Attach sidecars to a scan through the API: stage the files, then ask for a new generation.

Every workflow that publishes something beside a scan's tiles -- objects
(publish-instances.yml), an inferred fill (publish-fill.yml), a collision grid
(collision-backfill.yml), the streamed LOD (streamed-lod-backfill.yml), a plant rig
(living-plants.yml) -- does it through this one script, which does it the one way there is
(docs/DEPLOYMENT.md, "Sidecars: one publisher"; docs/SCENE_OBJECTS.md section 8):

1. the files are uploaded to the **private** bucket under
   ``staging/assets/<asset id>/<token>/``, laid out exactly as they should sit beside
   ``tileset.json`` (the token is the workflow run and its attempt);
2. ``POST /api/v1/assets/<asset id>/sidecars`` asks the API to cut a new generation: the
   asset's current tiles and every sidecar they already have, copied server side, the staged
   files beside them, ``tileset.json`` with the merged root extras written last, and the
   asset repointed under a row lock. Nothing here writes the public bucket or a
   ``tileset.json``.

A workflow's build job writes the request beside the files as ``attach.json`` (``manifest``)
-- the asset, the tileset URL the files were computed against, the files, the root extras --
so the review artifact says exactly what will be sent; its publish job runs ``attach`` on
that directory. Subcommands::

    attach_sidecars.py resolve --asset <uuid>                   # the asset's CURRENT url
    attach_sidecars.py resolve --scan camp --legacy-url <url>   # a scan by name
    attach_sidecars.py manifest DIR --asset <uuid> --based-on <url> [--extras JSON]
                                    [--rig-url rig.json]
    attach_sidecars.py attach DIR                               # stage, POST, report

``resolve`` prints ``{"assetId", "url", "name"}``. ``attach`` exits 0 when the asset moved
to the new generation, 3 on a 409 whose ``code`` is ``tiles_changed`` (the tiles changed under
the run: compute again on the asset's current tiles), and 1 on anything else -- with the
API's own detail -- after retrying a 409 whose ``code`` is ``busy``.

Environment: ``TWIN_API_URL`` (the API's origin; ``--api``), and for ``attach``
``API_WRITE_TOKEN`` and the private bucket -- ``OBJECT_STORAGE_ENDPOINT_URL``,
``OBJECT_STORAGE_ACCESS_KEY``, ``OBJECT_STORAGE_SECRET_KEY``, ``OBJECT_STORAGE_BUCKET`` --
plus ``GITHUB_RUN_ID`` and ``GITHUB_RUN_ATTEMPT`` for the staging token where a workflow
runs it. ``resolve --scan`` reads the scan-to-asset map from ``SCAN_ASSET_IDS`` (JSON,
``{"camp": "<asset id>", ...}``).

Standard library, and boto3 for the upload (imported only by ``attach``, which a workflow
runs with ``uv run --with boto3``): the tests drive it with a stub bucket and a stub API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_API = "https://twin-api.fly.dev"
USER_AGENT = "hexapod-attach-sidecars/1"
MANIFEST = "attach.json"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# What the API accepts, checked here first so a mistake fails before anything is uploaded
# (apps/api/app/services/sidecars.py; tests/test_attach_sidecars.py reads them from there).
STAGING_ROOT = "staging/assets"
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_DEPTH = 6
MAX_PATH = 512
SIDECAR_TYPES: dict[str, str] = {
    "json": "application/json",
    "glb": "model/gltf-binary",
    "bin": "application/octet-stream",
    "emb": "application/octet-stream",
    "f32": "application/octet-stream",
    "u8": "application/octet-stream",
    "webp": "image/webp",
}
MAX_SIDECAR_FILES = 5_000
MAX_SIDECAR_BYTES = 1024**3
MAX_ATTACH_BYTES = 8 * 1024**3
#: A split rewrites the scan's tiles; it is a new tileset, not files beside one.
NOT_ATTACHABLE = ("objects/", "fills/")

#: An attach holds the asset for seconds per thousand tiles; the API's own lock wait is 60 s.
POST_TIMEOUT_S = 900
#: A 409 because another attach (or a worker's publish) holds the asset is retried.
BUSY_RETRIES = 4
BUSY_WAIT_S = 30.0
#: The API's machine-readable reasons for a 409, its Problem's `code`
#: (app/services/attach.py; tests/test_attach_sidecars.py reads them from there). Every
#: other 409 -- an asset that is not a run's 3D Tiles, a directory too big to copy -- is
#: `not_attachable`, and neither waiting nor computing again would change it.
BUSY = "busy"
TILES_CHANGED_CODE = "tiles_changed"
#: `attach`'s exit status for a 409 that means the tiles changed under the run.
TILES_CHANGED = 3


class AttachError(SystemExit):
    """A refusal with a message for the workflow log, and the exit status it ends with.

    A `SystemExit`, so a workflow's inline Python that calls `resolve_scan` ends with it
    unhandled; the message is printed when it is raised, because the interpreter prints
    nothing for an exit status.
    """

    def __init__(self, message: str, status: int = 1) -> None:
        print(f"attach_sidecars: {message}", file=sys.stderr)
        super().__init__(status)
        self.message = message

    def __str__(self) -> str:
        return self.message


# --- the API ---------------------------------------------------------------------------


def api_url(api: str | None) -> str:
    return (api or os.environ.get("TWIN_API_URL") or DEFAULT_API).rstrip("/")


def get_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise AttachError(f"GET {url}: {error.code} {_body(error)}") from error
    except urllib.error.URLError as error:
        raise AttachError(f"GET {url}: {error.reason}") from error


def resolve_asset(asset_id: str, *, api: str | None = None) -> dict[str, str]:
    """The asset and its **current** tileset URL: what sidecars are computed against now.

    Not a URL remembered from elsewhere (infra/modal/segment.py `SCANS`): once anything has
    been attached, or a run republished, the asset points at a generation of its own.
    """
    if not UUID.match(asset_id):
        raise AttachError(f"{asset_id!r} is not an asset id (a UUID)")
    asset = get_json(f"{api_url(api)}/api/v1/assets/{asset_id}")
    source = asset.get("source") if isinstance(asset, dict) else None
    url = source.get("url") if isinstance(source, dict) else None
    if not isinstance(source, dict) or source.get("type") != "3d-tiles-url" or not url:
        raise AttachError(f"asset {asset_id} is not a 3D Tiles URL, so it has no sidecars")
    return {"assetId": asset_id, "url": str(url), "name": str(asset.get("name", ""))}


def scan_assets(raw: str | None = None) -> dict[str, str]:
    """The scan-to-asset map: `raw`, else `SCAN_ASSET_IDS`, as a JSON object."""
    text = raw if raw is not None else os.environ.get("SCAN_ASSET_IDS", "")
    if not text.strip():
        return {}
    try:
        mapping = json.loads(text)
    except ValueError as error:
        raise AttachError(f"SCAN_ASSET_IDS is not JSON: {error}") from error
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise AttachError('SCAN_ASSET_IDS must be a JSON object, {"<scan>": "<asset id>"}')
    return mapping


def resolve_scan(
    scan: str,
    *,
    legacy_url: str | None = None,
    api: str | None = None,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A scan's asset by its short name (infra/modal/segment.py `SCANS`), and its current URL.

    The asset id comes from the map (`SCAN_ASSET_IDS`). Where the map has no entry, the API's
    asset list is searched for the one gaussian-splat asset whose tileset is the scan's run
    (`runs/<job>/...` in `legacy_url`, at the legacy prefix or in a generation cut from it);
    anything but exactly one is a refusal that says what to set.
    """
    found = (mapping if mapping is not None else scan_assets()).get(scan)
    if found:
        return {**resolve_asset(found, api=api), "scan": scan}
    job = re.search(r"/runs/([0-9a-f-]{36})/", legacy_url or "")
    hint = (
        f"set the repository variable SCAN_ASSET_IDS to a JSON object naming it, "
        f'{{"{scan}": "<asset id>"}} (GET {api_url(api)}/api/v1/assets lists them)'
    )
    if job is None:
        raise AttachError(f"no asset id for scan {scan!r}: {hint}")
    assets = get_json(f"{api_url(api)}/api/v1/assets")
    candidates = [
        asset
        for asset in (assets if isinstance(assets, list) else [])
        if isinstance(asset, dict)
        and asset.get("representation") == "gaussian-splat"
        and isinstance(asset.get("source"), dict)
        and asset["source"].get("type") == "3d-tiles-url"
        and f"/runs/{job[1]}/" in str(asset["source"].get("url", ""))
    ]
    if len(candidates) != 1:
        raise AttachError(
            f"no asset id for scan {scan!r}, and {len(candidates)} assets show its run "
            f"{job[1]}: {hint}"
        )
    print(
        f"attach_sidecars: scan {scan!r} is asset {candidates[0]['id']} (found by its run; "
        "SCAN_ASSET_IDS would make it explicit)",
        file=sys.stderr,
    )
    return {**resolve_asset(str(candidates[0]["id"]), api=api), "scan": scan}


# --- what may be staged -----------------------------------------------------------------


def check_name(rel: str) -> str:
    """`rel` if the API will take a staged file of that name, else `AttachError`."""
    parts = rel.split("/")
    if (
        not rel
        or len(rel) > MAX_PATH
        or len(parts) > MAX_DEPTH
        or not all(SEGMENT.fullmatch(part) for part in parts)
    ):
        raise AttachError(
            f"{rel!r} is not a plain path beside tileset.json (letters, digits, '.', '_' and "
            f"'-' in at most {MAX_DEPTH} segments, none starting with '.')"
        )
    extension = parts[-1].rsplit(".", 1)[-1].lower() if "." in parts[-1] else ""
    if extension not in SIDECAR_TYPES:
        raise AttachError(f"{rel!r}: a sidecar's extension must be one of {sorted(SIDECAR_TYPES)}")
    if rel == "tileset.json":
        raise AttachError("tileset.json is written by the API, from `extras`; do not stage it")
    if rel.startswith(NOT_ATTACHABLE):
        raise AttachError(
            f"{rel!r}: a split rewrites the scan's tiles and cannot be attached beside them"
        )
    return rel


def files_under(directory: Path) -> list[str]:
    """Every file under `directory` but the manifest, relative and sorted, each checked (none
    is an attach of `extras` alone)."""
    found = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() != MANIFEST
    )
    for rel in found:
        check_name(rel)
    if len(found) > MAX_SIDECAR_FILES:
        raise AttachError(f"{len(found)} files; one attach may hold {MAX_SIDECAR_FILES}")
    total = 0
    for rel in found:
        size = (directory / rel).stat().st_size
        if size > MAX_SIDECAR_BYTES:
            raise AttachError(f"{rel} is {size:,} bytes; a sidecar may be {MAX_SIDECAR_BYTES:,}")
        total += size
    if total > MAX_ATTACH_BYTES:
        raise AttachError(f"{total:,} bytes in all; one attach may hold {MAX_ATTACH_BYTES:,}")
    return found


def write_manifest(
    directory: Path,
    *,
    asset_id: str,
    based_on: str,
    extras: Mapping[str, Any] | None = None,
    rig_url: str | None = None,
) -> dict[str, Any]:
    """`attach.json` for the files under `directory`: the request `attach` will send."""
    if not UUID.match(asset_id):
        raise AttachError(f"{asset_id!r} is not an asset id (a UUID)")
    if not based_on.startswith(("https://", "http://")):
        raise AttachError(f"basedOn must be the tileset's URL, not {based_on!r}")
    manifest: dict[str, Any] = {
        "assetId": asset_id,
        "basedOn": based_on,
        "files": files_under(directory),
        "extras": dict(extras or {}),
    }
    if not manifest["files"] and not manifest["extras"] and rig_url is None:
        raise AttachError(f"{directory} holds nothing to attach and no extras were given")
    if rig_url is not None:
        manifest["rigUrl"] = check_name(rig_url)
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AttachError(f"{path}: {error}") from error
    if not isinstance(manifest, dict) or not UUID.match(str(manifest.get("assetId", ""))):
        raise AttachError(f"{path} names no asset id")
    if not isinstance(manifest.get("basedOn"), str):
        raise AttachError(f"{path} names no basedOn URL")
    files = manifest.get("files")
    if not isinstance(files, list) or files != files_under(directory):
        raise AttachError(f"{path}'s files are not exactly the files beside it")
    if not isinstance(manifest.get("extras", {}), dict):
        raise AttachError(f"{path}'s extras are not an object")
    if not files and not manifest.get("extras") and manifest.get("rigUrl") is None:
        raise AttachError(f"{path} attaches nothing: no files and no extras")
    return manifest


# --- staging and the request ------------------------------------------------------------


def staging_token() -> str:
    run, attempt = os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    token = f"{run}-{attempt}" if run else f"local-{secrets.token_hex(6)}"
    if not TOKEN.fullmatch(token):
        raise AttachError(f"{token!r} is not a staging token")
    return token


def s3_client() -> tuple[Any, str]:
    """The private bucket, from `OBJECT_STORAGE_*`."""
    names = (
        "OBJECT_STORAGE_ENDPOINT_URL",
        "OBJECT_STORAGE_ACCESS_KEY",
        "OBJECT_STORAGE_SECRET_KEY",
        "OBJECT_STORAGE_BUCKET",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise AttachError(f"missing {', '.join(missing)}: the private bucket stages the files")
    # Only `attach` needs it, and a workflow runs that with `uv run --with boto3`.
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
        aws_access_key_id=os.environ["OBJECT_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["OBJECT_STORAGE_SECRET_KEY"],
        region_name=os.environ.get("OBJECT_STORAGE_REGION", "auto"),
    )
    return client, os.environ["OBJECT_STORAGE_BUCKET"]


def stage(s3: Any, bucket: str, prefix: str, directory: Path, files: list[str]) -> None:
    """Upload `files` under `prefix`, after removing anything else a failed try left there:
    the API attaches every object under the prefix and refuses one `files` does not name."""
    stale: list[str] = []
    token: str | None = None
    while True:
        page = s3.list_objects_v2(
            Bucket=bucket, Prefix=prefix, **({"ContinuationToken": token} if token else {})
        )
        stale += [
            item["Key"]
            for item in page.get("Contents", [])
            if item["Key"][len(prefix) :] not in files
        ]
        token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
        if token is None:
            break
    for start in range(0, len(stale), 1000):
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in stale[start : start + 1000]]},
        )
    for rel in files:
        extension = rel.rsplit(".", 1)[-1].lower()
        s3.upload_file(
            str(directory / rel),
            bucket,
            prefix + rel,
            ExtraArgs={"ContentType": SIDECAR_TYPES[extension]},
        )
        print(f"staged {bucket}/{prefix}{rel}")


def post(url: str, body: Mapping[str, Any], token: str) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=POST_TIMEOUT_S) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, _body(error)
    except urllib.error.URLError as error:
        raise AttachError(f"POST {url}: {error.reason}") from error


def attach(
    directory: Path,
    *,
    api: str | None = None,
    s3: Any = None,
    bucket: str | None = None,
    write_token: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Stage `directory`'s files and attach them as its `attach.json` says; the response."""
    manifest = read_manifest(directory)
    token = write_token or os.environ.get("API_WRITE_TOKEN", "")
    if not token:
        raise AttachError("missing API_WRITE_TOKEN: the attach is a write")
    if s3 is None:
        s3, bucket = s3_client()
    assert bucket is not None
    asset_id = manifest["assetId"]
    prefix = f"{STAGING_ROOT}/{asset_id}/{staging_token()}/"
    stage(s3, bucket, prefix, directory, manifest["files"])

    body: dict[str, Any] = {
        "stagingPrefix": prefix,
        "basedOn": manifest["basedOn"],
        "files": manifest["files"],
        "extras": manifest.get("extras", {}),
    }
    if "rigUrl" in manifest:
        body["rigUrl"] = manifest["rigUrl"]
    url = f"{api_url(api)}/api/v1/assets/{asset_id}/sidecars"
    for attempt in range(1, BUSY_RETRIES + 2):
        status, answer = post(url, body, token)
        if status == 200 and isinstance(answer, dict):
            report(answer)
            return answer
        code, detail = refusal(answer)
        if status == 409 and code == BUSY and attempt <= BUSY_RETRIES:
            print(f"attach_sidecars: the asset is busy ({detail}); again in {BUSY_WAIT_S:.0f} s")
            sleep(BUSY_WAIT_S)
            continue
        if status == 409 and code == TILES_CHANGED_CODE:
            raise AttachError(
                f"409 from {url}: {detail}. The asset's tiles are no longer the ones these "
                f"files were computed against ({manifest['basedOn']}); run the workflow again "
                "on its current tiles.",
                TILES_CHANGED,
            )
        raise AttachError(f"{status} from {url}" + (f" ({code})" if code else "") + f": {detail}")
    raise AttachError("unreachable")  # pragma: no cover - the loop returns or raises


def refusal(answer: Any) -> tuple[str | None, str]:
    """A refusal's `code` (None where it has none) and the API's own words for it: the
    Problem's `detail`, or the whole body where it has none (a request validation error's
    `errors`, a proxy's page)."""
    if isinstance(answer, dict):
        code = answer.get("code")
        detail = answer.get("detail")
        return (
            code if isinstance(code, str) else None,
            detail if isinstance(detail, str) else json.dumps(answer),
        )
    return None, answer if isinstance(answer, str) else json.dumps(answer)


def report(answer: Mapping[str, Any]) -> None:
    lines = [
        f"- asset moved to generation `{answer.get('generation')}`: {answer.get('url')}",
        f"- from {answer.get('previousUrl')}",
        (
            f"- attached {answer.get('attached')}, carried {answer.get('carried')}; "
            f"{answer.get('copied')} objects copied, {len(answer.get('staged') or [])} staged"
        ),
    ]
    if answer.get("dropped"):
        # Keyed by the ids of a kind this attach replaced, and not sent with it.
        flags = (answer.get("asset") or {}).get("sidecarFlags") or []
        reasons = {flag.get("kind"): flag.get("reason") for flag in flags}
        lines += [f"- dropped {name}: {reasons.get(name, '')}" for name in answer["dropped"]]
    if answer.get("removed"):
        lines.append(f"- not copied from the previous generation: {answer['removed']}")
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write("### Sidecars attached\n\n" + "\n".join(lines) + "\n\n")


def _body(error: urllib.error.HTTPError) -> Any:
    try:
        text = error.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a body that cannot be read is not the error
        return ""
    try:
        return json.loads(text)
    except ValueError:
        return text[:2000]


# --- the command line -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", help=f"the API's origin (TWIN_API_URL, else {DEFAULT_API})")
    commands = parser.add_subparsers(dest="command", required=True)

    resolve = commands.add_parser("resolve", help="an asset and its current tileset URL")
    which = resolve.add_mutually_exclusive_group(required=True)
    which.add_argument("--asset", help="the asset's id")
    which.add_argument("--scan", help="a scan's short name (infra/modal/segment.py SCANS)")
    resolve.add_argument("--legacy-url", help="the scan's legacy URL, to find its run")
    resolve.add_argument("--out", type=Path, help="also write the answer here")

    manifest = commands.add_parser("manifest", help="write DIR/attach.json")
    manifest.add_argument("directory", type=Path)
    manifest.add_argument("--asset", required=True)
    manifest.add_argument("--based-on", required=True, help="the tileset URL they were made on")
    manifest.add_argument("--extras", default="{}", help="root extras to set, as JSON")
    manifest.add_argument("--extras-file", type=Path, help="... or read from this file")
    manifest.add_argument("--rig-url", help="set the asset's renderConfig.rigUrl")

    run = commands.add_parser("attach", help="stage DIR's files and attach them")
    run.add_argument("directory", type=Path)

    args = parser.parse_args(argv)
    if args.command == "resolve":
        found = (
            resolve_asset(args.asset, api=args.api)
            if args.asset
            else resolve_scan(args.scan, legacy_url=args.legacy_url, api=args.api)
        )
        text = json.dumps(found)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
    elif args.command == "manifest":
        raw = args.extras_file.read_text(encoding="utf-8") if args.extras_file else args.extras
        try:
            extras = json.loads(raw)
        except ValueError as error:
            raise AttachError(f"--extras is not JSON: {error}") from error
        if not isinstance(extras, dict):
            raise AttachError("--extras must be a JSON object")
        written = write_manifest(
            args.directory,
            asset_id=args.asset,
            based_on=args.based_on,
            extras=extras,
            rig_url=args.rig_url,
        )
        print(json.dumps(written, indent=1))
    else:
        attach(args.directory, api=args.api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
