"""Private R2 reader for export-run-inputs.yml. AWS CLI list/get only; never log keys
or response bodies. The workflow owns encryption and workspace cleanup."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
ALLOWED = {
    "normalize/frames/",
    "normalize/frames_meta.json",
    "pose/poses/",
    "place/placement.json",
}


class ExportError(Exception):
    """A fixed message safe to print in a public workflow log."""


def aws(operation, *args):
    result = subprocess.run(
        [
            "aws", "--endpoint-url", os.environ["OBJECT_STORAGE_ENDPOINT_URL"],
            "--region", "auto", "--output", "json", "s3api", operation,
            "--bucket", os.environ["OBJECT_STORAGE_BUCKET"], *args,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        # CLI errors can contain private keys/metadata; do not echo stdout/stderr.
        raise ExportError("R2 read failed; check credentials, bucket and run availability.")
    return json.loads(result.stdout)


def inventory(prefix, items):
    found = {}
    for item in items:
        objects = aws("list-objects-v2", "--prefix", prefix + item).get("Contents", [])
        members = []
        for obj in objects:
            key, size = obj["Key"], obj["Size"]
            if not key.startswith(prefix + item):
                raise ExportError("R2 returned an object outside the requested prefix.")
            if not item.endswith("/") and key != prefix + item:
                continue  # A similarly named file is not the requested file.
            if key.endswith("/") and size == 0:
                continue  # S3 directory markers are not inputs.
            relative = key[len(prefix):]
            if (
                any(part in ("", ".", "..") for part in relative.split("/"))
                or "\\" in relative
                or any(ord(char) < 32 or ord(char) == 127 for char in relative)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ExportError("Unsafe object path or size; export refused.")
            members.append((key, relative, size, obj["ETag"]))
        found[item] = members
    return found


def export(destination):
    prefix = os.environ["RUN_PREFIX"]
    items = os.environ["EXPORT_ITEMS"].split(" ")
    if not re.fullmatch(rf"runs/{UUID}/(?:p[0-9a-f]{{16}}/)?", prefix):
        raise ExportError("Invalid run prefix; nothing downloaded.")
    if not items or not set(items) <= ALLOWED or len(items) != len(set(items)):
        raise ExportError("Invalid items; nothing downloaded.")

    candidates = [prefix]
    if re.fullmatch(rf"runs/{UUID}/", prefix):
        listing = aws("list-objects-v2", "--prefix", prefix, "--delimiter", "/")
        for entry in listing.get("CommonPrefixes", []):
            candidate = entry["Prefix"]
            if re.fullmatch(re.escape(prefix) + r"p[0-9a-f]{16}/", candidate):
                candidates.append(candidate)
    populated = []
    for candidate in sorted(set(candidates)):
        found = inventory(candidate, items)
        if any(found.values()):
            populated.append((candidate, found))
    if not populated:
        raise ExportError("No requested inputs found under this run.")
    if len(populated) != 1:
        raise ExportError("Multiple populated prefixes; specify runs/<uuid>/p<16 hex>/ explicitly.")
    selected, found = populated[0]
    if not all(found.values()):
        raise ExportError("A requested item is missing; choose an available subset or another prefix.")

    objects = [obj for members in found.values() for obj in members]
    print(f"Selected prefix: {selected}", flush=True)
    destination.mkdir()
    for key, relative, size, etag in objects:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        aws("get-object", "--key", key, "--if-match", etag, str(target))
        if not target.is_file() or target.stat().st_size != size:
            raise ExportError("Downloaded size changed; export refused.")
    print(f"Downloaded {len(objects)} objects, {sum(obj[2] for obj in objects)} bytes.")


if __name__ == "__main__":
    try:
        export(Path(sys.argv[1]))
    except ExportError as error:
        sys.exit(f"::error::{error}")
    except Exception:
        # Unexpected local/JSON/CLI errors must not expose filenames or data either.
        sys.exit("::error::Export failed; no artifact will be uploaded.")
