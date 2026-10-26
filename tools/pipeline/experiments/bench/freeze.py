"""Only list/get operations on the private bucket. Local publication is transactional."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .scene import seal

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
ITEMS = {
    "normalize/frames/": "frames/",
    "normalize/frames_meta.json": "frames_meta.json",
    "pose/poses/": "poses/",
}


class ReadOnlyR2:
    """Capability boundary: deliberately neither S3Transfer nor a forwarding proxy."""

    def __init__(self, client: Any, bucket: str) -> None:
        self.__client, self.__bucket = client, bucket

    def list(self, prefix: str) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            extra = {"ContinuationToken": token} if token else {}
            page = self.__client.list_objects_v2(Bucket=self.__bucket, Prefix=prefix, **extra)
            objects.extend(obj for obj in page.get("Contents", []) if not obj["Key"].endswith("/"))
            if not page.get("IsTruncated"):
                return objects
            token = page.get("NextContinuationToken")
            if not token:
                raise ValueError("truncated R2 listing without continuation token")

    def get(self, obj: dict[str, Any], target: Path) -> None:
        response = self.__client.get_object(
            Bucket=self.__bucket, Key=obj["Key"], IfMatch=obj["ETag"]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        body = response["Body"]
        try:
            with target.open("wb") as sink:
                shutil.copyfileobj(body, sink)
        finally:
            body.close()
        if target.stat().st_size != obj["Size"]:
            raise ValueError("R2 object size changed during freeze")


def resolve_run(store: ReadOnlyR2, run: str) -> str:
    if not re.fullmatch(rf"(?:{UUID}|[0-9a-f]{{8}})", run):
        raise ValueError("run must be a lowercase UUID or 8-hex prefix")
    candidates = {
        obj["Key"].split("/")[1]
        for obj in store.list(f"runs/{run}")
        if re.match(rf"^runs/{UUID}/", obj["Key"])
    }
    if len(candidates) != 1:
        raise ValueError(f"run prefix resolves to {len(candidates)} runs; use a full UUID")
    return f"runs/{candidates.pop()}/"


def select_generation(
    objects: list[dict[str, Any]], root: str, generation: str | None = None
) -> tuple[str, list[tuple[dict[str, Any], str]]]:
    if (
        generation is not None
        and generation != "root"
        and not re.fullmatch(r"p[0-9a-f]{16}", generation)
    ):
        raise ValueError("generation must be root or p<16 hex>")
    prefixes = {root}
    for obj in objects:
        match = re.match(re.escape(root) + r"(p[0-9a-f]{16})/", obj["Key"])
        if match:
            prefixes.add(root + match[1] + "/")
    if generation is not None:
        prefixes = {root if generation == "root" else root + generation + "/"}
    populated = []
    for prefix in sorted(prefixes):
        members = []
        found = set()
        for obj in objects:
            key = obj["Key"]
            for source, destination in ITEMS.items():
                base = prefix + source
                if (source.endswith("/") and key.startswith(base)) or key == base:
                    relative = destination + key[len(base) :]
                    parts = relative.split("/")
                    if (
                        any(p in {"", ".", ".."} for p in parts)
                        or "\\" in relative
                        or any(ord(c) < 32 for c in relative)
                    ):
                        raise ValueError("unsafe object path")
                    if PurePosixPath(relative).is_absolute():
                        raise ValueError("absolute object path")
                    members.append((obj, relative))
                    found.add(source)
        if members:
            populated.append((prefix, members, found))
    if len(populated) != 1:
        raise ValueError("need exactly one populated root or generation; specify --generation")
    prefix, members, found = populated[0]
    if found != set(ITEMS):
        raise ValueError("selected generation is missing required inputs")
    return prefix, members


def freeze_scene(
    store: ReadOnlyR2,
    scenes: Path,
    scene: str,
    run: str,
    commit: str,
    *,
    generation: str | None = None,
    refreeze: bool = False,
) -> dict[str, Any]:
    target = scenes / scene
    if target.exists() and not refreeze:
        raise ValueError("scene is already frozen; use --refreeze explicitly")
    root = resolve_run(store, run)
    prefix, members = select_generation(store.list(root), root, generation)
    scenes.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=scenes) as temporary:
        stage = Path(temporary) / "scene"
        stage.mkdir()
        for obj, relative in members:
            store.get(obj, stage / relative)
        selected = prefix[len(root) :].strip("/") or "root"
        manifest = seal(stage, scene, prefix, selected, commit)
        if target.exists():
            # Download and validate the replacement before touching the old snapshot.
            target.rename(Path(temporary) / "previous")
        stage.rename(target)
    return manifest
