"""`infra/modal/app.py`'s `S3Transfer`, the container's side of every byte, without `modal`.

The class is read out of `app.py` with `ast` (the file imports `modal` at the top, which
this project deliberately does not install) and driven against an in-memory stand-in for
boto3's client: the five calls it makes, with S3's semantics for them. What is pinned is
the mirror the checkpoint syncer relies on -- and that what a call has just downloaded is
not sent straight back: a block's part used to re-upload the prior it was sent, and the
join every finished block it was sent, within the syncer's first minute.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[3] / "infra" / "modal" / "app.py"
WANTED = {"S3Transfer", "_parallel", "_signature", "TRANSFER_WORKERS"}


def s3_transfer() -> Any:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    body: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in WANTED)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in WANTED for t in node.targets)
        )
    ]
    assert {getattr(n, "name", None) for n in body} >= {"S3Transfer", "_parallel", "_signature"}
    namespace: dict[str, Any] = {"Any": Any, "Path": Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace["S3Transfer"]


class Bucket:
    """boto3's S3 client, the calls `S3Transfer` makes, over a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.deletes: list[str] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[key] = Path(filename).read_bytes()
        self.uploads.append(key)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def list_objects_v2(self, Bucket: str, Prefix: str, **_: Any) -> dict[str, Any]:  # noqa: N803
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        self.objects.pop(Key, None)
        self.deletes.append(Key)


def checkpoint(root: Path) -> Path:
    (root / "block-prior").mkdir(parents=True)
    (root / "block-prior" / "prior.npz").write_bytes(b"p" * 5000)
    (root / "blocks").mkdir()
    (root / "blocks" / "plan.json").write_text("{}", encoding="utf-8")
    return root


def test_what_a_call_downloaded_is_not_uploaded_again(tmp_path: Path) -> None:
    bucket = Bucket()
    worker = s3_transfer()(bucket, "b")
    worker.put("runs/r/train/parts/b0/checkpoint", checkpoint(tmp_path / "sent"))
    bucket.uploads.clear()

    # The container: a fresh transfer, the checkpoint fetched, then the syncer's puts.
    container = s3_transfer()(bucket, "b")
    local = tmp_path / "sandbox" / "checkpoint"
    fetched = container.get("runs/r/train/parts/b0/checkpoint", local)
    assert fetched == 5002

    assert container.put("runs/r/train/parts/b0/checkpoint", local) == 0
    assert bucket.uploads == [] and bucket.deletes == []

    # What the stage writes goes up, and only that; a file it changes goes up again.
    (local / "blocks" / "block_000.json").write_text('{"index": 0}', encoding="utf-8")
    assert container.put("runs/r/train/parts/b0/checkpoint", local) == len('{"index": 0}')
    assert bucket.uploads == ["runs/r/train/parts/b0/checkpoint/blocks/block_000.json"]
    (local / "blocks" / "plan.json").write_text('{"v": 2}', encoding="utf-8")
    container.put("runs/r/train/parts/b0/checkpoint", local)
    assert bucket.uploads[-1] == "runs/r/train/parts/b0/checkpoint/blocks/plan.json"
    assert bucket.objects["runs/r/train/parts/b0/checkpoint/blocks/plan.json"] == b'{"v": 2}'


def test_a_downloaded_member_the_stage_deletes_is_deleted_from_the_bucket(tmp_path: Path) -> None:
    """The join merges the finished blocks and deletes them from `checkpoint/`; the mirror
    must still delete them remotely, or the next restore brings them back."""
    bucket = Bucket()
    s3_transfer()(bucket, "b").put("k", checkpoint(tmp_path / "sent"))
    container = s3_transfer()(bucket, "b")
    local = tmp_path / "local"
    container.get("k", local)

    shutil.rmtree(local / "block-prior")
    uploaded = len(bucket.uploads)
    container.put("k", local)

    assert sorted(bucket.objects) == ["k/blocks/plan.json"]
    assert bucket.deletes == ["k/block-prior/prior.npz"]
    assert len(bucket.uploads) == uploaded  # nothing re-sent on the way


def test_a_single_object_round_trips(tmp_path: Path) -> None:
    bucket = Bucket()
    transfer = s3_transfer()(bucket, "b")
    source = tmp_path / "one.bin"
    source.write_bytes(b"x" * 10)
    assert transfer.put("k/one", source) == 10
    assert transfer.get("k/one", tmp_path / "back" / "one.bin") == 10
    assert (tmp_path / "back" / "one.bin").read_bytes() == b"x" * 10
