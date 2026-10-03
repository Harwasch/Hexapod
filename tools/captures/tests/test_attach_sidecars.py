"""attach_sidecars.py: what every publishing workflow sends, against a stub bucket and API.

The five workflows that publish beside a scan's tiles stage their files in the private
bucket and POST `/api/v1/assets/{id}/sidecars` through this one script. A stub S3 client
records what is uploaded where; a stub HTTP server plays the API (the asset reads, the
attach and its refusals). Nothing here touches a network beyond 127.0.0.1.
"""

from __future__ import annotations

import ast
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import attach_sidecars as attach

ASSET = "3f2b8c1e-0d4a-4c6b-9e1f-2a3b4c5d6e7f"
OTHER = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
JOB = "50c25673-0940-4574-9b96-0b21362f83ca"
PUBLIC = "https://pub-0123.r2.dev"
LEGACY = f"{PUBLIC}/runs/{JOB}/package/splat/tileset.json"
CURRENT = f"{PUBLIC}/runs/{JOB}/p0123456789abcdef/package/splat/tileset.json"
API_SIDECARS = Path(__file__).resolve().parents[3] / "apps/api/app/services/sidecars.py"


class StubS3:
    """The calls `attach_sidecars.stage` makes, on a dict of key -> (bytes, content type)."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.deleted: list[str] = []

    def list_objects_v2(self, *, Bucket: str, Prefix: str, **kw: Any) -> dict[str, Any]:
        keys = sorted(k for k in self.objects if k.startswith(f"{Bucket}/{Prefix}"))
        return {"Contents": [{"Key": k.split("/", 1)[1]} for k in keys], "IsTruncated": False}

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> None:
        for entry in Delete["Objects"]:
            self.deleted.append(entry["Key"])
            self.objects.pop(f"{Bucket}/{entry['Key']}", None)

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs: dict[str, str]) -> None:
        self.objects[f"{bucket}/{key}"] = (Path(filename).read_bytes(), ExtraArgs["ContentType"])


class StubApi:
    """A tiny API: canned answers by method and path, and every request it was sent."""

    def __init__(self) -> None:
        self.answers: dict[tuple[str, str], list[tuple[int, Any]]] = {}
        self.requests: list[dict[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _answer(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                stub.requests.append(
                    {
                        "method": method,
                        "path": self.path,
                        "body": body,
                        "authorization": self.headers.get("Authorization"),
                    }
                )
                queue = stub.answers.get((method, self.path)) or [(404, {"detail": "no"})]
                status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._answer("GET")

            def do_POST(self) -> None:
                self._answer("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def on(self, method: str, path: str, *answers: tuple[int, Any]) -> None:
        self.answers[(method, path)] = list(answers)


@pytest.fixture
def api() -> Iterator[StubApi]:
    stub = StubApi()
    thread = threading.Thread(target=stub.server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield stub
    stub.server.shutdown()
    stub.server.server_close()


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TWIN_API_URL", "API_WRITE_TOKEN", "SCAN_ASSET_IDS", "GITHUB_STEP_SUMMARY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")


def an_asset(asset_id: str = ASSET, url: str = CURRENT, **fields: Any) -> dict[str, Any]:
    return {
        "id": asset_id,
        "name": "Camp splat",
        "representation": "gaussian-splat",
        "source": {"type": "3d-tiles-url", "url": url},
        **fields,
    }


def instances_dir(tmp_path: Path) -> Path:
    out = tmp_path / "camp"
    out.mkdir()
    (out / "instances.json").write_text('{"tiles": {}}')
    (out / "instances.emb").write_bytes(b"\0\1\2")
    return out


def attachment(**fields: Any) -> dict[str, Any]:
    return {
        "url": CURRENT.replace("p0123456789abcdef", "pfedcba9876543210"),
        "previousUrl": CURRENT,
        "generation": "fedcba9876543210",
        "copied": 12,
        "staged": ["instances.emb", "instances.json"],
        "attached": ["instances"],
        "carried": [],
        "extras": ["gaussians", "instances"],
        **fields,
    }


# --- resolve ----------------------------------------------------------------------------


def test_resolve_reads_the_assets_current_url(api: StubApi) -> None:
    api.on("GET", f"/api/v1/assets/{ASSET}", (200, an_asset()))
    assert attach.resolve_asset(ASSET, api=api.url) == {
        "assetId": ASSET,
        "url": CURRENT,
        "name": "Camp splat",
    }


def test_an_asset_that_is_not_a_tileset_url_has_no_sidecars(api: StubApi) -> None:
    api.on(
        "GET",
        f"/api/v1/assets/{ASSET}",
        (200, {**an_asset(), "source": {"type": "cesium-ion", "assetId": 7}}),
    )
    with pytest.raises(attach.AttachError, match="not a 3D Tiles URL"):
        attach.resolve_asset(ASSET, api=api.url)
    with pytest.raises(attach.AttachError, match="not an asset id"):
        attach.resolve_asset("camp", api=api.url)


def test_a_scan_is_its_asset_in_the_map(api: StubApi, monkeypatch: pytest.MonkeyPatch) -> None:
    api.on("GET", f"/api/v1/assets/{ASSET}", (200, an_asset()))
    monkeypatch.setenv("SCAN_ASSET_IDS", json.dumps({"camp": ASSET}))
    found = attach.resolve_scan("camp", legacy_url=LEGACY, api=api.url)
    assert (found["assetId"], found["url"], found["scan"]) == (ASSET, CURRENT, "camp")
    assert [r["path"] for r in api.requests] == [f"/api/v1/assets/{ASSET}"]


def test_a_scan_missing_from_the_map_is_found_by_its_run(api: StubApi) -> None:
    """The one gaussian-splat asset whose tileset is the scan's run, at the legacy prefix or
    in a generation cut from it; a mesh of the same run, or another run, is not it."""
    listing = [
        an_asset(),
        an_asset(OTHER, f"{PUBLIC}/runs/{'1' * 8}-0000-4000-8000-{'0' * 12}/package/splat/x.json"),
        {**an_asset(OTHER, CURRENT), "representation": "mesh"},
    ]
    api.on("GET", "/api/v1/assets", (200, listing))
    api.on("GET", f"/api/v1/assets/{ASSET}", (200, an_asset()))
    assert attach.resolve_scan("camp", legacy_url=LEGACY, api=api.url, mapping={})["assetId"] == (
        ASSET
    )


def test_a_scan_nothing_identifies_says_what_to_set(api: StubApi) -> None:
    api.on("GET", "/api/v1/assets", (200, [an_asset(), an_asset(OTHER, LEGACY)]))
    with pytest.raises(attach.AttachError) as two:
        attach.resolve_scan("camp", legacy_url=LEGACY, api=api.url, mapping={})
    assert "2 assets show its run" in two.value.message
    assert 'SCAN_ASSET_IDS to a JSON object naming it, {"camp": "<asset id>"}' in two.value.message
    with pytest.raises(attach.AttachError, match="SCAN_ASSET_IDS"):
        attach.resolve_scan("yard", api=api.url, mapping={})


def test_a_map_that_is_not_an_object_of_ids_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("[1]", "{not json", '{"camp": 7}'):
        monkeypatch.setenv("SCAN_ASSET_IDS", raw)
        with pytest.raises(attach.AttachError, match="SCAN_ASSET_IDS"):
            attach.scan_assets()


# --- the manifest -----------------------------------------------------------------------


def test_the_manifest_names_every_file_beside_it_and_the_request(tmp_path: Path) -> None:
    out = instances_dir(tmp_path)
    (out / "sog").mkdir()
    (out / "sog" / "lod-meta.json").write_text("{}")
    written = attach.write_manifest(
        out,
        asset_id=ASSET,
        based_on=CURRENT,
        extras={"instances": {"uri": "instances.json", "count": 2}},
    )
    assert written == {
        "assetId": ASSET,
        "basedOn": CURRENT,
        "files": ["instances.emb", "instances.json", "sog/lod-meta.json"],
        "extras": {"instances": {"uri": "instances.json", "count": 2}},
    }
    assert json.loads((out / "attach.json").read_text()) == written
    assert attach.read_manifest(out) == written
    # Written again, it does not list itself.
    assert attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT)["files"] == written["files"]


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("tileset.json", "written by the API"),
        ("objects/3/tileset.json", "split"),
        ("fills/3/0.glb", "split"),
        ("page.html", "extension"),
        ("noextension", "extension"),
        (".hidden.json", "plain path"),
        ("a/../b.json", "plain path"),
        ("a/b/c/d/e/f/g.json", "plain path"),
    ],
)
def test_what_the_api_would_refuse_is_refused_before_anything_is_uploaded(
    name: str, why: str
) -> None:
    with pytest.raises(attach.AttachError, match=why):
        attach.check_name(name)


def test_an_inferred_layers_own_tileset_may_be_staged() -> None:
    assert attach.check_name("inferred/fixer/tileset.json") == "inferred/fixer/tileset.json"


def test_a_manifest_that_does_not_match_its_directory_is_refused(tmp_path: Path) -> None:
    out = instances_dir(tmp_path)
    attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT)
    (out / "instances.emb").unlink()
    with pytest.raises(attach.AttachError, match="not exactly the files"):
        attach.read_manifest(out)


def test_the_rules_are_the_apis() -> None:
    """The API decides; these only fail earlier. Read from its source, so they cannot drift."""
    source = API_SIDECARS.read_text()
    types = re.search(r"SIDECAR_TYPES: dict\[str, str\] = \{(.*?)\n\}", source, re.DOTALL)
    assert types is not None
    extensions = dict(re.findall(r'"(\w+)": (?:"([\w/.-]+)"|OCTET)', types[1]))
    assert set(extensions) == set(attach.SIDECAR_TYPES)
    for name in ("MAX_DEPTH", "MAX_PATH", "MAX_SIDECAR_FILES", "STAGING_ROOT"):
        spelled = re.search(rf"^{name} = (.+)$", source, re.MULTILINE)
        assert spelled is not None, name
        assert ast.literal_eval(spelled[1]) == getattr(attach, name), name
    for name in ("_TOKEN", "_SEGMENT"):
        spelled = re.search(rf'^{name} = re.compile\(r"(.+)"\)$', source, re.MULTILINE)
        assert spelled is not None, name
        assert spelled[1] == getattr(attach, name.lstrip("_")).pattern, name


# --- attach -----------------------------------------------------------------------------


def test_attach_stages_the_files_then_posts_the_request(
    api: StubApi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = instances_dir(tmp_path)
    extras = {"instances": {"uri": "instances.json", "count": 2}}
    attach.write_manifest(out, asset_id=ASSET, based_on=LEGACY, extras=extras)
    s3 = StubS3()
    prefix = f"staging/assets/{ASSET}/1234-2/"
    # A failed earlier try of this attempt left a file nobody asked for.
    s3.objects[f"twin-assets/{prefix}stale.bin"] = (b"old", "application/octet-stream")
    api.on("POST", f"/api/v1/assets/{ASSET}/sidecars", (200, attachment()))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    answer = attach.attach(out, api=api.url, s3=s3, bucket="twin-assets", write_token="secret")

    assert answer["generation"] == "fedcba9876543210"
    assert s3.objects == {
        f"twin-assets/{prefix}instances.emb": (b"\0\1\2", "application/octet-stream"),
        f"twin-assets/{prefix}instances.json": (b'{"tiles": {}}', "application/json"),
    }
    assert s3.deleted == [f"{prefix}stale.bin"]
    (request,) = api.requests
    assert request["authorization"] == "Bearer secret"
    assert request["body"] == {
        "stagingPrefix": prefix,
        "basedOn": LEGACY,
        "files": ["instances.emb", "instances.json"],
        "extras": extras,
    }
    assert "fedcba9876543210" in summary.read_text()


def test_attach_sends_the_rig_url_when_the_manifest_sets_one(api: StubApi, tmp_path: Path) -> None:
    out = tmp_path / "rig"
    out.mkdir()
    for name in ("rig.json", "motion.json", "plants.json"):
        (out / name).write_text("{}")
    attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT, rig_url="rig.json")
    api.on("POST", f"/api/v1/assets/{ASSET}/sidecars", (200, attachment(attached=["rig"])))
    attach.attach(out, api=api.url, s3=StubS3(), bucket="b", write_token="t")
    assert api.requests[0]["body"]["rigUrl"] == "rig.json"
    assert api.requests[0]["body"]["files"] == ["motion.json", "plants.json", "rig.json"]


def test_the_report_says_what_the_attach_dropped_and_why(
    api: StubApi, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """New objects take the materials keyed by the old ones with them; the log says so."""
    out = instances_dir(tmp_path)
    attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT)
    reason = "it names instances ids, and an attach replaced instances"
    answer = attachment(
        dropped=["materials"],
        removed=["materials.json"],
        asset={"sidecarFlags": [{"kind": "materials", "reason": reason}]},
    )
    api.on("POST", f"/api/v1/assets/{ASSET}/sidecars", (200, answer))
    attach.attach(out, api=api.url, s3=StubS3(), bucket="b", write_token="t")
    printed = capsys.readouterr().out
    assert f"- dropped materials: {reason}" in printed
    assert "not copied from the previous generation: ['materials.json']" in printed


def test_a_busy_asset_is_asked_again(api: StubApi, tmp_path: Path) -> None:
    out = instances_dir(tmp_path)
    attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT)
    busy = {"detail": f"another attach or publish of asset {ASSET} is in progress; retry shortly"}
    api.on(
        "POST", f"/api/v1/assets/{ASSET}/sidecars", (409, busy), (409, busy), (200, attachment())
    )
    waits: list[float] = []

    attach.attach(out, api=api.url, s3=StubS3(), bucket="b", write_token="t", sleep=waits.append)

    assert len(api.requests) == 3
    assert waits == [attach.BUSY_WAIT_S, attach.BUSY_WAIT_S]


def test_tiles_that_changed_under_the_run_end_it_with_its_own_status(
    api: StubApi, tmp_path: Path
) -> None:
    out = instances_dir(tmp_path)
    attach.write_manifest(out, asset_id=ASSET, based_on=LEGACY)
    changed = {"detail": "the asset's tiles are no longer the ones at ..."}
    api.on("POST", f"/api/v1/assets/{ASSET}/sidecars", (409, changed))

    with pytest.raises(attach.AttachError) as refused:
        attach.attach(out, api=api.url, s3=StubS3(), bucket="b", write_token="t")

    assert refused.value.code == attach.TILES_CHANGED
    assert "run the workflow again on its current tiles" in refused.value.message
    assert len(api.requests) == 1


def test_any_other_refusal_is_reported_whole(api: StubApi, tmp_path: Path) -> None:
    out = instances_dir(tmp_path)
    attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT)
    api.on("POST", f"/api/v1/assets/{ASSET}/sidecars", (422, {"detail": "instances.json is bad"}))
    with pytest.raises(attach.AttachError) as refused:
        attach.attach(out, api=api.url, s3=StubS3(), bucket="b", write_token="t")
    assert refused.value.code == 1
    assert "422" in refused.value.message and "instances.json is bad" in refused.value.message


def test_attach_needs_the_write_token_and_the_private_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = instances_dir(tmp_path)
    attach.write_manifest(out, asset_id=ASSET, based_on=CURRENT)
    with pytest.raises(attach.AttachError, match="API_WRITE_TOKEN"):
        attach.attach(out, api="http://127.0.0.1:9", s3=StubS3(), bucket="b")
    monkeypatch.setenv("API_WRITE_TOKEN", "t")
    for name in ("OBJECT_STORAGE_ENDPOINT_URL", "OBJECT_STORAGE_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(attach.AttachError, match="OBJECT_STORAGE_BUCKET"):
        attach.attach(out, api="http://127.0.0.1:9")


def test_the_command_line_writes_a_manifest_and_resolves(
    api: StubApi, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = instances_dir(tmp_path)
    assert (
        attach.main(
            [
                "manifest",
                str(out),
                "--asset",
                ASSET,
                "--based-on",
                CURRENT,
                "--extras",
                '{"collision": {"uri": "collision.bin"}}',
            ]
        )
        == 0
    )
    assert json.loads((out / "attach.json").read_text())["extras"] == {
        "collision": {"uri": "collision.bin"}
    }
    api.on("GET", f"/api/v1/assets/{ASSET}", (200, an_asset()))
    capsys.readouterr()
    assert attach.main(["--api", api.url, "resolve", "--asset", ASSET]) == 0
    assert json.loads(capsys.readouterr().out)["url"] == CURRENT
