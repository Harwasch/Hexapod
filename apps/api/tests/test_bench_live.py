"""The standalone bench console: pure projection, fake volume and socket-free requests."""

import importlib
import io
import json
import os
import threading
from datetime import UTC, datetime
from http.client import HTTPConnection, HTTPMessage
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.schemas.live import LiveState
from app.scripts.bench_live import Handler, Watch, snapshot_name, state_from_log

RUN = "0123456789abcdef0123456789abcdef"
NAME = "splat_000010.spz"
STARTED = datetime(2026, 11, 5, tzinfo=UTC)


def log_text() -> str:
    rows = [
        (
            "plan",
            {
                "id": "stage-plan-id",
                "stepsMax": 20,
                "refineStop": 10,
                "usdPerHour": 1,
                "benchScene": "spool",
                "benchVariant": "baseline",
                "metresPerUnit": None,
                "startedAt": STARTED.isoformat(),
            },
        ),
        (
            "cameras",
            {
                "seq": 1,
                "final": True,
                "registered": 1,
                "frames": 1,
                "origin": [0, 0, 0],
                "scale": 1,
                "up": None,
                "aspect": 1,
                "cameraCount": 1,
                "cameras": "AAAAAAAAAAAAAAAA",
                "pointCount": 0,
                "pointsTotal": 0,
                "points": "",
            },
        ),
        ("train", {"k": "s", "step": 10, "t": 1, "gs": 64, "sh": 1, "its": 10}),
        ("train", {"k": "e", "step": 10, "psnr": 20}),
        (
            "splat",
            {
                "step": 10,
                "count": 64,
                "total": 20,
                "of": 64,
                "bytes": 7,
                "key": f"runs/{RUN}/train/checkpoint/live/{NAME}",
                "up": None,
            },
        ),
    ]
    rows[0], rows[1] = rows[1], rows[0]
    return "noise\n" + "\n".join(
        f"live-{kind}: " + json.dumps({"v": 1, **value}) for kind, value in rows
    )


@pytest.mark.parametrize(
    "result,status,error",
    [
        (None, "in-progress", None),
        ({"status": "complete"}, "complete", None),
        ({"status": "failed", "error": "trainer stopped"}, "error", "trainer stopped"),
        ({"status": "preempted", "error": "restart refused"}, "error", "restart refused"),
    ],
)
def test_state_validates_and_keeps_events_snapshots_and_no_georef(
    result: Any, status: Any, error: Any
) -> None:
    state = state_from_log(RUN, log_text(), result, cached={NAME})
    document = state.model_dump(by_alias=True, mode="json")
    assert LiveState.model_validate(document) == state
    assert document["status"] == status and state.error == error
    assert state.capture_name == "spool · baseline" and state.site_id is None
    assert state.stage is not None
    assert state.train is not None and state.train.plan is not None
    assert state.cameras is not None and state.splat is not None
    assert state.stage.started_at == STARTED and state.stage.stage_id == "train"
    assert state.train.plan.metres_per_unit is None
    assert state.cameras.camera_count == 1
    assert state.splat.url == f"/snapshots/{NAME}"
    assert state.train.snapshots[0].url == state.splat.url
    assert {event.kind for event in state.train.events} >= {"setup", "sh", "eval", "densify"}
    assert state.steps_done == int(status == "complete")


def test_empty_or_invalid_log_is_a_valid_running_state_and_uncached_urls_are_null() -> None:
    state = state_from_log(RUN, "live-train: invalid\n", None, started_at=STARTED)
    assert state.train is None and state.cameras is None and state.splat is None
    assert state.stage is not None
    assert state.capture_name == RUN and state.stage.started_at == STARTED
    state = state_from_log(RUN, log_text(), None)
    assert state.splat is not None and state.train is not None
    assert state.splat.url is None and state.train.snapshots[0].url is None
    state = state_from_log(RUN, "", {"status": "failed"})
    assert state.error == "Bench run failed"


def test_ledger_name_and_rehearsal_label() -> None:
    state = state_from_log(
        RUN,
        "",
        None,
        plan={
            "scene": "spool",
            "variant": {"name": "trial"},
            "rehearsal": True,
        },
    )
    assert state.capture_name == "spool · trial (Simulated rehearsal)"


@pytest.mark.parametrize(
    "name",
    [
        "../splat_1.spz",
        "splat_../bad.spz",
        "splat_1.spz/other",
        "other.spz",
        "splat_1.spz?x",
        "/splat_1.spz",
        "splat_1.spz.part",
        "splat_%2e%2e.spz",
    ],
)
def test_snapshot_name_rejects_paths_and_non_snapshots_even_in_listing(name: Any) -> None:
    assert not snapshot_name(name, {name})


def test_snapshot_name_requires_cached_listing() -> None:
    assert snapshot_name(NAME, {NAME})
    assert not snapshot_name(NAME, set())


class Volume:
    def __init__(self) -> None:
        self.files = {
            f"runs/{RUN}/live/train.log": log_text().encode(),
            f"runs/{RUN}/live/{NAME}": b"preview",
        }
        self.reads: list[str] = []

    def read_file(self, path: str) -> list[bytes]:
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    def listdir(self, path: str) -> list[SimpleNamespace]:
        return [SimpleNamespace(path=f"{path}/{NAME}"), SimpleNamespace(path=f"{path}/result.json")]


def test_fake_volume_polls_downloads_once_and_finishes(tmp_path: Path) -> None:
    volume = Volume()
    watch = Watch(RUN, tmp_path)
    watch.poll(volume)
    assert watch.state.status == "in-progress"
    assert (tmp_path / NAME).read_bytes() == b"preview"
    assert watch.state.train is not None
    assert watch.state.train.snapshots[0].url == f"/snapshots/{NAME}"
    assert not list(tmp_path.glob("*.part"))
    volume.files[f"runs/{RUN}/result.json"] = b'{"status":"complete"}'
    watch.poll(volume)
    assert watch.state.status == "complete"
    assert volume.reads.count(f"runs/{RUN}/live/{NAME}") == 1


def request(watch: Watch, path: str, web_dist: Path | None = None) -> tuple[bytes, bytes]:
    # Exercise real header/body output without opening a listening socket.
    handler = object.__new__(Handler)
    handler.watch = watch
    handler.web_dist = web_dist
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.command = "GET"
    handler.headers = HTTPMessage()
    handler.headers["Host"] = "bench.tail:8791"
    handler.wfile = io.BytesIO()
    handler.do_GET()
    headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
    return headers, body


def test_handler_routes_cors_and_traversal_without_sockets(tmp_path: Path) -> None:
    watch = Watch(RUN, tmp_path / "cache")
    watch.poll(Volume())
    headers, body = request(watch, f"/api/v1/captures/{RUN}/live")
    assert b"200 OK" in headers and b"Access-Control-Allow-Origin: *" in headers
    state = LiveState.model_validate_json(body)
    assert state.status == watch.state.status
    assert state.splat is not None and state.train is not None
    assert state.splat.url == f"http://bench.tail:8791/snapshots/{NAME}"
    assert state.train.snapshots[0].url == state.splat.url
    assert watch.state.splat is not None
    assert watch.state.splat.url == f"/snapshots/{NAME}"
    assert request(watch, f"/snapshots/{NAME}")[1] == b"preview"
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "view.html").write_text("console")
    (dist / "assets").mkdir()
    (dist / "assets/app.js").write_text("app")
    (dist / "outside").symlink_to(tmp_path)
    (tmp_path / "secret").write_text("private")
    assert request(watch, "/", dist)[1] == b"console"
    assert request(watch, "/assets/app.js", dist)[1] == b"app"
    for path in (
        "/api/v1/sites",
        "/api/unknown",
        "/snapshots/splat_unknown.spz",
        "/snapshots/../secret",
        "/snapshots/%2e%2e/secret",
        "/../secret",
        "/%2e%2e/secret",
        "/outside/secret",
        "/cache/train.log",
    ):
        assert b"404" in request(watch, path, dist)[0], path


@pytest.mark.skipif(
    os.environ.get("BENCH_LIVE_SOCKET_TEST") != "1",
    reason="opt-in socket test; sandbox refuses sockets",
)
def test_http_server_on_loopback(tmp_path: Path) -> None:
    from functools import partial

    watch = Watch(RUN, tmp_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, watch=watch, web_dist=None))
    except PermissionError:
        pytest.skip("sandbox refuses localhost sockets")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", f"/api/v1/captures/{RUN}/live")
        response = connection.getresponse()
        assert response.status == 200
        LiveState.model_validate_json(response.read())
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("variant", ["baseline", {"name": "baseline", "overrides": {}}])
def test_finished_state_from_real_bench_result(tmp_path: Path, variant: Any) -> None:
    parse_result = importlib.import_module("experiments.bench.results").parse_result

    result = parse_result(
        {
            "runId": RUN,
            "scene": "spool",
            "variant": variant,
            "seed": 42,
            "schedule": "screen",
            "tier": "l4",
            "benchCommit": "test",
            "timeoutSeconds": 1800,
        },
        tmp_path,
        params={},
        split_digest="split",
        manifest_digest="manifest",
        wall=5,
        train_seconds=4,
        gpu_name="stand-in",
    )
    # Exercise the serialized volume document rather than a simplified status-only stub.
    state = state_from_log(RUN, log_text(), json.loads(json.dumps(result)), cached={NAME})
    assert state.status == "complete" and state.capture_name == "spool · baseline"
    assert state.train is not None and state.train.events
    assert LiveState.model_validate_json(state.model_dump_json(by_alias=True)) == state


def test_a_missing_live_folder_stays_quiet_and_a_finished_run_stops_polling(tmp_path: Path) -> None:
    class NotFoundError(Exception):
        """Modal's `modal.exception.NotFoundError`, which `listdir` raises for a missing path."""

    class Volume:
        reads = 0

        def read_file(self, path: str) -> list[bytes]:
            self.reads += 1
            if path.endswith("result.json"):
                return [b'{"status": "complete"}']
            raise FileNotFoundError(path)

        def listdir(self, path: str) -> list[Any]:
            raise NotFoundError(path)

    run_id = "19c88457eeee4f3ca60e046d03a93d80"
    watch = Watch(run_id, tmp_path)
    volume = Volume()
    watch.poll(volume)
    watch.poll(volume)
    settled = volume.reads
    watch.poll(volume)
    assert volume.reads == settled
    assert watch.errors == set()
    assert watch.state.status == "complete"


def test_running_console_combines_train_and_quality_metrics() -> None:
    text = (
        log_text()
        + "\nlive-metrics: "
        + json.dumps(
            {
                "step": 10,
                "splatCount": 64,
                "farFieldShare": 0.1,
                "elongationShare": 0.2,
                "trainMemoryCurve": [{"step": 10, "peakAllocatedGb": 3}],
                "lossTerms": {"depth": 0.01},
            }
        )
    )
    text += '\nlive-metrics: {"trainMemoryCurve":[{"step":20,"peakAllocatedGb":4}]}'
    state = state_from_log(RUN, text, None)
    assert state.train is not None and state.train.quality is not None
    assert state.train.quality.far_field_share == 0.1
    assert state.train.quality.elongation_share == 0.2
    assert state.train.series.gs == [64]
