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
        return [SimpleNamespace(path=key) for key in self.files if key.rsplit("/", 1)[0] == path]


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


def seeds_line() -> str:
    return "live-seeds: " + json.dumps(
        {
            "v": 1,
            "key": f"runs/{RUN}/train/checkpoint/live/seeds.bin",
            "count": 1,
            "of": 1,
            "sources": {"sfm": 1, "dense": 0, "seed": 0},
            "origin": [0, 0, 0],
            "scale": 1,
            "bytes": 10,
        }
    )


def test_seeds_cached_served_and_signed_for_the_watcher_origin(tmp_path: Path) -> None:
    volume = Volume()
    volume.files[f"runs/{RUN}/live/train.log"] += ("\n" + seeds_line()).encode()
    volume.files[f"runs/{RUN}/live/seeds.bin"] = bytes(10)
    volume.files[f"runs/{RUN}/live/other.bin"] = b"private"
    watch = Watch(RUN, tmp_path)
    watch.poll(volume)
    watch.poll(volume)
    assert watch.state.train is not None and watch.state.train.seeds is not None
    assert watch.state.train.seeds.url == "/snapshots/seeds.bin"
    assert volume.reads.count(f"runs/{RUN}/live/seeds.bin") == 1
    headers, body = request(watch, "/snapshots/seeds.bin")
    assert (
        b"200 OK" in headers and b"Access-Control-Allow-Origin: *" in headers and body == bytes(10)
    )
    served = LiveState.model_validate_json(request(watch, f"/api/v1/captures/{RUN}/live")[1])
    assert served.train is not None and served.train.seeds is not None
    assert served.train.seeds.url == "http://bench.tail:8791/snapshots/seeds.bin"
    assert not (tmp_path / "other.bin").exists() and not list(tmp_path.glob("*.part"))
    pending = state_from_log(RUN, log_text() + "\n" + seeds_line(), None)
    assert (
        pending.train is not None
        and pending.train.seeds is not None
        and pending.train.seeds.url is None
    )
    for name in ("../seeds.bin", "seeds.bin.part", "seeds_other.bin", "seeds.bin/extra"):
        assert not snapshot_name(name, {name})
        assert b"404" in request(watch, "/snapshots/" + name)[0]
    assert not snapshot_name("seeds.bin", set())
    assert snapshot_name("seeds.bin", {"seeds.bin"})


def test_all_twenty_snapshots_survive_finished_volume_poll(tmp_path: Path) -> None:
    volume = Volume()
    lines = [log_text()]
    names = []
    for step in range(1, 21):
        name = f"splat_{step:06d}.spz"
        names.append(name)
        volume.files[f"runs/{RUN}/live/{name}"] = bytes([step])
        lines.append(
            "live-splat: "
            + json.dumps(
                {
                    "v": 1,
                    "step": step,
                    "total": 20,
                    "count": 1,
                    "of": 1,
                    "bytes": 1,
                    "key": f"runs/{RUN}/train/checkpoint/live/{name}",
                    "up": None,
                }
            )
        )
    # The ten-step fixture snapshot is replaced by its matching one in the twenty-step set.
    volume.files[f"runs/{RUN}/live/train.log"] = "\n".join(lines).encode()
    volume.files[f"runs/{RUN}/result.json"] = b'{"status":"complete"}'
    watch = Watch(RUN, tmp_path)
    for _ in range(3):
        watch.poll(volume)
    assert watch.state.status == "complete" and watch.state.train is not None
    assert len(watch.state.train.snapshots) == 20
    assert all(snapshot.url is not None for snapshot in watch.state.train.snapshots)
    assert len(list(tmp_path.glob("splat_*.spz"))) == 20
    assert all(volume.reads.count(f"runs/{RUN}/live/{name}") == 1 for name in names)


def test_real_quality_metrics_fixture_lights_up_bench_charts(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[3] / "tools/pipeline/tests/fixtures/run6fb-train.log"
    volume = Volume()
    volume.files[f"runs/{RUN}/live/train.log"] = path.read_bytes()
    watch = Watch(RUN, tmp_path)
    watch.poll(volume)
    state = watch.state
    assert state.train is not None and state.train.quality is not None
    quality = state.train.quality
    assert quality.far_field_share == 0.1418 and quality.elongation_share == 0.0658
    assert quality.series is not None
    assert quality.series.peak_allocated_gb[-1] == 5.9109
    assert quality.series.loss["depth"][-1] == 0.0537
    assert len(state.train.snapshots) == 2
    assert state.train.plan is not None and state.train.plan.cap_max == 3455958
    assert LiveState.model_validate_json(state.model_dump_json(by_alias=True)) == state


def test_final_test_scores_and_corrected_validation_scores_use_real_result_shape(
    tmp_path: Path,
) -> None:
    training = importlib.import_module("training")
    trainer = training.TrainMetrics(
        trainer="stand-in",
        psnr=18,
        ssim=0.7,
        lpips=0.3,
        cc_psnr=28,
        cc_ssim=0.95,
        cc_lpips=0.05,
    ).to_dict()
    (tmp_path / "test.json").write_text(json.dumps({"psnr": 25, "ssim": 0.9, "lpips": 0.1}))
    (tmp_path / "train_metrics.json").write_text(json.dumps(trainer))
    parse_result = importlib.import_module("experiments.bench.results").parse_result
    result = parse_result(
        {
            "runId": RUN,
            "scene": "spool",
            "variant": "baseline",
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
    # worker.run_training embeds its reports in result.json after parse_result.
    result["artifacts"] = {"train_metrics.json": trainer}
    volume = Volume()
    volume.files[f"runs/{RUN}/result.json"] = json.dumps(result).encode()
    watch = Watch(RUN, tmp_path / "cache")
    watch.poll(volume)
    assert watch.state.train is not None and watch.state.train.final is not None
    assert watch.state.train.final.model_dump(by_alias=True) == {
        "psnr": 25,
        "ssim": 0.9,
        "lpips": 0.1,
        "ccPsnr": 28,
        "ccSsim": 0.95,
        "ccLpips": 0.05,
    }
    # A result alone still supplies final scores when telemetry is absent.
    state = state_from_log(RUN, "", result)
    assert state.train is not None and state.train.final == watch.state.train.final
    running = state_from_log(RUN, log_text(), None)
    assert running.train is not None and running.train.final is None


def test_metrics_fallback_keeps_retry_order_and_split_line_semantics() -> None:
    rows = [
        'live-plan: {"v":1,"id":"old","stepsMax":100}',
        'live-metrics: {"step":100,"farFieldShare":0.9}',
        'live-plan: {"v":1,"id":"new","stepsMax":100}',
        'live-train: {"v":1,"k":"p","run":"new"}',
        '[provider] live-metrics: {"step":2,',
        '"farFieldShare":0.1,"lossTerms":{"depth":0.2}}',
        'live-train: {"v":1,"k":"s","run":"new","step":2}',
    ]
    state = state_from_log(RUN, "\n".join(rows), None)
    assert state.train is not None and state.train.quality is not None
    assert state.train.quality.series is not None
    assert state.train.quality.series.step == [2]
    assert state.train.quality.series.far_field_share == [0.1]
    assert state.train.quality.series.loss["depth"] == [0.2]


@pytest.mark.parametrize(
    "payload",
    [
        "invalid",
        "[]",
        '{"step":-1}',
        '{"step":true}',
        '{"step":1,"trainMemoryCurve":3}',
        '{"step":1,"valPsnrCurve":{}}',
        '{"step":1,"lossTerms":{"depth":0.1},"padding":"' + "x" * 96000 + '"}',
    ],
)
def test_malformed_metrics_cannot_stale_the_running_state(payload: str) -> None:
    before = state_from_log(RUN, log_text(), None)
    after = state_from_log(RUN, log_text() + "\nlive-metrics: " + payload, None)
    assert after == before


def test_upstream_metrics_parser_takes_precedence_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("app.scripts.bench_live")
    original = module.parse_line

    def parse(line: str, following: Any = ()) -> Any:
        if line.startswith("live-metrics: "):
            return "metrics", {"step": 2, "farFieldShare": 0.8}
        return original(line, following)

    monkeypatch.setattr(module, "parse_line", parse)
    records = module.bench_records('live-metrics: {"step":2,"farFieldShare":0.1}')
    assert records == [("metrics", {"step": 2, "farFieldShare": 0.8})]


def test_missing_or_nonfinite_final_scores_stay_unknown() -> None:
    for result in (
        {"status": "complete", "val": {"psnr": 30}},
        {"status": "complete", "test": {"psnr": float("nan"), "ssim": True, "lpips": float("inf")}},
    ):
        state = state_from_log(RUN, log_text(), result)
        assert state.train is not None and state.train.final is None
        assert LiveState.model_validate_json(state.model_dump_json(by_alias=True)) == state
    partial = state_from_log(RUN, "", {"status": "complete", "test": {"psnr": 25}})
    assert partial.train is not None and partial.train.final is not None
    assert partial.train.final.psnr == 25 and partial.train.final.cc_psnr is None
