"""The standalone bench console: pure projection, fake volume and socket-free requests."""

import importlib
import io
import json
import os
import subprocess
import threading
from datetime import UTC, datetime
from http.client import HTTPConnection, HTTPMessage
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.schemas.live import LiveState
from app.scripts.bench_live import Handler, Watch, Watches, snapshot_name, state_from_log

RUN = "0123456789abcdef0123456789abcdef"
NAME = "splat_000010.spz"
STARTED = datetime(2026, 11, 5, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_real_modal_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        pytest.fail("bench tests must fake the Modal subprocess")

    monkeypatch.setattr("app.scripts.bench_live.subprocess.run", refuse)


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
    assert state.splat.url == f"/snapshots/{RUN}/{NAME}"
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
    assert watch.state.train.snapshots[0].url == f"/snapshots/{RUN}/{NAME}"
    assert not list(tmp_path.glob("*.part"))
    volume.files[f"runs/{RUN}/result.json"] = b'{"status":"complete"}'
    watch.poll(volume)
    assert watch.state.status == "complete"
    assert volume.reads.count(f"runs/{RUN}/live/{NAME}") == 1


def request(watch: Watch | Watches, path: str, web_dist: Path | None = None) -> tuple[bytes, bytes]:
    # Exercise real header/body output without opening a listening socket.
    handler = object.__new__(Handler)
    handler.watches = watch if isinstance(watch, Watches) else Watches(watch)
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
    assert state.splat.url == f"http://bench.tail:8791/snapshots/{RUN}/{NAME}"
    assert state.train.snapshots[0].url == state.splat.url
    assert watch.state.splat is not None
    assert watch.state.splat.url == f"/snapshots/{RUN}/{NAME}"
    assert request(watch, f"/snapshots/{RUN}/{NAME}")[1] == b"preview"
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
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(Handler, watches=Watches(watch), web_dist=None)
        )
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
    assert watch.state.train.seeds.url == f"/snapshots/{RUN}/seeds.bin"
    assert volume.reads.count(f"runs/{RUN}/live/seeds.bin") == 1
    headers, body = request(watch, "/snapshots/seeds.bin")
    assert (
        b"200 OK" in headers and b"Access-Control-Allow-Origin: *" in headers and body == bytes(10)
    )
    served = LiveState.model_validate_json(request(watch, f"/api/v1/captures/{RUN}/live")[1])
    assert served.train is not None and served.train.seeds is not None
    assert served.train.seeds.url == f"http://bench.tail:8791/snapshots/{RUN}/seeds.bin"
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


@pytest.mark.parametrize(
    "error,reason",
    [
        ("Preempted; restart refused, no second attempt: run already started", "failed"),
        ("Remote outcome unknown: FunctionTimeoutError: timed out", "failed"),
        ("trainer stopped", "failed"),
        ("Runner killed (SIGKILL), exit code: 137", "oom"),
        ("torch.OutOfMemoryError: CUDA out of memory", "oom"),
    ],
)
@pytest.mark.parametrize("source", ["volume", "ledger"])
def test_failed_bench_freezes_at_last_telemetry(error: str, reason: str, source: str) -> None:
    failure = {"event": "failed", "runId": RUN, "error": error}
    result = {"status": "failed", "error": error} if source == "volume" else None
    text = log_text() + '\nlive-train: {"v":1,"k":"e","step":12,"psnr":21}'
    state = state_from_log(RUN, text, result, failure=failure if source == "ledger" else None)
    assert state.status == "error" and state.error == error
    assert state.train is not None and state.train.ended is not None
    assert state.train.ended.model_dump() == {"reason": reason, "step": 12}


def test_failure_without_telemetry_ends_at_zero_and_complete_result_wins() -> None:
    failure = {"event": "failed", "runId": RUN, "error": "Preempted; restart refused"}
    state = state_from_log(RUN, "", None, failure=failure)
    assert state.train is not None and state.train.ended is not None
    assert state.train.ended.step == 0
    complete = state_from_log(RUN, log_text(), {"status": "complete"}, failure=failure)
    assert complete.status == "complete"
    assert complete.train is not None and complete.train.ended is None


def replacement_rows() -> list[dict[str, Any]]:
    fixture = Path(__file__).parent / "fixtures/bench-replacement.jsonl"
    return [json.loads(line) for line in fixture.read_text().splitlines()]


def test_real_ledger_failure_replacement_and_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, new, *_ = replacement_rows()
    book = tmp_path / "custom.jsonl"
    book.write_text(json.dumps(old) + "\n")
    volume = Volume()
    watch = Watch(old["runId"], tmp_path / "cache")
    watch.poll(volume)
    assert watch.state.status == "in-progress"
    with book.open("a") as output:
        output.write(json.dumps(failure) + "\n" + json.dumps(new) + "\n")
    watch.poll(volume)
    assert watch.state.error == failure["error"]
    assert watch.state.train is not None and watch.state.train.ended is not None
    assert watch.state.train.ended.reason == "failed"
    assert watch.state.train.replaced_by == new["runId"]
    assert watch.state.capture_name == "beach-full · full-data"


def test_finished_volume_still_rereads_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, _, new, *_ = replacement_rows()
    book = tmp_path / "custom.jsonl"
    book.write_text(json.dumps(old) + "\n")
    watch = Watch(old["runId"], tmp_path / "cache")
    volume = Volume()
    volume.files[f"runs/{old['runId']}/result.json"] = b'{"status":"complete"}'
    watch.poll(volume)
    watch.poll(volume)
    with book.open("a") as output:
        output.write(json.dumps(new) + "\n")
    watch.poll(volume)
    assert watch.state.train is not None and watch.state.train.replaced_by == new["runId"]
    assert watch.state.train.ended is None


def test_replacement_across_files_ignores_tags_and_respects_variant_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, _, new, *_ = replacement_rows()
    first = tmp_path / "z-first.jsonl"
    first.write_text(json.dumps(old) + "\n")
    second = tmp_path / "a-second.jsonl"
    others = [
        {**new, "runId": f"{i:032x}", key: value}
        for i, (key, value) in enumerate(
            [
                ("scene", "other"),
                ("variant", {"name": "other"}),
                ("seed", 43),
                ("schedule", "screen"),
            ]
        )
    ]
    second.write_text("\n".join(json.dumps(row) for row in [*others, new]) + "\n")
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    ledger = bench_live.LocalLedger.read()
    assert ledger.replacement(old["runId"]) == new["runId"]
    assert ledger.replacement(new["runId"]) is None
    # A timestamp overrides cross-file mtime; line order within one file remains authoritative.
    first.write_text(json.dumps({**old, "timestamp": "2026-11-06T02:00:00Z"}) + "\n")
    second.write_text(json.dumps({**new, "timestamp": "2026-11-06T01:00:00Z"}) + "\n")
    assert bench_live.LocalLedger.read().replacement(old["runId"]) is None


def test_multi_run_routes_cap_and_idle_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    started: list[Watch] = []
    monkeypatch.setattr(Watches, "start", lambda self, watch: started.append(watch))
    old, _, new, *_ = replacement_rows()
    extras = [{**new, "runId": f"{i:032x}"} for i in range(10)]
    rows = [old, new, *extras]
    (tmp_path / "custom.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    primary = Watch(old["runId"], tmp_path / "primary")
    watches = Watches(primary)
    with watches.acquire(old["runId"]) as pinned:
        assert pinned is primary
        for row in rows[1:]:
            headers, body = request(watches, f"/api/v1/captures/{row['runId']}/live")
            assert b"200 OK" in headers
            assert LiveState.model_validate_json(body).capture_id.hex == row["runId"]
            assert len(watches.watches) <= 8
        assert not primary.stop.is_set()
        assert started[0].stop.is_set()  # oldest idle watch, not the pinned primary
    assert len(watches.watches) == 8
    assert b"404" in request(watches, f"/api/v1/captures/{RUN}/live")[0]
    assert b"404" in request(watches, "/api/v1/captures/../live")[0]
    # Requesting a recently added ledger id works without restarting the server.
    replacement = new["runId"]
    assert b"200 OK" in request(watches, f"/api/v1/captures/{replacement}/live")[0]
    current = watches.watches[replacement]
    current.cache.mkdir(parents=True)
    (current.cache / NAME).write_bytes(b"replacement")
    current.cached = frozenset({NAME})
    assert request(watches, f"/snapshots/{replacement}/{NAME}")[1] == b"replacement"
    primary.cache.mkdir(parents=True)
    (primary.cache / NAME).write_bytes(b"primary")
    primary.cached = frozenset({NAME})
    assert request(watches, f"/snapshots/{NAME}")[1] == b"primary"
    for path in (
        f"/snapshots/{replacement}/../{NAME}",
        f"/snapshots/{replacement}/%2e%2e/{NAME}",
        f"/snapshots/%2e%2e/{NAME}",
        f"/snapshots/{RUN}/{NAME}",
        f"/snapshots/{replacement}/seeds.bin.part",
    ):
        assert b"404" in request(watches, path)[0]
    watches.close()
    assert all(watch.stop.is_set() for watch in started)


def test_all_busy_watches_are_kept_and_new_known_run_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import ExitStack

    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    monkeypatch.setattr(Watches, "start", lambda self, watch: None)
    old, _, new, *_ = replacement_rows()
    rows = [old, *[{**new, "runId": f"{i:032x}"} for i in range(8)]]
    (tmp_path / "custom.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    watches = Watches(Watch(old["runId"], tmp_path / "primary"))
    with ExitStack() as stack:
        for row in rows[:-1]:
            assert stack.enter_context(watches.acquire(row["runId"])) is not None
        assert b"503" in request(watches, f"/api/v1/captures/{rows[-1]['runId']}/live")[0]
        assert len(watches.watches) == 8
    assert b"200 OK" in request(watches, f"/api/v1/captures/{rows[-1]['runId']}/live")[0]
    watches.close()


def test_ledger_failure_waits_for_first_volume_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, *_ = replacement_rows()
    (tmp_path / "custom.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(failure) + "\n")
    watch = Watch(old["runId"], tmp_path / "cache")
    assert watch.state.train is None and watch.state.status == "in-progress"
    volume = Volume()
    volume.files[f"runs/{old['runId']}/live/train.log"] = log_text().encode()
    watch.poll(volume)
    assert watch.state.train is not None and watch.state.train.ended is not None
    assert watch.state.train.ended.step == 10


@pytest.mark.parametrize("saved", ["stop", "final"])
def test_saved_training_has_no_ended_if_result_reports_failure(saved: str) -> None:
    text = log_text()
    result: dict[str, Any] = {"status": "failed", "error": "later report failed"}
    if saved == "stop":
        text += '\nlive-train: {"v":1,"k":"x","step":10}'
    else:
        result["test"] = {"psnr": 25}
    state = state_from_log(RUN, text, result)
    assert state.train is not None and state.train.ended is None


def test_partial_or_invalid_ledger_row_does_not_hide_other_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, new, *_ = replacement_rows()
    (tmp_path / "custom.jsonl").write_text(
        json.dumps(old) + "\n" + json.dumps(failure) + '\n{"runId":'
    )
    (tmp_path / "second.jsonl").write_text(
        "\n".join(
            [
                "null",
                "[]",
                json.dumps({"runId": "../bad", "event": "planned"}),
                json.dumps(new),
            ]
        )
        + "\n"
    )
    ledger = bench_live.LocalLedger.read()
    assert ledger.failures[old["runId"]] == failure
    assert ledger.replacement(old["runId"]) == new["runId"]
    assert "../bad" not in ledger.ids


def test_polling_watch_is_not_evicted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    monkeypatch.setattr(Watches, "start", lambda self, watch: None)
    old, _, new, *_ = replacement_rows()
    rows = [old, *[{**new, "runId": f"{i:032x}"} for i in range(8)]]
    (tmp_path / "custom.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    primary = Watch(old["runId"], tmp_path / "primary")
    watches = Watches(primary)
    for row in rows[1:-1]:
        with watches.acquire(row["runId"]):
            pass
    with primary.poll_lock:
        with watches.acquire(rows[-1]["runId"]) as watch:
            assert watch is not None
        assert not primary.stop.is_set()
        assert old["runId"] in watches.watches
        assert rows[1]["runId"] not in watches.watches
    watches.close()


OOM = "Runner killed (SIGKILL), exit code: 137. Most often this happens when the process runs out of memory."


@pytest.mark.parametrize("timeout", [False, True])
def test_app_log_reader_keeps_only_oom_stdout(
    monkeypatch: pytest.MonkeyPatch, timeout: bool
) -> None:
    from app.scripts import bench_live

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args == ["modal", "app", "logs", "ap-test"]
        assert kwargs["timeout"] == 20
        assert kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.DEVNULL
        text = f"unrelated log\n{OOM}\nnormal memory usage: 10 GB\nCUDA out of memory\n"
        if timeout:
            raise subprocess.TimeoutExpired(args, 20, output=text.encode(), stderr=b"ignored")
        return subprocess.CompletedProcess(args, 0, stdout=text)

    monkeypatch.setattr("app.scripts.bench_live.subprocess.run", run)
    assert bench_live.app_oom_lines("ap-test") == (OOM, "CUDA out of memory")


@pytest.mark.parametrize("outcome", ["oom", "timeout-oom", "timeout", "unavailable", "quiet"])
def test_failed_watch_reads_app_cause_once_outside_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, *_ = replacement_rows()
    failure = {**failure, "appId": "ap-test"}
    (tmp_path / "custom.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(failure) + "\n")
    calls = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if outcome == "unavailable":
            raise FileNotFoundError("no modal CLI")
        if outcome.startswith("timeout"):
            raise subprocess.TimeoutExpired(
                args, 20, output=OOM.encode() if outcome == "timeout-oom" else b""
            )
        return subprocess.CompletedProcess(
            args, 0, stdout=OOM if outcome == "oom" else "ordinary log"
        )

    monkeypatch.setattr("app.scripts.bench_live.subprocess.run", run)
    watch = Watch(old["runId"], tmp_path / "cache")
    request(watch, f"/api/v1/captures/{old['runId']}/live")
    assert not calls  # Neither constructing a watch nor serving HTTP runs the CLI.
    for _ in range(3):
        watch.poll(Volume())
    assert len(calls) == 1
    assert watch.state.train is not None and watch.state.train.ended is not None
    assert watch.state.train.ended.reason == (
        "oom" if outcome in {"oom", "timeout-oom"} else "failed"
    )
    assert watch.state.error == failure["error"]
    # The cache belongs to the server, so a recreated watch does not retry either.
    watches = Watches(watch)
    monkeypatch.setattr(Watches, "start", lambda self, watch: None)
    book = tmp_path / "custom.jsonl"
    replacements = [{**old, "runId": f"{i:032x}"} for i in range(8)]
    with book.open("a") as output:
        output.write("\n".join(json.dumps(row) for row in replacements) + "\n")
    for row in replacements:
        request(watches, f"/api/v1/captures/{row['runId']}/live")
    assert watch.stop.is_set()
    with watches.acquire(watch.run_id) as reopened:
        assert reopened is not None
        reopened.poll(Volume())
        assert reopened.state.train is not None and reopened.state.train.ended is not None
        assert reopened.state.train.ended.reason == watch.state.train.ended.reason
    assert len(calls) == 1
    watches.close()


def test_complete_watch_does_not_read_app_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, *_ = replacement_rows()
    failure = {**failure, "appId": "ap-test"}
    (tmp_path / "custom.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(failure) + "\n")
    monkeypatch.setattr(
        "app.scripts.bench_live.subprocess.run", lambda *a, **kw: pytest.fail("no log lookup")
    )
    volume = Volume()
    volume.files[f"runs/{old['runId']}/result.json"] = b'{"status":"complete"}'
    watch = Watch(old["runId"], tmp_path / "cache")
    for _ in range(3):
        watch.poll(volume)
    assert watch.state.status == "complete"


def test_volume_and_ledger_error_texts_both_reach_shared_end_reason() -> None:
    state = state_from_log(
        RUN,
        log_text(),
        {"status": "failed", "error": "restart refused"},
        failure={"event": "failed", "error": OOM},
    )
    assert state.train is not None and state.train.ended is not None
    assert state.train.ended.reason == "oom"


def test_real_replacement_chain_links_both_old_runs_to_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, middle, newest = replacement_rows()
    first = tmp_path / "first.jsonl"
    first.write_text("\n".join(json.dumps(r) for r in (old, failure, middle)) + "\n")
    second = tmp_path / "second.jsonl"
    second.write_text(json.dumps(newest) + "\n")
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    ledger = bench_live.LocalLedger.read()
    for source in (old, middle):
        assert ledger.replacement(source["runId"]) == newest["runId"]
        watch = Watch(source["runId"], tmp_path / source["runId"])
        watch.poll(Volume())
        assert watch.state.train is not None
        assert watch.state.train.replaced_by == newest["runId"]
    assert ledger.replacement(newest["runId"]) is None


def test_replacement_cycle_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, _, middle, newest = replacement_rows()
    # Same-file line order and cross-file timestamps can contradict one another:
    # old -> middle -> newest -> old. Never return a self-link or loop forever.
    rows = [
        {**old, "timestamp": "2026-11-06T03:00:00Z"},
        {**middle, "timestamp": "2026-11-06T01:00:00Z"},
    ]
    (tmp_path / "first.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "second.jsonl").write_text(
        json.dumps({**newest, "timestamp": "2026-11-06T02:00:00Z"}) + "\n"
    )
    ledger = bench_live.LocalLedger.read()
    assert all(ledger.replacement(row["runId"]) is None for row in (old, middle, newest))


@pytest.mark.parametrize("app_id", [None, "--not-an-app", 123])
def test_missing_or_invalid_app_id_is_failed_without_a_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_id: Any
) -> None:
    from app.scripts import bench_live

    monkeypatch.setattr(bench_live, "STATE", tmp_path)
    old, failure, *_ = replacement_rows()
    failure = {**failure, "appId": app_id}
    (tmp_path / "custom.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(failure) + "\n")
    watch = Watch(old["runId"], tmp_path / "cache")
    watch.poll(Volume())
    assert watch.state.train is not None and watch.state.train.ended is not None
    assert watch.state.train.ended.reason == "failed"


def test_an_app_shared_by_several_runs_is_never_read_for_one_runs_cause() -> None:
    from app.scripts.bench_live import LedgerRow, LocalLedger

    rows = [
        LedgerRow(
            {"runId": run, "event": "failed", "appId": "ap-shared", "error": "x"}, Path("l"), i
        )
        for i, run in enumerate(("a" * 32, "b" * 32))
    ]
    rows.append(LedgerRow({"runId": "c" * 32, "event": "failed", "appId": "ap-own"}, Path("l"), 2))
    ledger = LocalLedger(rows)
    assert ledger.app_runs["ap-shared"] == {"a" * 32, "b" * 32}
    assert ledger.app_runs["ap-own"] == {"c" * 32}
