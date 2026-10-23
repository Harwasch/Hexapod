"""nwm_import.py: NWM frames become JPEGs the pipeline's own readers place and date.

The check that matters is the round trip through `tools/pipeline`'s `exif.read_fix` and
`frame_meta.read_photo` -- what `georeference`, pose priors and the epoch's date read --
not a re-reading of the tags this module wrote. Two clips stand in for two seasons, one in
summer time and one not, so the offset is the zone's on each day rather than a constant.
"""

from __future__ import annotations

import http.server
import json
import re
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import PIL.Image
import pytest

import nwm_import

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

import exif
import frame_meta

#: The API's `SLUG_RE` (apps/api app/schemas/capture.py).
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

RECORDS: list[dict[str, Any]] = [
    # Listed out of time order, as camera_info.json is (by flight-plan prefix).
    {
        "image": "006_DJI_20250829153742_0593_D_2.png",
        "video_frame": 180,
        "src_video": "006_DJI_20250829153742_0593_D.MP4",
        "latitude": 48.206436,
        "longitude": 16.2306,
        "rel_alt": 20.0,
        "abs_alt": 317.406,
        "type": "train",
    },
    {
        "image": "009_DJI_20250829152012_0571_D_1.png",
        "video_frame": 60,
        "src_video": "009_DJI_20250829152012_0571_D.MP4",
        "latitude": 48.206304,
        "longitude": 16.231049,
        "rel_alt": 30.0,
        "abs_alt": 326.732,
        "type": "test",
    },
    {
        "image": "DJI_20251122152217_0042_D_1.png",
        "video_frame": 1498,
        "src_video": "DJI_20251122152217_0042_D.MP4",
        "latitude": -33.8568,
        "longitude": -151.2153,
        "rel_alt": 16.8,
        "abs_alt": 152.774,
        "type": "train",
    },
]


def _dataset(root: Path, records: list[dict[str, Any]]) -> Path:
    dataset = root / "nwm_2025-08-25"
    (dataset / "input").mkdir(parents=True)
    for index, record in enumerate(records):
        PIL.Image.new("RGB", (32, 18), (40 * index, 120, 60)).save(
            dataset / "input" / record["image"]
        )
    (dataset / "camera_info.json").write_text(json.dumps(records))
    return dataset


def test_frames_carry_the_fix_and_time_the_pipeline_reads(tmp_path: Path) -> None:
    out = tmp_path / "jpeg"
    manifest = nwm_import.prepare(_dataset(tmp_path, RECORDS), out)

    names = [f["name"] for f in manifest["files"]]
    assert names == [
        "DJI_20250829_152012_0571_f00060.jpg",
        "DJI_20250829_153742_0593_f00180.jpg",
        "DJI_20251122_152217_0042_f01498.jpg",
    ]
    assert names == sorted(names), "the pipeline windows a photo set by name: name = time"

    expected = {
        names[0]: (48.206304, 16.231049, 326.732, "2025-08-29T13:20:14.002Z", "+02:00"),
        names[1]: (48.206436, 16.2306, 317.406, "2025-08-29T13:37:48.006Z", "+02:00"),
        names[2]: (-33.8568, -151.2153, 152.774, "2025-11-22T14:23:06.983Z", "+01:00"),
    }
    for name, (lat, lon, alt, utc, offset) in expected.items():
        fix = exif.read_fix(out / name)
        assert fix is not None
        assert fix.lat == pytest.approx(lat, abs=1e-8)
        assert fix.lon == pytest.approx(lon, abs=1e-8)
        assert fix.alt == pytest.approx(alt, abs=1e-3)
        record = frame_meta.read_photo(
            out / name, frame_name=name, upload_root=out, frame_size=(32, 18)
        )
        assert record.unreadable is None
        assert record.time.source == "exif-offset"
        assert record.time.offset == offset
        assert record.time.utc == utc
        # Nothing the data does not state: no make, so no platform is read off the files.
        assert record.make is None

    assert manifest["firstUtc"] == "2025-08-29T13:20:14.002Z"
    assert manifest["recordedOn"] == ["2025-08-29", "2025-11-22"]
    assert manifest["frames"] == 3
    assert (out / nwm_import.MANIFEST).is_file()


def test_a_frame_the_json_lists_but_the_archive_lacks_fails(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, RECORDS)
    (dataset / "input" / RECORDS[0]["image"]).unlink()
    with pytest.raises(FileNotFoundError, match=RECORDS[0]["image"]):
        nwm_import.prepare(dataset, tmp_path / "jpeg")


def test_a_clip_that_is_not_named_by_dji_fails_rather_than_guessing_a_time(
    tmp_path: Path,
) -> None:
    records = [{**RECORDS[0], "src_video": "flight_one.MP4"}]
    with pytest.raises(ValueError, match="not a DJI clip name"):
        nwm_import.prepare(_dataset(tmp_path, records), tmp_path / "jpeg")


def test_the_capture_is_located_credited_and_named_for_the_dataset(tmp_path: Path) -> None:
    manifest = nwm_import.prepare(_dataset(tmp_path, RECORDS[:2]), tmp_path / "jpeg")
    payload = nwm_import.capture_payload(manifest)

    assert payload["name"] == "NWM test (TU Wien, CC BY 4.0) · 2025-08-25"
    assert SLUG_RE.match(payload["slug"])
    assert payload["kind"] == "images"
    assert payload["capturedAt"] == "2025-08-29T13:20:14.002Z"
    # The site join's point for a photo set (epochs.located_or_metadata).
    assert payload["metadata"]["lat"] == pytest.approx(48.20637)
    assert payload["metadata"]["lon"] == pytest.approx(16.2308245)
    assert payload["metadata"]["platform"] == "drone"
    assert payload["license"]["spdxId"] == "CC-BY-4.0"
    assert "TU Wien" in payload["attribution"][0]["text"]
    assert "CC BY 4.0" in payload["description"]
    assert "2025-08-29" in payload["description"]


RESUME_ID = "c0ffee00-0000-0000-0000-000000000001"
NWM_CAPTURE = {"slug": "nwm-tuwien-2025-08-25", "metadata": {"dataset": {"doi": nwm_import.DOI}}}


class FakeApi(nwm_import.Api):
    """The API as one capture; records every call in order."""

    def __init__(
        self, files: list[dict[str, Any]] | None = None, capture: dict[str, Any] | None = None
    ) -> None:
        super().__init__("https://api.invalid", "token")
        self.calls: list[tuple[str, str, Any]] = []
        self.files = files or []
        self.capture = NWM_CAPTURE if capture is None else capture

    def call(self, method: str, path: str, body: object | None = None) -> Any:
        self.calls.append((method, path, body))
        window = {
            "partSize": 8 * 1024 * 1024,
            "parts": [{"partNumber": 1, "url": "https://bucket.invalid/part"}],
            "nextPartNumber": None,
        }
        if (method, path) == ("POST", "/captures"):
            return {"id": "c0ffee00-0000-0000-0000-000000000000", "files": []}
        if method == "GET":
            return {**self.capture, "id": path.rsplit("/", 1)[1], "files": self.files}
        if path.endswith("/files"):
            return {"file": {"id": f"file-{len(self.calls)}"}, "upload": window}
        if path.endswith("/parts"):
            return window
        if path.endswith("/process"):
            return {"id": "job-1"}
        return {}


@pytest.fixture
def puts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    sent: list[int] = []
    monkeypatch.setattr(nwm_import, "_put", lambda url, data: sent.append(len(data)) or '"etag"')
    return sent


def test_upload_resumes_without_resending_and_queues_only_when_asked(
    tmp_path: Path, puts: list[int]
) -> None:
    folder = tmp_path / "jpeg"
    manifest = nwm_import.prepare(_dataset(tmp_path, RECORDS), folder)
    first, second, third = (f["name"] for f in manifest["files"])
    api = FakeApi(
        files=[
            {"id": "done", "filename": first, "status": "complete"},
            # An upload that stopped before completing: resumed on its own row.
            {"id": "open", "filename": second, "status": "in-progress"},
            # One that failed for good: registered again, its row left as a record.
            {"id": "gone", "filename": third, "status": "aborted"},
        ]
    )
    reports: list[dict[str, Any]] = []
    result = nwm_import.upload(folder, api, capture_id=RESUME_ID, report=reports.append)
    assert result == {
        "captureId": RESUME_ID,
        "uploaded": 2,
        "resumedRows": 1,
        "alreadyThere": 1,
        "jobId": None,
    }
    assert reports[0] == {"captureId": RESUME_ID, "resumed": True}
    assert reports[-1] == result
    registered = [b["filename"] for _, p, b in api.calls if p.endswith("/files")]
    assert registered == [third]
    assert ("POST", f"/captures/{RESUME_ID}/files/open/parts", {"firstPartNumber": 1}) in api.calls
    assert not any(p.endswith("/process") for _, p, _ in api.calls)
    assert len(puts) == 2
    completes = [(p, b) for _, p, b in api.calls if p.endswith("/complete")]
    assert len(completes) == 2
    assert f"/captures/{RESUME_ID}/files/open/complete" in [p for p, _ in completes]
    assert all(b["parts"] == [{"partNumber": 1, "etag": '"etag"'}] for _, b in completes)
    assert all(b["checksum"].startswith("sha256:") for _, b in completes)

    fresh = FakeApi()
    reports.clear()
    result = nwm_import.upload(folder, fresh, process=True, report=reports.append)
    assert fresh.calls[0][:2] == ("POST", "/captures")
    # The id is reported before any file goes up, so a failed run still names it.
    assert reports[0] == {"captureId": result["captureId"], "resumed": False}
    assert fresh.calls[-1] == (
        "POST",
        f"/captures/{result['captureId']}/process",
        {"recipe": "photo-reconstruct"},
    )
    assert result["uploaded"] == 3
    assert result["jobId"] == "job-1"


@pytest.mark.parametrize(
    "capture",
    [
        {"slug": "pumpkin", "metadata": {}},
        {"slug": "nwm-tuwien-2025-08-25", "metadata": {}},
        {"slug": "nwm-tuwien-2025-11-22", "metadata": NWM_CAPTURE["metadata"]},
    ],
)
def test_resume_refuses_any_capture_but_this_timesteps_nwm_one(
    tmp_path: Path, puts: list[int], capture: dict[str, Any]
) -> None:
    folder = tmp_path / "jpeg"
    nwm_import.prepare(_dataset(tmp_path, RECORDS), folder)
    api = FakeApi(capture=capture)
    with pytest.raises(ValueError, match="nothing was uploaded"):
        nwm_import.upload(folder, api, capture_id=RESUME_ID, process=True)
    assert [m for m, _, _ in api.calls] == ["GET"]
    assert puts == []


def test_a_failed_file_fails_the_upload_and_queues_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "jpeg"
    nwm_import.prepare(_dataset(tmp_path, RECORDS), folder)

    def refuse(url: str, data: bytes) -> str:
        raise RuntimeError("part PUT: HTTP 403")

    monkeypatch.setattr(nwm_import, "_put", refuse)
    api = FakeApi()
    reports: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError, match="HTTP 403"):
        nwm_import.upload(folder, api, process=True, report=reports.append)
    assert reports == [{"captureId": "c0ffee00-0000-0000-0000-000000000000", "resumed": False}]
    assert not any(p.endswith("/process") for _, p, _ in api.calls)


def test_split_timesteps_get_a_valid_slug() -> None:
    assert nwm_import.slug_for("2025-12-04_1") == "nwm-tuwien-2025-12-04-1"
    assert SLUG_RE.match(nwm_import.slug_for("2026-01-10_detail_2"))


def test_a_frame_name_that_leaves_input_is_refused(tmp_path: Path) -> None:
    records = [{**RECORDS[0], "image": "../camera_info.png"}]
    dataset = _dataset(tmp_path / "a", [RECORDS[0]])
    (dataset / "camera_info.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="not a plain file name"):
        nwm_import.prepare(dataset, tmp_path / "jpeg")


def test_frames_are_encoded_at_the_stated_quality(tmp_path: Path) -> None:
    """Decoded pixels equal a reference encode at quality 95: a lower setting shows."""
    dataset = _dataset(tmp_path, RECORDS[:1])
    source = dataset / "input" / RECORDS[0]["image"]
    noise = PIL.Image.effect_noise((64, 36), 60).convert("RGB")
    noise.save(source)
    manifest = nwm_import.prepare(dataset, tmp_path / "jpeg")
    reference = tmp_path / "reference.jpg"
    noise.save(reference, "JPEG", quality=95)
    assert nwm_import.JPEG_QUALITY >= 90
    with (
        PIL.Image.open(tmp_path / "jpeg" / manifest["files"][0]["name"]) as ours,
        PIL.Image.open(reference) as theirs,
    ):
        assert ours.tobytes() == theirs.tobytes()


class _Server(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[tuple[str, str | None]]] = []

    def _answer(self) -> None:
        self.seen.append((self.path, self.headers.get("Authorization")))
        if self.path == "/api/v1/captures":
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()
            return
        body = json.dumps({"title": "Conflict", "detail": "echo of secret-body"}).encode()
        self.send_response(409)
        self.send_header("Content-Type", "application/problem+json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _answer

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Server.seen = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def test_a_redirect_is_refused_and_the_token_goes_nowhere_else(server: str) -> None:
    api = nwm_import.Api(server, "test-only-token")
    with pytest.raises(RuntimeError, match="HTTP 302"):
        api.call("POST", "/captures", {})
    assert _Server.seen == [("/api/v1/captures", "Bearer test-only-token")]


def test_an_error_names_its_status_and_title_but_not_its_body(server: str) -> None:
    api = nwm_import.Api(server, "test-only-token")
    with pytest.raises(RuntimeError) as raised:
        api.call("GET", "/captures/x")
    assert str(raised.value) == "GET /captures/x: HTTP 409 (Conflict)"
    assert "secret-body" not in str(raised.value)


def test_upload_needs_the_token_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("API_WRITE_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        nwm_import.main(["upload", str(tmp_path)])
