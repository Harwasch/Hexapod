"""nwm_import.py: NWM frames become JPEGs the pipeline's own readers place and date.

The check that matters is the round trip through `tools/pipeline`'s `exif.read_fix` and
`frame_meta.read_photo` -- what `georeference`, pose priors and the epoch's date read --
not a re-reading of the tags this module wrote. Two clips stand in for two seasons, one in
summer time and one not, so the offset is the zone's on each day rather than a constant.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

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


class FakeApi(nwm_import.Api):
    """The API as a dict of captures; records every call in order."""

    def __init__(self, files: list[dict[str, Any]] | None = None) -> None:
        super().__init__("https://api.invalid", "token")
        self.calls: list[tuple[str, str, Any]] = []
        self.files = files or []

    def call(self, method: str, path: str, body: object | None = None) -> Any:
        self.calls.append((method, path, body))
        if (method, path) == ("POST", "/captures"):
            return {"id": "c0ffee00-0000-0000-0000-000000000000", "files": []}
        if method == "GET":
            return {"id": path.rsplit("/", 1)[1], "files": self.files}
        if path.endswith("/files"):
            return {
                "file": {"id": f"file-{len(self.calls)}"},
                "upload": {
                    "partSize": 8 * 1024 * 1024,
                    "parts": [{"partNumber": 1, "url": "https://bucket.invalid/part"}],
                    "nextPartNumber": None,
                },
            }
        if path.endswith("/process"):
            return {"id": "job-1"}
        return {}


def test_upload_resumes_without_resending_and_queues_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "jpeg"
    manifest = nwm_import.prepare(_dataset(tmp_path, RECORDS), folder)
    put: list[int] = []
    monkeypatch.setattr(nwm_import, "_put", lambda url, data: put.append(len(data)) or '"etag"')

    first = manifest["files"][0]["name"]
    api = FakeApi(files=[{"filename": first, "status": "complete"}])
    result = nwm_import.upload(folder, api, capture_id="c0ffee00-0000-0000-0000-000000000001")
    assert result == {
        "captureId": "c0ffee00-0000-0000-0000-000000000001",
        "uploaded": 2,
        "alreadyThere": 1,
        "jobId": None,
    }
    registered = sorted(b["filename"] for m, p, b in api.calls if p.endswith("/files"))
    assert registered == sorted(f["name"] for f in manifest["files"][1:])
    assert not any(p.endswith("/process") for _, p, _ in api.calls)
    assert len(put) == 2
    completes = [b for _, p, b in api.calls if p.endswith("/complete")]
    assert all(c["parts"] == [{"partNumber": 1, "etag": '"etag"'}] for c in completes)
    assert all(c["checksum"].startswith("sha256:") for c in completes)

    fresh = FakeApi()
    result = nwm_import.upload(folder, fresh, process=True)
    assert fresh.calls[0][:2] == ("POST", "/captures")
    assert fresh.calls[-1] == (
        "POST",
        f"/captures/{result['captureId']}/process",
        {"recipe": "photo-reconstruct"},
    )
    assert result["uploaded"] == 3
    assert result["jobId"] == "job-1"


def test_upload_needs_the_token_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("API_WRITE_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        nwm_import.main(["upload", str(tmp_path)])
