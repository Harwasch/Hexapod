"""Turn one NWM timestep into a photo capture the pipeline reads as it is, and upload it.

NWM, "No Wolf in the Meadow" (TU Wien, CC BY 4.0, doi:10.48436/bphp1-hbb20), is drone
video of a 100 x 200 m flood-control site in Vienna, filmed on 8 days over 6.5 months and
exported as 1600 x 900 PNG frames. It is the first public data with two dates of one
place, which is what alignment (M3) and the change map (M5) need to be tried on.

Its frames carry no EXIF. Each frame's GPS is in the timestep's `camera_info.json`
instead, and `georeference` (`exif_gps`) and pose priors read EXIF. Rather than teach the
pipeline a sidecar for one dataset, this writes what the JSON says into JPEG copies:

* **position**: `latitude`, `longitude` and `abs_alt` as `GPSLatitude` / `GPSLongitude` /
  `GPSAltitude` (ref 0, above sea level), datum WGS-84. `abs_alt` is the drone's own
  figure and is not consistent between days: the 2025-08-25 timestep flies 20 m above
  take-off at 317 m, the 2025-11-22 one at 156 m. It is written as given -- a real drone
  photo set is no better -- and the run records what the pipeline made of it.
* **time**: the JSON has none per frame. A DJI file name carries the clip's start in the
  drone's local clock (`DJI_20250829153742_0593_D.MP4`), and `video_frame` is the frame's
  index in that clip, so the frame was taken at start + index / fps. 29.97 fps is DJI's
  "30p" (the index steps by 119.8 every 4 s, which fits it and not 30.000); Europe/Vienna
  is the site's zone. Both are assumptions, and `--fps` / `--timezone` override them.
  Written as `DateTimeOriginal` + `SubSecTimeOriginal` + `OffsetTimeOriginal`, the tags
  `frame_meta.exif_time` reads first.
* **nothing else**: no make, model or focal length. The data does not state them, and a
  guessed focal would be a claim. COLMAP estimates it; the capture says `drone`, which the
  pipeline records as `claimed`.

Note that a timestep's label is not always its recording day: the 2025-08-25 timestep's
clips are all named 2025-08-29. The capture is named by the label (that is how the dataset
is cited) and dated by the clips (that is when it was filmed).

Usage::

    uv run python nwm_import.py prepare ~/.cache/nwm/raw/nwm_2025-08-25 ~/.cache/nwm/jpeg/2025-08-25
    uv run python nwm_import.py upload ~/.cache/nwm/jpeg/2025-08-25 --api https://twin-api.fly.dev \\
        [--process] [--capture-id <this timestep's NWM capture, to resume>] [--report out.json]

`upload` reads the write token from `API_WRITE_TOKEN` and never from an argument, so it
stays out of process listings and shell history. `.github/workflows/import-nwm.yml` runs
both steps on a runner, where the token is a repository secret.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import PIL.Image
from PIL.TiffImagePlugin import IFDRational

DOI = "10.48436/bphp1-hbb20"
RECORD_URL = "https://researchdata.tuwien.ac.at/records/bphp1-hbb20"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
CREDIT = (
    "No Wolf in the Meadow (NWM) dataset by Tobias Batik, Diana Marin, Peter Kán and "
    f"Hannes Kaufmann, TU Wien, CC BY 4.0, doi:{DOI}"
)
#: DJI's "30p".
DEFAULT_FPS = 30000 / 1001
DEFAULT_TIMEZONE = "Europe/Vienna"
JPEG_QUALITY = 95
MANIFEST = "nwm_manifest.json"
USER_AGENT = "hexapod-nwm-import/1"
DEFAULT_API = "https://twin-api.fly.dev"

#: `DJI_<YYYYMMDDhhmmss>_<sequence>_D.MP4`, with or without NWM's flight-plan prefix.
DJI_CLIP = re.compile(r"DJI_(\d{14})_(\d{4})_")
#: A frame's file name as camera_info.json may give it: no folders, no control characters.
PLAIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png")

#: EXIF tag numbers, spelled out as `tools/pipeline/exif.py` does.
_IMAGE_DESCRIPTION, _SOFTWARE, _ARTIST, _COPYRIGHT = 270, 305, 315, 33432
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825
_DATETIME_ORIGINAL, _OFFSET_ORIGINAL, _SUBSEC_ORIGINAL = 36867, 36881, 37521
_GPS_VERSION, _LAT_REF, _LAT, _LON_REF, _LON, _ALT_REF, _ALT, _MAP_DATUM = 0, 1, 2, 3, 4, 5, 6, 18


@dataclass(frozen=True)
class Frame:
    """One NWM frame, and the JPEG it becomes."""

    source: str
    name: str
    lat: float
    lon: float
    alt: float | None
    taken: datetime
    split: str | None


def clip_start(src_video: str, zone: ZoneInfo) -> tuple[datetime, str]:
    """The clip's start in the drone's local clock, made aware, and its DJI sequence."""
    match = DJI_CLIP.search(src_video)
    if match is None:
        raise ValueError(f"{src_video!r} is not a DJI clip name (DJI_<14 digits>_<4 digits>_)")
    start = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=zone)
    return start, match.group(2)


def read_frames(dataset: Path, *, fps: float, zone: ZoneInfo) -> list[Frame]:
    """Every frame `camera_info.json` lists, in the order it was filmed.

    A frame the JSON names but the archive lacks is an error, not a skip: the record is
    the dataset's own and a gap means the archive is not the one it describes.
    """
    records = json.loads((dataset / "camera_info.json").read_text())
    frames = []
    for record in records:
        image = str(record["image"])
        # The JSON is archive content: a name that is not one plain file name could reach
        # outside input/.
        if not PLAIN_NAME.fullmatch(image):
            raise ValueError(f"camera_info.json names {image!r}, not a plain file name")
        source = dataset / "input" / image
        if not source.is_file():
            raise FileNotFoundError(f"camera_info.json lists {record['image']}, not in input/")
        start, sequence = clip_start(record["src_video"], zone)
        index = int(record["video_frame"])
        # To the millisecond, which is all `SubSecTimeOriginal` carries here.
        taken = start + timedelta(milliseconds=round(index / fps * 1000))
        frames.append(
            Frame(
                source=record["image"],
                # Time order is name order: the pipeline windows a photo set by name.
                name=f"DJI_{start:%Y%m%d_%H%M%S}_{sequence}_f{index:05d}.jpg",
                lat=float(record["latitude"]),
                lon=float(record["longitude"]),
                alt=None if record.get("abs_alt") is None else float(record["abs_alt"]),
                taken=taken,
                split=record.get("type"),
            )
        )
    frames.sort(key=lambda f: (f.taken, f.name))
    names = [f.name for f in frames]
    if len(set(names)) != len(names):
        raise ValueError("two frames map to one file name")
    return frames


def _dms(value: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    """Degrees as EXIF's three rationals; seconds to 1e-6", about 3 cm on the ground."""
    value = abs(value)
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = Fraction((value - degrees - minutes / 60) * 3600).limit_denominator(1_000_000)
    return (
        IFDRational(degrees, 1),
        IFDRational(minutes, 1),
        IFDRational(seconds.numerator, seconds.denominator),
    )


def exif_for(frame: Frame, timestep: str) -> PIL.Image.Exif:
    """The EXIF block `frame`'s JPEG carries: GPS, time, credit, and nothing invented."""
    exif = PIL.Image.Exif()
    exif[_IMAGE_DESCRIPTION] = f"NWM timestep {timestep}, frame {frame.source}"
    exif[_ARTIST] = "Batik, Marin, Kán, Kaufmann (TU Wien)"
    exif[_COPYRIGHT] = f"CC BY 4.0, doi:{DOI}"
    exif[_SOFTWARE] = "hexapod tools/captures/nwm_import.py"
    local = frame.taken
    offset = local.utcoffset() or timedelta(0)
    sign = "-" if offset < timedelta(0) else "+"
    hours, rest = divmod(abs(int(offset.total_seconds())), 3600)
    exif_ifd = exif.get_ifd(_EXIF_IFD)
    exif_ifd[_DATETIME_ORIGINAL] = f"{local:%Y:%m:%d %H:%M:%S}"
    exif_ifd[_SUBSEC_ORIGINAL] = f"{local.microsecond // 1000:03d}"
    exif_ifd[_OFFSET_ORIGINAL] = f"{sign}{hours:02d}:{rest // 60:02d}"
    gps = exif.get_ifd(_GPS_IFD)
    gps[_GPS_VERSION] = b"\x02\x03\x00\x00"
    gps[_LAT_REF] = "S" if frame.lat < 0 else "N"
    gps[_LAT] = _dms(frame.lat)
    gps[_LON_REF] = "W" if frame.lon < 0 else "E"
    gps[_LON] = _dms(frame.lon)
    if frame.alt is not None:
        gps[_ALT_REF] = b"\x01" if frame.alt < 0 else b"\x00"
        altitude = Fraction(abs(frame.alt)).limit_denominator(1000)
        gps[_ALT] = IFDRational(altitude.numerator, altitude.denominator)
    gps[_MAP_DATUM] = "WGS-84"
    return exif


def prepare(
    dataset: Path,
    out: Path,
    *,
    fps: float = DEFAULT_FPS,
    timezone: str = DEFAULT_TIMEZONE,
    limit: int | None = None,
    timestep: str | None = None,
) -> dict[str, Any]:
    """Write every frame of `dataset` (an unzipped `nwm_<timestep>/`) to `out` as JPEG,
    and `out/nwm_manifest.json`, which is what `upload` sends and what the summary reads."""
    zone = ZoneInfo(timezone)
    timestep = timestep or dataset.name.removeprefix("nwm_")
    frames = read_frames(dataset, fps=fps, zone=zone)
    if limit is not None:
        frames = frames[:limit]
    if not frames:
        raise ValueError(f"{dataset} lists no frames")
    out.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        with PIL.Image.open(dataset / "input" / frame.source) as image:
            image.convert("RGB").save(
                out / frame.name,
                "JPEG",
                quality=JPEG_QUALITY,
                exif=exif_for(frame, timestep),
            )
    first, last = frames[0].taken, frames[-1].taken
    altitudes = [f.alt for f in frames if f.alt is not None]
    manifest: dict[str, Any] = {
        "dataset": "NWM",
        "doi": DOI,
        "timestep": timestep,
        "fps": fps,
        "timezone": timezone,
        "frames": len(frames),
        "firstLocal": first.isoformat(timespec="milliseconds"),
        "firstUtc": _utc(first),
        "lastUtc": _utc(last),
        "recordedOn": sorted({f.taken.date().isoformat() for f in frames}),
        "medianLat": statistics.median(f.lat for f in frames),
        "medianLon": statistics.median(f.lon for f in frames),
        "altitudeM": (
            {"min": min(altitudes), "median": statistics.median(altitudes), "max": max(altitudes)}
            if altitudes
            else None
        ),
        "files": [
            {
                "name": f.name,
                "source": f.source,
                "split": f.split,
                "taken": f.taken.isoformat(timespec="milliseconds"),
            }
            for f in frames
        ],
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _utc(moment: datetime) -> str:
    utc = moment.astimezone(ZoneInfo("UTC"))
    return utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def capture_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """`POST /captures`: named and credited as NWM, located for the site join.

    `metadata.lat` / `lon` are the frames' median fix: a photo set has no video location,
    so this is the point the worker matches a site by before the run
    (`epochs.located_or_metadata`). Without it every timestep would make its own site.
    """
    timestep = manifest["timestep"]
    recorded = ", ".join(manifest["recordedOn"])
    return {
        "name": f"NWM test (TU Wien, CC BY 4.0) · {timestep}",
        "slug": slug_for(timestep),
        "description": (
            f"Test data: timestep {timestep} of the public NWM drone dataset, filmed "
            f"{recorded}. {CREDIT}. {manifest['frames']} video frames, converted from PNG "
            "to JPEG with each frame's GPS (camera_info.json) and time (clip name + frame "
            f"index at {manifest['fps']:.3f} fps, {manifest['timezone']}) written into "
            "EXIF by tools/captures/nwm_import.py. Not a Hexapod capture."
        ),
        "kind": "images",
        "capturedAt": manifest["firstUtc"],
        "metadata": {
            "lat": manifest["medianLat"],
            "lon": manifest["medianLon"],
            "platform": "drone",
            "dataset": {"name": "NWM", "doi": DOI, "timestep": timestep},
        },
        "attribution": [{"text": CREDIT, "organization": "TU Wien", "url": RECORD_URL}],
        "license": {
            "name": "Creative Commons Attribution 4.0 International",
            "spdxId": "CC-BY-4.0",
            "url": LICENCE_URL,
            "requiresAttribution": True,
            "notes": "Frames re-encoded as JPEG with GPS and time written into EXIF.",
        },
        "provenance": {
            "sourceOrganization": "TU Wien",
            "sourceUrl": RECORD_URL,
            "notes": f"nwm_{timestep}.zip from doi:{DOI}",
        },
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: urllib would follow it with the Authorization header on."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)
#: Tries for a GET; a write is sent once.
GET_ATTEMPTS = 4
#: How long `wake` waits for a cold API. Measured 2026-10-23: 0.2-6.9 s (DEPLOYMENT.md,
#: "Cold start"); a run's first POST timed out at 120 s on one, so the budget is generous.
WAKE_BUDGET_S = 60.0


class _Transient(Exception):
    """A failure worth another try: a timeout, a dropped connection, a 5xx."""


class Api:
    """The handful of API calls an upload makes, over urllib, with the token in a header.

    A GET is retried with backoff on a timeout, a dropped connection or a 5xx: a read is
    safe to repeat, and a Fly API waking from idle answers the first request late. Nothing
    else is ever sent twice -- a POST that timed out may still have created its row.
    `wake` is what keeps the first write from meeting a cold API.

    A failure says the method, path, status and the Problem's `title`, never the body: a
    body is the server's to word and is not repeated into a public run log.
    """

    def __init__(self, base: str, token: str | None, *, timeout_s: float = 120.0) -> None:
        self.base = base.rstrip("/") + "/api/v1"
        self.token = token
        self.timeout_s = timeout_s

    def call(
        self,
        method: str,
        path: str,
        body: object | None = None,
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
    ) -> Any:
        tries = attempts or (GET_ATTEMPTS if method == "GET" else 1)
        for attempt in range(tries):
            try:
                return self._send(method, path, body, timeout_s or self.timeout_s)
            except _Transient as error:
                if attempt == tries - 1:
                    raise RuntimeError(f"{method} {path}: {error}") from None
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def wake(self, budget_s: float = WAKE_BUDGET_S, try_timeout_s: float = 15.0) -> None:
        """Wait until `GET /health` answers, for up to `budget_s`: the API (and the database
        behind it) may be asleep, and the first write should not be what wakes it."""
        deadline = time.monotonic() + budget_s
        while True:
            try:
                self.call("GET", "/health", timeout_s=try_timeout_s, attempts=1)
                return
            except RuntimeError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"the API did not answer /health within {budget_s:g} s ({error})"
                    ) from None
            time.sleep(2)

    def _send(self, method: str, path: str, body: object | None, timeout_s: float) -> Any:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with _OPENER.open(request, timeout=timeout_s) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            if error.code >= 500:
                raise _Transient(f"HTTP {error.code}{_title(error)}") from None
            raise RuntimeError(f"{method} {path}: HTTP {error.code}{_title(error)}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise _Transient(type(error).__name__) from None


def _title(error: urllib.error.HTTPError) -> str:
    try:
        title = json.loads(error.read() or b"null").get("title")
    except (ValueError, AttributeError):
        return ""
    return f" ({str(title)[:120]})" if title else ""


def _put(url: str, payload: bytes, attempts: int = 4) -> str:
    """PUT one part to its presigned URL and return the ETag S3 answered with."""
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=payload, method="PUT")
        try:
            with _OPENER.open(request, timeout=120) as response:
                etag = response.headers.get("ETag")
                if not etag:
                    raise RuntimeError("the storage answered a part PUT with no ETag")
                return str(etag)
        except urllib.error.HTTPError as error:
            # A storage 5xx is worth another try; a 4xx (an expired or refused URL) is not.
            # The URL carries its signature, so only the status is said.
            if error.code < 500 or attempt == attempts - 1:
                raise RuntimeError(f"part PUT: HTTP {error.code}{_title(error)}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"part PUT: {type(error).__name__}") from None
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def slug_for(timestep: str) -> str:
    """The capture's slug: `nwm-tuwien-<timestep>`, `_` as `-` (`2025-12-04_1` -> `-1`)."""
    return f"nwm-tuwien-{timestep}".lower().replace("_", "-")


#: An upload row that is still open, so it is resumed rather than registered again. Only
#: `in-progress`: the API presigns parts for no other status (`presign_parts`).
OPEN_UPLOADS = ("in-progress",)


def resumable(capture: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Refuse to upload into any capture but this timestep's NWM one.

    `--capture-id` exists to finish an upload that stopped part-way; it must not become a
    way to add files to someone's real capture, so the slug and the dataset record both
    have to be the ones `capture_payload` wrote.
    """
    expected = slug_for(manifest["timestep"])
    dataset = (capture.get("metadata") or {}).get("dataset") or {}
    if capture.get("slug") != expected or dataset.get("doi") != DOI:
        raise ValueError(
            f"capture {capture.get('id')} is not this timestep's NWM capture (slug "
            f"{capture.get('slug')!r}, want {expected!r} with metadata.dataset.doi {DOI}); "
            "nothing was uploaded"
        )


def upload_file(api: Api, capture_id: str, path: Path, open_row: str | None = None) -> str:
    """Upload and complete one file; every NWM JPEG is a single part.

    `open_row` is a row this filename already has whose upload never completed: its parts
    are presigned again and the same row is completed, so a resume registers nothing twice.
    """
    payload = path.read_bytes()
    if open_row is None:
        started = api.call(
            "POST",
            f"/captures/{capture_id}/files",
            {"filename": path.name, "contentType": "image/jpeg", "bytes": len(payload)},
        )
        file_id, window = started["file"]["id"], started["upload"]
    else:
        file_id = open_row
        window = api.call(
            "POST", f"/captures/{capture_id}/files/{file_id}/parts", {"firstPartNumber": 1}
        )
    size = int(window["partSize"])
    parts = []
    for part in window["parts"]:
        number = int(part["partNumber"])
        chunk = payload[(number - 1) * size : number * size]
        parts.append({"partNumber": number, "etag": _put(part["url"], chunk)})
    if window.get("nextPartNumber") is not None:
        raise RuntimeError(f"{path.name} needs more than one window of parts")
    api.call(
        "POST",
        f"/captures/{capture_id}/files/{file_id}/complete",
        {"parts": parts, "checksum": "sha256:" + hashlib.sha256(payload).hexdigest()},
    )
    return file_id


def upload(
    folder: Path,
    api: Api,
    *,
    capture_id: str | None = None,
    process: bool = False,
    workers: int = 8,
    report: Callable[[dict[str, Any]], None] = lambda _: None,
) -> dict[str, Any]:
    """Create the capture (or resume `capture_id`, if it is this timestep's NWM capture),
    upload every JPEG the manifest names that it does not hold complete, and queue
    `photo-reconstruct` if asked.

    `report` hears the capture id as soon as there is one, so a run that fails part-way
    still says which capture to resume. Any file's failure fails the upload, and nothing
    is queued after one.
    """
    manifest = json.loads((folder / MANIFEST).read_text())
    api.wake()
    if capture_id is None:
        capture = api.call("POST", "/captures", capture_payload(manifest))
        capture_id = str(capture["id"])
    else:
        capture = api.call("GET", f"/captures/{capture_id}")
        resumable(capture, manifest)
    report({"captureId": capture_id, "resumed": bool(capture.get("files"))})
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in capture.get("files", []):
        rows.setdefault(str(row.get("filename")), []).append(row)
    done = {name for name, found in rows.items() if any(r["status"] == "complete" for r in found)}
    reopen = {
        name: str(next(r["id"] for r in found if r["status"] in OPEN_UPLOADS))
        for name, found in rows.items()
        if name not in done and any(r["status"] in OPEN_UPLOADS for r in found)
    }
    todo = [folder / f["name"] for f in manifest["files"] if f["name"] not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(upload_file, api, capture_id, path, reopen.get(path.name)) for path in todo
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    job = None
    if process:
        job = api.call("POST", f"/captures/{capture_id}/process", {"recipe": "photo-reconstruct"})
    result = {
        "captureId": capture_id,
        "uploaded": len(todo),
        "resumedRows": len(reopen),
        "alreadyThere": len(done),
        "jobId": None if job is None else job["id"],
    }
    report(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="unzipped nwm_<timestep>/ to EXIF'd JPEGs")
    prep.add_argument("dataset", type=Path)
    prep.add_argument("out", type=Path)
    prep.add_argument("--fps", type=float, default=DEFAULT_FPS)
    prep.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    prep.add_argument("--limit", type=int, help="the first N frames only (a trial)")
    prep.add_argument("--timestep", help="the timestep's label; default: the folder's name")
    send = commands.add_parser("upload", help="create the capture and upload the JPEGs")
    send.add_argument("folder", type=Path)
    send.add_argument("--api", default=DEFAULT_API)
    send.add_argument("--capture-id", help="resume an upload into this capture")
    send.add_argument("--process", action="store_true", help="queue photo-reconstruct after")
    send.add_argument("--report", type=Path, help="keep the capture id and outcome here (JSON)")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(
            args.dataset,
            args.out,
            fps=args.fps,
            timezone=args.timezone,
            limit=args.limit,
            timestep=args.timestep,
        )
        summary = {k: v for k, v in manifest.items() if k != "files"}
    else:
        token = os.environ.get("API_WRITE_TOKEN")
        if not token:
            parser.error("upload needs API_WRITE_TOKEN in the environment")

        def report(state: dict[str, Any]) -> None:
            print(f"capture {state['captureId']}", flush=True)
            if args.report is not None:
                args.report.write_text(json.dumps(state) + "\n")

        summary = upload(
            args.folder,
            Api(args.api, token),
            capture_id=args.capture_id,
            process=args.process,
            report=report,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
