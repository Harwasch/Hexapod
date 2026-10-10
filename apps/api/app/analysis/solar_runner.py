from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

from app.analysis.solar import SolarResult
from app.schemas.geojson import Footprint
from app.schemas.land_solar import SolarMetadata, SolarRequest


class SolarCancelledError(Exception):
    pass


def run(
    boundary: Footprint, request: SolarRequest, cancelled: threading.Event | None = None
) -> SolarResult:
    with tempfile.TemporaryDirectory(prefix="land-solar-") as temporary:
        directory = Path(temporary)
        (directory / "input.json").write_text(
            json.dumps(
                {"boundary": boundary.model_dump(), "request": request.model_dump(mode="json")}
            )
        )
        started = time.monotonic()
        with subprocess.Popen(  # noqa: S603 -- fixed processor, generated working directory, typed data only
            [sys.executable, "-m", "app.analysis.solar_job", str(directory)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as process:
            while process.poll() is None:
                if cancelled is not None and cancelled.is_set():
                    process.kill()
                    raise SolarCancelledError()
                if time.monotonic() - started > 110:
                    process.kill()
                    raise ValueError(
                        "Solar analysis reached its processing time limit. Try again later."
                    )
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.25)
        metadata_file, output = directory / "result.json", directory / "output.zip"
        if (
            process.returncode != 0
            or not metadata_file.is_file()
            or metadata_file.stat().st_size > 256 * 1024
        ):
            raise ValueError(
                "The solar processor exceeded its resource limits or could not finish."
            )
        payload = json.loads(metadata_file.read_text())
        if "error" in payload:
            raise ValueError(payload["error"])
        metadata = SolarMetadata.model_validate(payload)
        if not output.is_file() or output.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("The solar processor returned an invalid output file.")
        return SolarResult(output.read_bytes(), metadata)
