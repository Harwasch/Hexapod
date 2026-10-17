from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path

from app.analysis.terrain import TerrainResult
from app.schemas.geojson import Footprint
from app.schemas.land_rasters import RasterMetadata, RasterRequest


class RasterCancelledError(Exception):
    pass


#: Time a research run keeps after its processor stops, to save the raster, its evidence,
#: finding and map before the run's time budget ends it.
SAVE_SECONDS = 15
#: With less processing time than this left in a run's budget, no raster is started.
MIN_SECONDS = 30


def wall_seconds(request: RasterRequest) -> float:
    """How long a processor may run before it is stopped. Sentinel-2 runs read many scenes
    through the network; four signature windows measured 97 s on 1,349 ac through a proxy.
    A raster run's time budget must cover this plus SAVE_SECONDS (web: rasterRuns.ts)."""
    return {"sentinel-2-signature": 200, "sentinel-2-ndvi": 165}.get(request.dataset, 90)


def run(
    boundary: Footprint,
    request: RasterRequest,
    cancelled: threading.Event | None = None,
    seconds: float | None = None,
) -> TerrainResult:
    """Run the processor in a sandboxed subprocess. `seconds` is the processing time left in
    the research run's budget: the processor stops at that or wall_seconds, whichever is
    sooner, so a run never ends mid-calculation claiming its outputs were kept."""
    own_limit = wall_seconds(request)
    limit = own_limit if seconds is None else min(seconds, own_limit)
    with tempfile.TemporaryDirectory(prefix="land-raster-") as temporary:
        directory = Path(temporary)
        (directory / "input.json").write_text(
            json.dumps(
                {"boundary": boundary.model_dump(), "request": request.model_dump(mode="json")}
            )
        )
        started = time.monotonic()
        with subprocess.Popen(  # noqa: S603 -- fixed processor, generated working directory, typed data only
            [sys.executable, "-m", "app.analysis.raster_job", str(directory)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as process:
            while process.poll() is None:
                if cancelled is not None and cancelled.is_set():
                    process.kill()
                    raise RasterCancelledError()
                if time.monotonic() - started > limit:
                    process.kill()
                    raise ValueError(
                        "Raster analysis reached its processing time limit. Try a smaller area."
                        if limit >= own_limit
                        else "This run's time budget ran out before the raster finished, so no "
                        "raster was saved. Start the analysis again from Explore this land, "
                        "which gives it a fresh time budget."
                    )
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.25)
        metadata_file, output = directory / "result.json", directory / "output.tif"
        if (
            process.returncode != 0
            or not metadata_file.is_file()
            or metadata_file.stat().st_size > 256 * 1024
        ):
            raise ValueError(
                "The raster processor exceeded its resource limits or could not finish."
            )
        payload = json.loads(metadata_file.read_text())
        if "error" in payload:
            raise ValueError(payload["error"])
        metadata = RasterMetadata.model_validate(payload)
        if not output.is_file() or output.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("The raster processor returned an invalid output file.")
        return TerrainResult(output.read_bytes(), metadata)
