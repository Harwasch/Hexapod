"""Disposable fixed solar processor. Inputs are typed data, never generated code."""

from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (96, 96))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    directory = Path(sys.argv[1])
    try:
        import httpx
        from pydantic import TypeAdapter

        from app.analysis.solar import analyze
        from app.schemas.geojson import Footprint
        from app.schemas.land_solar import SolarRequest

        payload = json.loads((directory / "input.json").read_text())
        boundary: Footprint = TypeAdapter(Footprint).validate_python(payload["boundary"])
        request = SolarRequest.model_validate(payload["request"])
        with httpx.Client(headers={"User-Agent": "LivingWorld-LandAnalysis/1.0"}) as client:
            result = analyze(boundary, request, client)
        (directory / "output.zip").write_bytes(result.data)
        (directory / "result.json").write_text(result.metadata.model_dump_json())
    except Exception as error:
        (directory / "result.json").write_text(json.dumps({"error": str(error)[:1000]}))


if __name__ == "__main__":
    main()
