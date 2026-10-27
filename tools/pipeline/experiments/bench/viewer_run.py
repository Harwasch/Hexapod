"""Offline static server + Python Playwright driver; also used by infra/modal/viewer_perf.py.

Playwright is an optional runtime dependency (1.55.0 in the Modal image). Importing this
module for unit tests requires only the standard library. No production API is contacted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .viewer import (
    frame_stats,
    gpu_stats,
    markdown,
    pacing_warning,
    relative_costs,
    require_renderer,
)

GPU_FLAGS = [
    "--no-sandbox",
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-webgpu",
]
SOFTWARE_FLAGS = [
    "--no-sandbox",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]
UNCAPPED_FLAGS = ["--disable-gpu-vsync", "--disable-frame-rate-limit"]


def inventory(directory: Path) -> dict[str, Any]:
    """Hash inputs and count the full-detail cut; support both sh_compare directory layouts."""
    directory = directory.resolve()
    if (directory / "splat/tileset.json").is_file():
        directory /= "splat"
    doc = json.loads((directory / "tileset.json").read_text())
    additive = doc["root"].get("refine", "REPLACE") == "ADD"
    selected: list[dict[str, Any]] = []
    contents: list[Path] = []

    def visit(tile: dict[str, Any]) -> None:
        uri = tile.get("content", {}).get("uri")
        if uri:
            parsed = urlsplit(uri)
            path = (directory / parsed.path).resolve()
            if parsed.scheme or parsed.netloc or not path.is_relative_to(directory):
                raise ValueError("Tiles must be local files inside their tileset directory")
            if path.suffix != ".glb" or not path.is_file():
                raise ValueError(f"Expected a packed GLB tile: {uri}")
            contents.append(path)
            if additive or not tile.get("children"):
                selected.append(tile)
        for child in tile.get("children", []):
            visit(child)

    visit(doc["root"])
    splats = sum(tile["extras"]["gaussians"] for tile in selected)
    if not selected or splats <= 0:
        raise ValueError("No full-detail gaussians")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return {
        "directory": str(directory),
        "splats": splats,
        "tiles": len(selected),
        "tile_bytes": sum(path.stat().st_size for path in set(contents)),
        "tileset_bytes": sum(path.stat().st_size for path in files),
        "sha256": hashes,
        "bounds": doc["root"]["boundingVolume"],
    }


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def serve(dist: Path, cases: dict[int, dict[str, Any]]) -> Iterator[str]:
    """One loopback origin, no API stubs needed by the bare renderer entry."""
    if not (dist / "viewer-perf.html").is_file():
        raise ValueError("Missing viewer-perf.html; build with VIEWER_PERF=1 pnpm build")
    with tempfile.TemporaryDirectory(prefix="viewer-static-") as temporary:
        root = Path(temporary)
        for path in dist.resolve().iterdir():
            (root / path.name).symlink_to(path, target_is_directory=path.is_dir())
        (root / "perf").mkdir()
        for degree, case in cases.items():
            (root / "perf" / f"sh{degree}").symlink_to(case["directory"], target_is_directory=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def smi_sample() -> dict[str, float]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi missing on hardware run")
    result = subprocess.run(  # noqa: S603
        [binary, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Expected exactly one GPU")
    memory, utilization = (float(value.strip()) for value in lines[0].split(","))
    return {
        "at_monotonic_s": time.monotonic(),
        "memory_mib": memory,
        "utilization_pct": utilization,
    }


async def measure_all(
    dist: Path,
    tiles: Path,
    *,
    software: bool = False,
    width: int = 1600,
    height: int = 900,
    warmup: int = 120,
    orbit: int = 450,
    dolly: int = 150,
    executable: str | None = None,
    timeout_s: int = 600,
    sync: bool = False,
) -> list[dict[str, Any]]:
    if min(width, height, timeout_s) <= 0 or warmup < 0 or orbit < 2 or dolly < 1:
        raise ValueError("Invalid viewport, frame count or timeout")
    cases = (
        {0: inventory(tiles)}
        if software
        else {degree: inventory(tiles / f"sh{degree}") for degree in range(4)}
    )
    for case in cases.values():
        if any(case[key] != cases[0][key] for key in ("splats", "tiles", "bounds")):
            raise ValueError("SH variants must have matching counts and bounds")
    # Optional dependency: no Playwright needed by pipeline tests or training images.
    api = importlib.import_module("playwright.async_api")
    flags = [*(SOFTWARE_FLAGS if software else GPU_FLAGS), *UNCAPPED_FLAGS]
    results = []
    with serve(dist, cases) as origin:
        async with api.async_playwright() as playwright:
            for degree, case in cases.items():
                # Browser restart also drops Chromium's GPU-process allocations/shader cache.
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=flags,
                    executable_path=executable,
                )
                try:
                    context = await browser.new_context(
                        viewport={"width": width, "height": height},
                        device_scale_factor=1,
                        is_mobile=False,
                        has_touch=False,
                        service_workers="block",
                    )
                    blocked: list[str] = []

                    async def local_only(route: Any, blocked: list[str] = blocked) -> None:
                        url = route.request.url
                        if url.startswith(origin + "/"):
                            await route.continue_()
                        else:
                            blocked.append(url)
                            await route.abort()

                    await context.route("**/*", local_only)
                    page = await context.new_page()
                    errors: list[str] = []
                    page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                    page.set_default_timeout(timeout_s * 1000)
                    await page.goto(origin + "/viewer-perf.html")
                    await page.wait_for_function("Boolean(window.viewerPerf)")
                    metadata = await asyncio.wait_for(
                        page.evaluate(
                            """async options => {
                          window.bench = await window.viewerPerf.prepare(options);
                          const {measure, destroy, ...metadata} = window.bench;
                          return metadata;
                        }""",
                            {
                                "tilesetUrl": f"{origin}/perf/sh{degree}/tileset.json",
                                "degree": degree,
                                "warmup": warmup,
                                "orbit": orbit,
                                "dolly": dolly,
                                "timeoutMs": timeout_s * 1000,
                                "sync": sync,
                            },
                        ),
                        timeout_s,
                    )
                    require_renderer(metadata["renderer"], software=software)
                    if metadata["splats"] != case["splats"] or metadata["tiles"] != case["tiles"]:
                        raise ValueError("Browser drew a different full-detail cut")
                    samples: list[dict[str, float]] = []
                    stop = asyncio.Event()

                    async def sample_gpu(
                        stop: asyncio.Event = stop,
                        samples: list[dict[str, float]] = samples,
                    ) -> None:
                        while not stop.is_set():
                            samples.append(await asyncio.to_thread(smi_sample))
                            with suppress(TimeoutError):
                                await asyncio.wait_for(stop.wait(), 0.5)

                    sampler = None if software else asyncio.create_task(sample_gpu())
                    try:
                        measured = await asyncio.wait_for(
                            page.evaluate("window.bench.measure()"),
                            timeout_s,
                        )
                    finally:
                        stop.set()
                        if sampler is not None:
                            await sampler
                    if errors or blocked:
                        raise RuntimeError(
                            f"Page errors: {errors}; blocked external requests: {blocked}"
                        )
                    timestamps = measured["timestamps"]
                    if len(timestamps) != orbit + dolly + 1:
                        raise ValueError("Incomplete frame measurement")
                    results.append(
                        {
                            **{k: v for k, v in case.items() if k != "directory"},
                            **frame_stats(timestamps),
                            **gpu_stats(samples),
                            "degree": degree,
                            "renderer": metadata["renderer"],
                            "backend": "playcanvas-webgl2",
                            "software": software,
                            "label": "software renderer, relative only"
                            if software
                            else "NVIDIA hardware",
                            "viewport": {"width": width, "height": height, "pixel_ratio": 1},
                            "flags": flags,
                            "timing": "serialized: gl.finish() in each frame callback"
                            if sync
                            else "pipelined rAF wall intervals",
                            "browser_version": browser.version,
                            "path": {
                                "orbit": orbit,
                                "dolly": dolly,
                                "warmup": warmup,
                                "start_radii": 3,
                                "end_radii": 0.15,
                                "elevation_deg": 20,
                            },
                            "timestamps_ms": timestamps,
                            "gpu_samples_raw": samples,
                            "gpu_ms": measured["gpuMs"],
                            "gpu_ms_note": measured["gpuNote"],
                            "pacing_warning": pacing_warning(timestamps),
                            "measured_at": datetime.now(UTC).isoformat(),
                            "browser_metadata": metadata,
                        }
                    )
                    await page.evaluate("window.bench.destroy()")
                finally:
                    await browser.close()
    return relative_costs(results)


def write_results(results: Sequence[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for result in relative_costs(results):
        (output / f"sh{result['degree']}.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "summary.md").write_text(markdown(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="SwiftShader smoke, SH0 only")
    parser.add_argument("--dist", type=Path, default=Path("apps/web/dist"))
    parser.add_argument("--tiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--orbit", type=int)
    parser.add_argument("--dolly", type=int)
    parser.add_argument("--executable")
    args = parser.parse_args()
    tiles = args.tiles or (Path("data/tiles/synthetic-tree") if args.local else None)
    if tiles is None:
        parser.error("--tiles is required for hardware runs")
    results = asyncio.run(
        measure_all(
            args.dist,
            tiles,
            software=args.local,
            width=args.width,
            height=args.height,
            warmup=args.warmup if args.warmup is not None else (10 if args.local else 120),
            orbit=args.orbit if args.orbit is not None else (12 if args.local else 450),
            dolly=args.dolly if args.dolly is not None else (4 if args.local else 150),
            executable=args.executable,
        )
    )
    write_results(results, args.output)


if __name__ == "__main__":
    main()
