"""Ephemeral viewer frame-cost bench: modal run, never deploy.

Set VIEWER_PERF_TILES to the directory containing sh0..sh3. add_local_dir(copy=False)
mounts that run's inputs at container startup, without baking data into the image or
creating persistent storage. The built dist defaults to apps/web/dist; override with
VIEWER_PERF_DIST. No secrets, volumes, API, or object storage are used.

Local smoke, without even importing Modal:
  PYTHONPATH=tools/pipeline python infra/modal/viewer_perf.py --local --output <new-dir>
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent

# The same driver as the cloud run, with software explicitly required and labelled.
if __name__ == "__main__" and "--local" in sys.argv:
    sys.path.insert(0, str(ROOT / "tools/pipeline"))
    from experiments.bench.viewer_run import main as smoke

    smoke()
    raise SystemExit(0)

import modal  # noqa: E402

app = modal.App("hexapod-viewer-perf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    # The system EGL (glvnd) and Vulkan loaders: without them Chromium's GPU process
    # cannot reach the NVIDIA driver Modal mounts ("Extension not supported:
    # VK_KHR_surface") and silently falls back to SwiftShader.
    .apt_install("libegl1", "libvulkan1")
    .pip_install("playwright==1.55.0")
    .run_commands("playwright install --with-deps chromium")
    .env({"PYTHONPATH": "/root/pipeline", "OMP_NUM_THREADS": "4"})
)
if modal.is_local():
    tiles = os.environ.get("VIEWER_PERF_TILES")
    if not tiles:
        raise ValueError(
            "Set VIEWER_PERF_TILES to the directory holding sh0/ through sh3/"
        )
    dist = Path(
        os.environ.get("VIEWER_PERF_DIST", str(ROOT / "apps/web/dist"))
    ).expanduser()
    if not (dist / "viewer-perf.html").is_file():
        raise ValueError("Build first: VIEWER_PERF=1 pnpm build")
    image = (
        image.add_local_dir(dist, "/root/dist", copy=False)
        .add_local_dir(Path(tiles).expanduser(), "/root/tiles", copy=False)
        .add_local_dir(
            ROOT / "tools/pipeline/experiments/bench",
            "/root/pipeline/experiments/bench",
            ignore=["__pycache__", "*.pyc"],
            copy=False,
        )
    )


@app.function(
    image=image,
    gpu="T4",
    cpu=(4, 4),
    memory=(8192, 8192),
    # 20 min caps a run at ~$0.26 (T4 + 4 cores): it is not on the bench ledger.
    timeout=1200,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def measure(
    width: int, height: int, warmup: int, orbit: int, dolly: int, sync: bool = True
) -> list[dict[str, Any]]:
    from experiments.bench.viewer_run import measure_all

    return asyncio.run(
        measure_all(
            Path("/root/dist"),
            Path("/root/tiles"),
            width=width,
            height=height,
            warmup=warmup,
            orbit=orbit,
            dolly=dolly,
            sync=sync,
        )
    )


@app.local_entrypoint()
def main(
    gpu: str = "T4",
    width: int = 1600,
    height: int = 900,
    warmup: int = 120,
    orbit: int = 450,
    dolly: int = 150,
    output: str = "",
    pipelined: bool = False,
) -> None:
    """Only local result files persist after the ephemeral app exits."""
    if gpu not in {"T4", "L4"}:
        raise ValueError("--gpu must be T4 or L4")
    if min(width, height) <= 0 or warmup < 0 or orbit < 2 or dolly < 1:
        raise ValueError("Invalid viewport or frame count")
    sys.path.insert(0, str(ROOT / "tools/pipeline"))
    from experiments.bench.viewer_run import inventory, write_results

    # Fail before provisioning a GPU for incomplete inputs or an existing output path.
    tiles = Path(os.environ["VIEWER_PERF_TILES"]).expanduser()
    cases = [inventory(tiles / f"sh{d}") for d in range(4)]
    if any(
        any(c[key] != cases[0][key] for key in ("splats", "tiles", "bounds"))
        for c in cases
    ):
        raise ValueError("SH variants must have matching counts and bounds")
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = (
        Path(output).expanduser()
        if output
        else Path.home() / ".local/share/hexapod-bench/viewer" / run
    )
    if target.exists():
        raise FileExistsError(target)
    # Serialized by default: pipelined uncapped intervals on a T4 came out shorter than the
    # GPU's own time per render, so they are not a cost (run 20261027T003316).
    results = measure.with_options(gpu=gpu).remote(
        width, height, warmup, orbit, dolly, not pipelined
    )
    write_results(results, target)
    print(target / "summary.md")
