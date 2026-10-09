"""Real pinned HunyuanWorld-Mirror inference, invoked only for an explicit job."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from geometry import (
    export_depth_mesh,
    export_gaussians,
    export_points,
    geometry_diagnostics,
    points_from_depth,
)
from sources import prepare_sources

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def update(job, stage, progress):
    temporary = job / "progress.tmp"
    temporary.write_text(json.dumps({"stage": stage, "progress": progress}))
    temporary.replace(job / "progress.json")


def infer(job: Path, source: Path, weights: Path, max_frames=12, target_size=518):
    update(job, "Preparing sources", 0.05)
    frames, source_diagnostics = prepare_sources(
        job / "inputs", job / "frames", max_frames, target_size
    )
    sys.path.insert(0, str(source))
    import torch
    from src.models.models.worldmirror import WorldMirror

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; no geometry was generated")
    model = (
        WorldMirror.from_pretrained(str(weights), local_files_only=True)
        .eval()
        .to("cuda")
    )
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to("cuda")
    use_amp = torch.cuda.is_bf16_supported()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update(job, "Reconstructing scene", 0.2)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp),
    ):
        predictions = model(
            views={"img": images}, cond_flags=[0, 0, 0], is_inference=True
        )
    torch.cuda.synchronize()
    duration = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated()

    def array(value):
        return value.detach().float().cpu().numpy()

    points = array(predictions["pts3d"][0])
    confidence = array(predictions["pts3d_conf"][0]).reshape(points.shape[:-1])
    depths = array(predictions["depth"][0])
    poses = array(predictions["camera_poses"][0])
    intrinsics = array(predictions["camera_intrs"][0])
    splats = predictions["splats"]
    update(job, "Exporting geometry", 0.8)
    output = job / "output"
    output.mkdir(exist_ok=True)
    gaussian_count = export_gaussians(
        output / "scene.ply",
        array(splats["means"][0]),
        array(splats["scales"][0]),
        array(splats["quats"][0]),
        array(splats["sh"][0]),
        array(splats["opacities"][0]),
    )
    point_count = export_points(output / "points.ply", points, frames, confidence)
    mesh_points = points_from_depth(depths, poses, intrinsics)
    depth_confidence = array(predictions["depth_conf"][0]).reshape(points.shape[:-1])
    triangles = export_depth_mesh(
        output / "mesh.glb", mesh_points, frames, depth_confidence
    )
    diagnostics = {
        **source_diagnostics,
        **geometry_diagnostics(points, confidence, depths, poses, intrinsics),
        "engine": MANIFEST["engine"],
        "sourceRevision": MANIFEST["sourceRevision"],
        "checkpointRevision": MANIFEST["checkpointRevision"],
        "generationSeconds": duration,
        "peakVRAMBytes": peak,
        "gaussianCount": gaussian_count,
        "pointCount": point_count,
        "meshTriangleCount": triangles,
        "meshKind": "unfused-predicted-depth-surfaces",
        "nativeCameraPriorsUsed": False,
        "priorNote": "Input pose conventions/intrinsics are not normalized by the current source contract; camera poses are estimated.",
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False)
    )
    (output / "cameras.json").write_text(
        json.dumps(
            {
                "convention": "OpenCV-camera-to-world",
                "units": "model-relative",
                "poses": poses.tolist(),
                "intrinsics": intrinsics.tolist(),
            },
            allow_nan=False,
        )
    )
    update(job, "Completed", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--target-size", type=int, default=518)
    args = parser.parse_args()
    # If the service is killed, do not leave paid inference orphaned indefinitely.
    parent = os.getppid()

    def parent_watch():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=parent_watch, daemon=True).start()
    try:
        infer(args.job, args.source, args.weights, args.max_frames, args.target_size)
    except Exception:  # noqa: BLE001 -- never expose upstream traces or source media
        # Never persist upstream exceptions: they may contain source paths or media.
        (args.job / "failure.json").write_text(
            json.dumps(
                {
                    "error": "Reconstruction failed. Check complete pinned weights, CUDA memory and overlapping source views."
                }
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
