"""Run the production-path smoke with Modal's storage secret; never deploy.

    modal run infra/modal/smoke_remote.py --app-name twin-pipeline-gsplat-upgrade-smoke

Only this CPU coordinator belongs to the ephemeral app. Pose and train still call the
target deployment through CloudRunner/ModalAdapter; pose uses its apt COLMAP 3.9.1.
The shared smoke pins held-out evaluation and an exact training step count.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from app import LOCAL_PIPELINE, cpu_image  # noqa: E402

REMOTE_ROOT = Path("/opt/smoke")
REPO = HERE.parents[1] if modal.is_local() else REMOTE_ROOT
DEFAULT_TARGET = "twin-pipeline-gsplat-upgrade-smoke"

# smoke.py locates the pipeline, renderer and fixture relative to its repository root.
# Reuse the CPU image's pipeline/captures instead of shipping a second copy. The app
# module must also be next to this runner when Modal imports it under /root.
image = (
    cpu_image.pip_install("modal==1.5.5")
    .run_commands(
        "mkdir -p /opt/smoke/tools",
        "ln -s /opt/pipeline /opt/smoke/tools/pipeline",
        "ln -s /opt/captures /opt/smoke/tools/captures",
    )
    .add_local_file(HERE / "app.py", "/root/app.py")
    .add_local_file(HERE / "app.py", str(REMOTE_ROOT / "infra/modal/app.py"))
    .add_local_file(HERE / "smoke.py", str(REMOTE_ROOT / "infra/modal/smoke.py"))
    .add_local_file(LOCAL_PIPELINE / "tests/tree_frames.py", "/opt/pipeline/tests/tree_frames.py")
    .add_local_file(
        REPO / "data/tiles/synthetic-tree/source/splat.ply",
        str(REMOTE_ROOT / "data/tiles/synthetic-tree/source/splat.ply"),
    )
)
app = modal.App("twin-pipeline-smoke-runner")


def _check_target(app_name: str, force_production: bool) -> None:
    if app_name == "twin-pipeline" and not force_production:
        raise ValueError("refusing production twin-pipeline; pass --force-production to override")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("twin-object-storage")],
)
def run_smoke(
    app_name: str = DEFAULT_TARGET,
    iterations: int = 500,
    tier: str = "l4",
    force_production: bool = False,
) -> None:
    _check_target(app_name, force_production)
    # Stay in the Modal process: a subprocess loses modal.is_local()'s container context.
    smoke = runpy.run_path(str(REMOTE_ROOT / "infra/modal/smoke.py"))
    status = smoke["main"](
        [
            "--app",
            app_name,
            "--iterations",
            str(iterations),
            "--tier",
            tier,
            "--work",
            str(REMOTE_ROOT / "work"),
        ]
    )
    if status:
        raise RuntimeError(f"production-path smoke failed with exit status {status}")


@app.local_entrypoint()
def main(
    app_name: str = DEFAULT_TARGET,
    iterations: int = 500,
    tier: str = "l4",
    force_production: bool = False,
) -> None:
    _check_target(app_name, force_production)
    run_smoke.remote(app_name, iterations, tier, force_production)
