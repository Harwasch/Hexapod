"""gsplat's compile as an image-build function. No imports beyond the standard library:
Modal imports this module in a container that does not have the pipeline yet."""

from __future__ import annotations

import os
import subprocess


def compile_gsplat(pip: str, source: str, archs: str) -> None:
    """Build gsplat from the checkout; `Image.run_function` gives this step its cores."""
    subprocess.run(  # noqa: S603
        [pip, "install", "--no-deps", "--no-build-isolation", source],
        check=True,
        env={
            **os.environ,
            "TORCH_CUDA_ARCH_LIST": archs,
            "CUDA_HOME": "/usr/local/cuda",
            "MAX_JOBS": "16",
            "BUILD_EXPERIMENTAL": "0",
        },
    )
