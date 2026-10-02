"""The two calls v1.5.3's `simple_trainer.py` makes of `imageio`, for the loop stand-in.

`eval()` writes each val frame's canvas with `imageio.imwrite(path, canvas)`, and
`render_traj` (off under `--disable_video`) opens `imageio.get_writer`. Beside the
stand-in trainer because the trainer's directory is first on `sys.path`, as it is for the
real one -- whose directory has no `imageio`, so there the real package is found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def imwrite(uri: Any, image: Any) -> None:
    Path(str(uri)).write_bytes(b"png" if not isinstance(image, bytes) else image)


def get_writer(uri: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("the loop stand-in renders no video")
