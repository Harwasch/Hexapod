"""The normalize stage's ffmpeg choice, held to A0 #6's finding.

ffmpeg is not preinstalled on ubuntu-latest, so the normalize stage cannot assume a system
binary; `static-ffmpeg` is ruled out because it downloads binaries from GitHub at import,
which fails on a machine with no network. `imageio-ffmpeg` ships the binary inside the
wheel. This test fails the moment that stops being true -- before B2 builds on it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def test_a_static_ffmpeg_ships_with_the_wheel_and_runs() -> None:
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())

    assert exe.is_file()
    # Inside the installed package: no runtime download, no system ffmpeg.
    assert "imageio_ffmpeg" in exe.parts

    version = subprocess.run([str(exe), "-version"], capture_output=True, text=True, check=True)
    assert version.stdout.startswith("ffmpeg version")


def test_there_is_no_ffprobe_so_metadata_comes_from_ffmpeg_stderr() -> None:
    binaries = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent

    assert not list(binaries.glob("ffprobe*"))
