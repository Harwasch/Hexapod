"""Real portable Swift recorder output must pass the server's bundle validator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import capture_bundle

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def swift_bundles(tmp_path_factory: pytest.TempPathFactory) -> Path:
    swift = shutil.which("swift")
    if swift is None:
        pytest.skip("Swift is not on PATH; see tools/pipeline/test_swift_bundle.sh")
    work = tmp_path_factory.mktemp("swift-bundle")
    package = work / "package"
    shutil.copytree(Path(__file__).with_name("swift_bundle"), package)
    # Compile the current App source, never a maintained harness reimplementation.
    # Staging permits ordinary SwiftPM target paths without symlinks outside the package.
    shutil.copyfile(
        REPO / "apps/ios/App/Support/BackgroundWriter.swift",
        package / "Sources/RecorderSupport/BackgroundWriter.swift",
    )
    output = work / "bundles"
    output.mkdir()
    environment = {
        **os.environ,
        "HEXAPOD_REPO_ROOT": str(REPO),
        "SWIFT_BUNDLE_OUT": str(output),
        "XDG_CACHE_HOME": str(work / "cache"),
        "HEAVY_CPUS": "4",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    command = [
        swift,
        "test",
        "-j",
        "4",
        "--package-path",
        str(package),
        "--scratch-path",
        str(work / "build"),
        "--cache-path",
        str(work / "cache"),
        "--config-path",
        str(work / "config"),
        "--security-path",
        str(work / "security"),
        "--disable-sandbox",
    ]
    if heavy := shutil.which("heavy"):
        command.insert(0, heavy)
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.mark.parametrize("seconds", [1, 18, 48])
@pytest.mark.parametrize("details", [False, True], ids=["control", "details"])
def test_real_swift_bundle(
    swift_bundles: Path, tmp_path: Path, seconds: int, details: bool
) -> None:
    name = f"{seconds}s-" + ("details" if details else "control")
    source = swift_bundles / name
    parsed = capture_bundle.read_bundle(source)
    unpacked = capture_bundle.unpack(swift_bundles / f"{name}.hexcap", tmp_path / "unpacked")
    assert unpacked.manifest == parsed.manifest
    assert unpacked.frames == parsed.frames
    assert len(parsed.frames) == parsed.manifest["frameCount"] == seconds * 15
    assert parsed.manifest["depthFrameCount"] == seconds * 15
    assert parsed.manifest["smoothedDepthFrameCount"] == seconds * 15
    assert len(parsed.locations) == 1
    inventory = {entry["path"] for entry in parsed.manifest["files"]}
    assert ("capture-info.json" in inventory) == details
    if details:
        expected = {"siteName": "Synthetic Linux test", "duration": seconds}
        assert json.loads((source / "capture-info.json").read_text()) == expected
        assert json.loads((unpacked.directory / "capture-info.json").read_text()) == expected
