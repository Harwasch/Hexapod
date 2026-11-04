"""Matched exports share frozen cameras; rehearsal needs neither torch nor Modal."""

import copy
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gsplat_stand_in import write_ply
from PIL import Image

from experiments.bench import __main__ as cli
from experiments.bench import ledger, matched_gpu, rehearsal
from experiments.bench.matched import source_run, sources
from experiments.bench.matched_maths import choose_frame, closeup_camera
from experiments.bench.scene import read_json, verify, write_json

A, B = "a" * 32, "b" * 32


def pair(state: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = state / "scenes/beach"
    rehearsal.synthetic_scene(scene, "beach", "test")
    manifest, split = verify(scene)
    for run in (A, B):
        directory = state / "runs" / run
        directory.mkdir(parents=True)
        write_ply(directory / "trained.ply", 32, 42)
        write_json(
            directory / "result.json",
            {
                "runId": run,
                "scene": "beach",
                "status": "complete",
                "rehearsal": True,
                "manifestDigest": manifest["digest"],
                "splitDigest": split["digest"],
                "trainParams": {"ship_sh_degree": 0},
                "test": {"psnr": 20},
            },
        )
    return manifest, split


def test_median_altitude_selection_and_test_only_override() -> None:
    names = ["high", "low", "mid-b", "mid-a"]
    centres = [[0, 0, 10], [0, 0, 0], [0, 0, 5], [0, 0, 5]]
    assert choose_frame(names, centres, [0, 0, 1], None) == "mid-a"
    assert choose_frame(names[::-1], centres[::-1], [0, 0, 1], None) == "mid-a"
    assert choose_frame(names, centres, [0, 0, 1], "high") == "high"
    with pytest.raises(ValueError, match="frozen test"):
        choose_frame(names, centres, [0, 0, 1], "train.png")


def test_closeup_has_same_pose_and_central_quarter_native_pixels() -> None:
    camera: dict[str, Any] = {
        "viewmat": np.eye(4).tolist(),
        "K": [[64, 0, 32], [0, 64, 24], [0, 0, 1]],
        "width": 64,
        "height": 48,
    }
    original = copy.deepcopy(camera)
    close, crop = closeup_camera(camera)
    assert camera == original and close["viewmat"] == camera["viewmat"]
    assert crop == (16, 12, 48, 36)
    assert (close["width"], close["height"]) == (32, 24)
    assert close["K"] == [[64, 0, 16], [0, 64, 12], [0, 0, 1]]
    np.testing.assert_array_equal(
        np.linalg.inv(close["K"]) @ [4, 7, 1], np.linalg.inv(camera["K"]) @ [20, 19, 1]
    )


def test_matched_rehearsal_returns_panels_and_accounts_for_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "rehearsal"
    manifest, split = pair(state)
    monkeypatch.setattr(cli, "REPO", tmp_path / "not-repo")
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("rehearsal imported Modal"))
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    assert (
        cli.main(
            [
                "matched-renders",
                "--scene",
                "beach",
                "--runs",
                A[:8],
                B[:8],
                "--rehearse",
                "--state-dir",
                str(tmp_path),
                "--tag",
                "quality-stack",
                "--total-cap",
                "30",
            ]
        )
        == 0
    )
    destination = state / "runs" / f"matched-{A[:8]}-{B[:8]}"
    assert set(p.name for p in destination.glob("*.png")) == set(matched_gpu.PNG_NAMES)
    result = read_json(destination / "result.json")
    assert result["status"] == "complete" and result["costUsd"] == 0 and result["rehearsal"]
    assert result["manifestDigest"] == manifest["digest"]
    cameras = read_json(destination / "cameras.json")
    assert cameras["frame"] in split["test"]
    gt = Image.open(state / "scenes/beach/frames" / cameras["frame"])
    np.testing.assert_array_equal(
        np.array(Image.open(destination / "GT-closeup.png")), np.array(gt.crop(cameras["crop"]))
    )
    assert cameras["cameras"]["ground"]["viewmat"] == cameras["cameras"]["closeup"]["viewmat"]
    entries = ledger.read(state / "ledger.jsonl")
    assert [e["event"] for e in entries] == ["planned", "finished"]
    assert entries[0]["schedule"] == "matched" and entries[0]["timeoutSeconds"] == 600
    assert entries[0]["tier"] == "l4"


@pytest.mark.parametrize(
    "change", ["manifest", "split", "scene", "missing", "failed", "degree", "rehearsal"]
)
def test_sources_refuse_unmatched_or_missing_exports(tmp_path: Path, change: str) -> None:
    manifest, _ = pair(tmp_path)
    path = tmp_path / "runs" / B / "result.json"
    result = read_json(path)
    if change == "missing":
        (path.parent / "trained.ply").unlink()
    elif change == "degree":
        result["trainParams"]["ship_sh_degree"] = 1
    else:
        key = {
            "manifest": "manifestDigest",
            "split": "splitDigest",
            "scene": "scene",
            "failed": "status",
            "rehearsal": "rehearsal",
        }[change]
        result[key] = False if change == "rehearsal" else "different"
    write_json(path, result)
    with pytest.raises(ValueError):
        sources(tmp_path / "runs", {"runs": [A, B], "scene": "beach", "rehearsal": True}, manifest)


def test_plain_old_baseline_and_faded_candidate_select_headline(tmp_path: Path) -> None:
    manifest, _ = pair(tmp_path)
    path = tmp_path / "runs" / B / "result.json"
    result = read_json(path)
    result.update(headlinePly="faded.ply", qualityGate="fade")
    write_json(path, result)
    write_ply(path.parent / "faded.ply", 32, 43)
    chosen = sources(
        tmp_path / "runs", {"runs": [A, B], "scene": "beach", "rehearsal": True}, manifest
    )
    assert [Path(s["ply"]).name for s in chosen] == ["trained.ply", "faded.ply"]
    (tmp_path / "runs" / ("a" * 31 + "b")).mkdir()
    with pytest.raises(ValueError, match="ambiguous"):
        source_run(tmp_path / "runs", A[:8])


def test_matched_cap_refuses_before_modal_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "REPO", tmp_path / "not-repo")
    monkeypatch.setattr(cli, "load_modal", lambda: pytest.fail("cap must run first"))
    monkeypatch.setattr(cli, "git_commit", lambda: "test")
    assert (
        cli.main(
            [
                "matched-renders",
                "--scene",
                "beach",
                "--runs",
                A,
                B,
                "--state-dir",
                str(tmp_path),
                "--cap",
                "0.01",
            ]
        )
        == 2
    )
    plan = cli.plans_for(
        cli.parser().parse_args(["matched-renders", "--scene", "beach", "--runs", A, B])
    )
    priced = ledger.price_plan([], plan, 2)
    assert priced["reservedUsd"] >= ledger.hourly_rate("l4") * 600 / 3600
    assert priced["plans"][0]["timeoutSeconds"] == 600


def test_gpu_entry_reuses_exact_cameras_and_crops_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Tensor:
        def __init__(self, data: Any) -> None:
            self.data = np.asarray(data)

        def numpy(self) -> Any:
            return self.data

    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    data = {
        "image": Tensor(pixels),
        "camtoworld": Tensor(np.eye(4)),
        "K": Tensor([[50, 0, 32], [0, 50, 24], [0, 0, 1]]),
    }

    class Parser:
        image_names = ["test.png"]

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["normalize"] is False

    class Dataset:
        indices = [0]

        def __init__(self, parser: Any, split: str) -> None:
            assert split == "val"

        def __getitem__(self, index: int) -> dict[str, Any]:
            return data

    module = types.ModuleType("datasets.colmap")
    monkeypatch.setattr(module, "Parser", Parser, raising=False)
    monkeypatch.setattr(module, "Dataset", Dataset, raising=False)
    monkeypatch.setitem(sys.modules, "datasets.colmap", module)
    calls = []

    def render(
        ply: Path, camera: dict[str, Any], output: Path, degree: int, antialiased: bool
    ) -> None:
        calls.append((ply.name, copy.deepcopy(camera), degree, antialiased))
        Image.new("RGB", (camera["width"], camera["height"])).save(output)

    monkeypatch.setattr(matched_gpu, "render", render)
    topdown = {
        "viewmat": np.eye(4).tolist(),
        "K": [[100, 0, 16], [0, 100, 16], [0, 0, 1]],
        "width": 32,
        "height": 32,
        "near": 1,
        "far": 100,
    }
    matched_gpu.render_pair(
        {
            "trainer": "trainer.py",
            "data": "data",
            "frame": "test.png",
            "topdown": topdown,
            "degree": 0,
            "sources": [{"ply": "old.ply"}, {"ply": "faded.ply"}],
        },
        tmp_path,
    )
    assert len(calls) == 6
    assert [c[1:] for c in calls[:3]] == [c[1:] for c in calls[3:]]
    assert calls[1][1] == topdown and all(not c[3] for c in calls)
    np.testing.assert_array_equal(
        np.array(Image.open(tmp_path / "GT-closeup.png")), pixels[12:36, 16:48]
    )
