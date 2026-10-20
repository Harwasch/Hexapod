"""The by-hand entry point. A7's worker calls execute_recipe directly, not this."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_recipe import main


def test_plan_only_validates_without_running_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["photo-reconstruct", "--workdir", str(tmp_path / "run"), "--plan-only"])

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["gpuStages"] == ["pose", "align", "train", "quality"]
    assert not (tmp_path / "run").exists()


def test_photo_reconstruct_is_given_an_empty_site_context_unless_one_is_seeded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import alignment
    from run_recipe import seed_empty_site_context

    target = seed_empty_site_context(tmp_path / "run")
    document = json.loads((target / alignment.SITE_CONTEXT_FILE).read_text())
    assert document == alignment.empty_site_context()
    # One already there (a seeded or resumed run) is never overwritten.
    (target / alignment.SITE_CONTEXT_FILE).write_text('{"version": 1, "siteId": "kept"}')
    seed_empty_site_context(tmp_path / "run")
    assert json.loads((target / alignment.SITE_CONTEXT_FILE).read_text())["siteId"] == "kept"


def test_a_run_seeds_its_inputs_and_prints_its_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    upload = tmp_path / "capture.ply"
    upload.write_bytes(b"pretend splat")

    exit_code = main(
        [
            "splat-ingest",
            "--workdir",
            str(tmp_path / "run"),
            "--runner",
            "stub",
            "--seed",
            f"upload={upload}",
        ]
    )

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert [artifact["name"] for artifact in document["artifacts"]][-1] == "registration.json"


def test_a_broken_recipe_exits_non_zero_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["no-such-recipe", "--workdir", str(tmp_path / "run")])

    assert exit_code == 1
    assert "unknown recipe 'no-such-recipe'" in capsys.readouterr().err
