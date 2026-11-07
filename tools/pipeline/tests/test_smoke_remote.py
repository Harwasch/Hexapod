"""The remote smoke uses the real smoke entry point and guards production targets."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    class Image:
        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: self

    class App:
        def __init__(self, name: str) -> None:
            assert name != "twin-pipeline"

        def function(self, **kwargs: Any) -> Any:
            assert kwargs["secrets"] == ["twin-object-storage"]
            assert "gpu" not in kwargs
            return lambda fn: SimpleNamespace(local=fn, remote=fn)

        def local_entrypoint(self) -> Any:
            return lambda fn: fn

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            App=App, Secret=SimpleNamespace(from_name=lambda name: name), is_local=lambda: True
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "app",
        SimpleNamespace(cpu_image=Image(), LOCAL_PIPELINE=REPO / "tools" / "pipeline"),
    )
    return runpy.run_path(str(REPO / "infra" / "modal" / "smoke_remote.py"))


def test_remote_smoke_defaults_to_isolated_app_and_forwards_arguments(
    runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def smoke_main(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    def load(path: str) -> dict[str, Any]:
        assert path == "/opt/smoke/infra/modal/smoke.py"
        return {"main": smoke_main}

    monkeypatch.setattr(runpy, "run_path", load)
    runner["main"]()
    assert calls == [
        [
            "--app",
            "twin-pipeline-gsplat-upgrade-smoke",
            "--iterations",
            "500",
            "--tier",
            "l4",
            "--work",
            "/opt/smoke/work",
        ]
    ]


@pytest.mark.parametrize("entry", ["main", "run_smoke"])
def test_remote_smoke_refuses_production_unless_forced(
    runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    calls = []

    def smoke_main(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(runpy, "run_path", lambda path: {"main": smoke_main})
    function = runner[entry] if entry == "main" else runner[entry].local
    with pytest.raises(ValueError, match="force-production"):
        function(app_name="twin-pipeline")
    assert calls == []
    function(app_name="twin-pipeline", force_production=True, iterations=123, tier="a10")
    assert calls[0][:6] == ["--app", "twin-pipeline", "--iterations", "123", "--tier", "a10"]


def test_remote_smoke_assertion_failure_fails_the_call(
    runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runpy, "run_path", lambda path: {"main": lambda argv: 1})
    with pytest.raises(RuntimeError, match="exit status 1"):
        runner["main"]()


def test_remote_smoke_preserves_real_main_dispatch_and_failure_cleanup(
    runner: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from workdir import Workdir

    smoke = runpy.run_path(str(REPO / "infra/modal/smoke.py"))
    main = smoke["main"]
    deleted: list[str] = []
    apps: list[str] = []

    def fail(recipe: Any, work: Any, runners: Any) -> None:
        assert recipe.stages[0].gpu.tier == "cpu4"
        assert recipe.stages[0].params == {"matcher": "exhaustive"}
        assert recipe.stages[1].gpu.tier == "l4"
        import training

        params = recipe.stages[1].params
        assert params["test_every"] == 8
        assert params["converge"] is False
        assert params["batch_size"] == 1
        for frames in (4, 40, 120, 2500):
            assert training.frame_schedule(frames, params) == 1.0
        raise RuntimeError("trainer failure")

    monkeypatch.setattr(smoke["tree_frames"], "render_orbit", lambda *a, **kw: None)
    monkeypatch.setitem(
        main.__globals__, "Workdir", SimpleNamespace(create=lambda path: Workdir.create(tmp_path))
    )
    monkeypatch.setitem(
        main.__globals__, "_s3_transfer", lambda: SimpleNamespace(delete=deleted.append)
    )
    monkeypatch.setitem(main.__globals__, "ModalAdapter", apps.append)
    monkeypatch.setitem(main.__globals__, "execute", fail)
    monkeypatch.setattr(runpy, "run_path", lambda path: smoke)
    with pytest.raises(RuntimeError, match="trainer failure"):
        runner["main"]()
    assert apps == ["twin-pipeline-gsplat-upgrade-smoke"]
    assert len(deleted) == 1 and deleted[0].startswith("runs/smoke-")


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ("none", None),
        ("psnr", "no PSNR"),
        ("iterations", "trained 335 of 500 steps"),
        ("version", "expected gsplat 1.6.0"),
        ("resume", "resume checkpoints remain"),
    ],
)
def test_smoke_checks_training_contract(monkeypatch, tmp_path, capsys, change, failure):
    import json

    from gsplat_stand_in import write_ply

    smoke = runpy.run_path(str(REPO / "infra/modal/smoke.py"))
    main = smoke["main"]

    def completed(recipe, work, runners):
        work.prepare_stage("train")
        work.prepare_stage("package")
        metrics = {"iterations": 500, "psnr": 27.5, "source": "stats", "gsplatVersion": "1.6.0"}
        if change == "psnr":
            metrics["psnr"] = None
        elif change == "iterations":
            metrics["iterations"] = 335
        elif change == "version":
            metrics["gsplatVersion"] = "1.5.3"
        elif change == "resume":
            resume = work.checkpoint_dir("train") / "resume" / "main"
            resume.mkdir(parents=True)
            (resume / "step_500.pt").write_bytes(b"checkpoint")
        (work.out_dir("train") / "train_metrics.json").write_text(json.dumps(metrics))
        write_ply(work.out_dir("train") / "trained.ply", 2000, seed=0)
        tiles = work.out_dir("package") / "splat"
        tiles.mkdir()
        (tiles / "tileset.json").write_text("{}")

    monkeypatch.setattr(smoke["tree_frames"], "render_orbit", lambda *a, **kw: None)
    monkeypatch.setitem(
        main.__globals__, "_s3_transfer", lambda: SimpleNamespace(delete=lambda k: None)
    )
    monkeypatch.setitem(main.__globals__, "ModalAdapter", lambda app: None)
    monkeypatch.setitem(main.__globals__, "execute", completed)
    assert main(["--work", str(tmp_path)]) == (1 if failure else 0)
    captured = capsys.readouterr()
    if failure:
        assert failure in captured.err
    else:
        assert "FAILED" not in captured.err
