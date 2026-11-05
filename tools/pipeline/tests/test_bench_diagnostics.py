from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import photometry_state
import quality_trainer
from experiments.bench import __main__ as cli
from experiments.bench.recipe import PIPELINE, patch_photometry
from experiments.bench.rescore_maths import aggregate_brightness, bottom_start, brightness
from experiments.bench.scene import read_json


def test_crashfix_native_no_fade_retains_ply_with_explicit_opt_out() -> None:
    argv = [
        "run",
        "--scene",
        "beach",
        "--schedule",
        "full",
        "--live",
        "--variants",
        str(PIPELINE / "experiments/variants/quality-stack-diagnostic.yaml"),
        "--variant",
        "crashfix",
    ]
    plan = cli.plans_for(cli.parser().parse_args(argv))[0]
    train = plan["trainParams"]
    assert plan["keepSplat"] is True and plan["live"] is True
    assert train["budget_max"] == 3456214
    assert "step_max_side" not in train and "max_side" not in train["priors"]
    assert train.get("train_max_side") is None
    assert train["aniso_cap"] == {"ratio": 10, "weight": 0.001}
    assert "_bench_quality" not in train
    assert train["ppisp_display_referred"] and not train["ppisp_use_controller"]
    assert not train["with_ut"] and not train["with_eval3d"]
    assert not cli.plans_for(cli.parser().parse_args([*argv, "--no-keep-splat"]))[0]["keepSplat"]


def test_brightness_excludes_invalid_and_aggregates_every_frame() -> None:
    gt = np.full((6, 4, 3), 0.4)
    pred = gt.copy()
    pred[4:] = 0.8
    valid = np.ones((6, 4), dtype=bool)
    valid[5, 3] = False
    pred[5, 3] = 1000
    start = bottom_start(len(gt))
    assert start == 4
    measurement = brightness(pred[start:], gt[start:], valid[start:])
    assert measurement["validPixels"] == 7
    assert measurement["renderMeanRgb"] == pytest.approx(204)
    assert measurement["gtMeanRgb"] == pytest.approx(102)
    second = brightness(gt[:1], gt[:1], valid[:1])
    rows = [{"brightness": {"raw": m, "aligned": m}} for m in (measurement, second)]
    mean = aggregate_brightness(rows)["raw"]
    assert mean["frames"] == 2 and mean["validPixels"] == 11
    assert mean["delta"] == pytest.approx(102 * 7 / 11)
    assert mean["frameMeanDelta"] == pytest.approx(51)
    with pytest.raises(ValueError, match="valid pixels"):
        brightness(pred, gt, np.zeros((6, 4), dtype=bool))


class Tensor:
    """Exercise serialization without loading the GPU environment locally."""

    dtype = "torch.float32"

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def detach(self) -> "Tensor":
        return self

    def cpu(self) -> "Tensor":
        return self

    def tolist(self) -> Any:
        return self.values.tolist()


def test_ppisp_sidecar_retains_checkpoint_state_and_dataset_frame_mapping(tmp_path: Path) -> None:
    runner = SimpleNamespace(
        cfg=SimpleNamespace(
            post_processing="ppisp", ppisp_use_controller=False, result_dir=tmp_path
        ),
        hex_settings={"ppisp_display_referred": True},
        parser=SimpleNamespace(image_names=["val.jpg", "train-a.jpg", "train-b.jpg"]),
        trainset=SimpleNamespace(indices=[2, 1]),
    )
    checkpoint = {
        "step": 199,
        "splats": object(),
        "post_processing": {
            "exposure_params": Tensor([0.1, -0.1]),
            "color_params": Tensor([[1, 2], [3, 4]]),
        },
    }
    photometry_state.save(runner, checkpoint)
    data = read_json(tmp_path / "post_processing.json")
    assert data["trainFrames"] == ["train-b.jpg", "train-a.jpg"]
    assert data["checkpointStep"] == 199 and data["displayReferred"]
    assert data["stateDict"]["color_params"] == {
        "dtype": "torch.float32",
        "shape": [2, 2],
        "values": [[1, 2], [3, 4]],
    }
    assert "splats" not in data
    assert not list(tmp_path.glob("*.tmp"))


def test_photometry_patch_coexists_with_quality_patch(tmp_path: Path) -> None:
    # Exercise both edits in production order on the exact anchors, without CUDA.
    anchor = (
        '                    data["post_processing"] = self.post_processing_module.state_dict()'
    )
    source = "\n".join((*quality_trainer.ANCHORS, "        # Losses & Metrics.", anchor))
    trainer = tmp_path / "simple_trainer.py"
    trainer.write_text(source)
    patched = patch_photometry(trainer)
    text = quality_trainer.patch_source(
        patched.read_text(), tmp_path / "settings.json", patched, display_referred=True
    )
    assert "_bench_photometry.save(self, data)" in text
    assert "_hex_quality.guard_scales(self)" in text
    assert "_hex_quality.configure_photometry(self)" in text
    trainer.write_text("moved")
    with pytest.raises(ValueError, match="checkpoint seam moved"):
        patch_photometry(trainer)
