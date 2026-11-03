"""Exercise the GPU evaluator with numpy stand-ins; these are plumbing, not quality scores."""

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gsplat_stand_in import write_ply
from PIL import Image

import gaussians
import harmonics
from experiments.bench.evaluate import evaluate
from holdout_error import read_ply, sh_rest, shipped_sh


class Tensor(np.ndarray[Any, Any]):
    def to(self, *args: Any) -> Any:
        return self

    def float(self) -> Any:
        return self.astype(np.float32)

    def exp(self) -> Any:
        return np.exp(self)

    def sigmoid(self) -> Any:
        return 1 / (1 + np.exp(-self))

    def permute(self, *axes: int) -> Any:
        return self.transpose(axes)

    def clamp(self, low: int, high: int) -> Any:
        return np.clip(self, low, high)

    def cpu(self) -> Any:
        return self

    def numpy(self) -> Any:
        return np.asarray(self)


def tensor(value: Any) -> Any:
    return np.asarray(value, dtype=np.float32).view(Tensor)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_all_degrees_same_frames_shipped_metrics_pngs_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degree: int
) -> None:
    raw = tmp_path / "raw.ply"
    write_ply(raw, 5, 42, 3)
    columns = read_ply(raw)
    ply = tmp_path / "trained.ply"
    shipped = {k: v for k, v in columns.items() if not k.startswith("f_rest_")}
    shipped.update(harmonics.truncate(columns, degree))
    gaussians.write_ply(ply, shipped, sh_degree=degree)
    names = ["b.jpg", "a.png"]
    # Dataset order deliberately differs from frozen order.
    frames = [tensor(np.full((4, 6, 3), v)) for v in (64, 128)]
    mask = tensor(np.ones((4, 6)))
    mask[0] = 0
    seen = []
    resets = []

    class Dataset:
        indices = [0, 1]

        def __init__(self, parser: Any, split: str) -> None:
            assert split == "val"

        def __getitem__(self, index: int) -> Any:
            return {
                "image": frames[index],
                "camtoworld": tensor(np.eye(4)),
                "K": tensor(np.eye(3)),
                "mask": mask,
            }

    class Metric:
        def __init__(self, **kwargs: Any) -> None:
            self.clean = False

        def to(self, device: Any) -> Any:
            return self

        def reset(self) -> None:
            self.clean = True
            resets.append(1)

        def __call__(self, prediction: Any, target: Any) -> Any:
            assert self.clean
            self.clean = False
            return np.mean((prediction - target) ** 2)

    def rasterization(**kwargs: Any) -> Any:
        d = kwargs["sh_degree"]
        dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None]
        expected, used = shipped_sh(dc, sh_rest(columns), d)
        assert used == d
        np.testing.assert_array_equal(kwargs["colors"], expected)
        assert kwargs["rasterize_mode"] == "antialiased"
        seen.append(d)
        return tensor(np.full((1, 4, 6, 3), (d + 1) / 5)), None, None

    modules = {
        "torch": SimpleNamespace(
            device=lambda d: d, from_numpy=tensor, no_grad=contextlib.nullcontext, linalg=np.linalg
        ),
        "datasets.colmap": SimpleNamespace(
            Dataset=Dataset, Parser=lambda **kwargs: SimpleNamespace(image_names=["a.png", "b.jpg"])
        ),
        "gsplat": SimpleNamespace(rasterization=rasterization),
        "torchmetrics.image": SimpleNamespace(
            PeakSignalNoiseRatio=Metric, StructuralSimilarityIndexMeasure=Metric
        ),
        "torchmetrics.image.lpip": SimpleNamespace(LearnedPerceptualImagePatchSimilarity=Metric),
    }
    for key, value in modules.items():
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    result = evaluate(
        tmp_path,
        ply,
        raw if degree < 3 else None,
        tmp_path / "trainer.py",
        names,
        degree,
        True,
        renders=1,
        render_dir=tmp_path / "renders",
    )
    assert result["shDegree"] == degree and result["fullShReason"] is None
    assert set(result["byDegree"]) == {"0", "1", "2", "3"}
    assert sorted(seen) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(resets) == 24
    for d in range(4):
        scores = result["byDegree"][str(d)]
        assert [row["name"] for row in scores["perFrame"]] == names
        assert scores["frames"] == 2
        for key in ("psnr", "ssim", "lpips"):
            assert scores[key] == pytest.approx(np.mean([r[key] for r in scores["perFrame"]]))
            if d == degree:
                assert scores[key] == result[key]
        with Image.open(tmp_path / "renders" / "b.jpg" / f"sh{d}.png") as image:
            pixels = np.asarray(image)
            assert image.size == (6, 4)
            assert np.all(pixels[0] == 0)  # same invalid-pixel mask as the metric
            assert np.all(pixels[1:] == round((d + 1) / 5 * 255))
    assert not (tmp_path / "renders" / "a.png").exists()
    with Image.open(tmp_path / "renders/b.jpg/gt.png") as image:
        assert np.all(np.asarray(image) == 128)
    assert result["psnrFullSh"] == result["byDegree"]["3"]["psnr"]
    assert result["psnrFullShGap"] == pytest.approx(result["psnrFullSh"] - result["psnr"])


def test_truncation_degree_two_is_channel_major_not_flat_prefix(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    write_ply(raw, 3, 42, 3)
    columns = read_ply(raw)
    dc = np.stack([columns[f"f_dc_{i}"] for i in range(3)], axis=1)[:, None]
    coefficients, used = shipped_sh(dc, sh_rest(columns), 2)
    kept = harmonics.truncate(columns, 2)
    assert used == 2
    for c in range(3):
        for k in range(8):
            np.testing.assert_array_equal(coefficients[:, k + 1, c], kept[f"f_rest_{c * 8 + k}"])
