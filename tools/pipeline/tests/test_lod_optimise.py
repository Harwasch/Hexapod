"""Optimised level-of-detail parents: the numpy half, the stages around it, and -- where an
interpreter with torch is available -- the torch half on a CPU.

**Nothing here runs on a GPU.** What is tested:

* `lod_maths`, which `lod_optimise.py` runs unchanged on the GPU box: the viewer's cut
  (CesiumJS REPLACE traversal on screen-space error), H3DGS's log-uniform tau, the
  iteration plan, the camera placement, H3DGS's position schedule and scene radius, and
  the frustum test;
* `place` writing `placement.json`, the similarity it applied, exactly;
* the `optimise_lod` -> `package` hand-over with a stand-in script
  (`lod_parents_stand_in.py`) that builds the tree with the packer's own code and writes
  keyed, fingerprinted parents: `package` draws them; a failed, rejected, disabled or
  mismatched optimisation leaves the merged parents and the run intact;
* the shipped recipe: `optimise_lod` between `place` and `package`, on an L4, with
  `package`'s packing parameters;
* `lod_optimise.py` itself, end to end on a tiny scene, with a torch reference rasteriser
  standing in for gsplat's CUDA one (tests/lod_stand_in/) -- only when `$LOD_TORCH_PYTHON`
  names an interpreter with torch (the trainer's CPU replica does), since this project's
  own environment has none.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import gaussians
import lod_maths
import lod_parents
import splat_tiles
import stages
from conftest import make_recipe
from executor import execute
from recipe import load_recipe
from runners import LocalRunner, RunnerSet
from workdir import Workdir

HERE = Path(__file__).resolve().parent
STAND_IN = HERE / "lod_parents_stand_in.py"
TORCH_STAND_IN = HERE / "lod_stand_in"
BUDGET = 400


# --- a hand-made tree --------------------------------------------------------------------


def _tile(uri: str, low: list[float], high: list[float], error: float, kids: list[Any]) -> Any:
    return SimpleNamespace(
        uri=uri, low=np.array(low), high=np.array(high), geometric_error=error, children=kids
    )


def _walk(tile: Any) -> list[Any]:
    out = [tile]
    for child in tile.children:
        out.extend(_walk(child))
    return out


def small_tree() -> lod_maths.Tree:
    """root (error 1) -> [a (error 0.25) -> [a0, a1], b]; boxes along x."""
    a0 = _tile("a0", [0, 0, 0], [1, 1, 1], 0.0, [])
    a1 = _tile("a1", [1, 0, 0], [2, 1, 1], 0.0, [])
    a = _tile("a", [0, 0, 0], [2, 1, 1], 0.25, [a0, a1])
    b = _tile("b", [2, 0, 0], [4, 1, 1], 0.0, [])
    root = _tile("root", [0, 0, 0], [4, 1, 1], 1.0, [a, b])
    tiles = _walk(root)
    return lod_maths.tree_from(tiles, [10, 5, 30, 30, 40])


def test_the_tree_lays_leaves_first_then_parents() -> None:
    tree = small_tree()
    assert tree.uris == ("root", "a", "a0", "a1", "b")
    assert tree.parents == (0, 1) and tree.leaves == (2, 3, 4)
    # Leaves 30 + 30 + 40 from row 0, then the parents' 10 + 5.
    assert tree.rows.tolist() == [[100, 110], [110, 115], [0, 30], [30, 60], [60, 100]]


def test_the_cut_is_cesiums_replace_traversal() -> None:
    """Refine a tile when geometricError * f / distance-to-box > tau; REPLACE by all
    children, each of which decides again. Checked against the rule written out."""
    tree = small_tree()
    focal = 100.0
    eye = np.array([-10.0, 0.5, 0.5])  # 10 from the root's and a's boxes
    # root: 1 * 100 / 10 = 10 px; a: 0.25 * 100 / 10 = 2.5 px.
    assert lod_maths.screen_space_error(tree, 0, eye, focal) == pytest.approx(10.0)
    assert lod_maths.cut(tree, eye, focal, 16.0) == [0]
    assert lod_maths.cut(tree, eye, focal, 10.0) == [0]  # not over tau: drawn
    assert lod_maths.cut(tree, eye, focal, 5.0) == [1, 4]
    assert lod_maths.cut(tree, eye, focal, 2.0) == [2, 3, 4]
    # A camera inside a box is infinite error: always refined.
    inside = np.array([1.0, 0.5, 0.5])
    assert lod_maths.cut(tree, inside, focal, 1e9) == [2, 3, 4]


def test_tau_is_log_uniform_between_its_bounds() -> None:
    """H3DGS Sec. 5.1: tau = tau_max^xi tau_min^(1 - xi) -- uniform in log tau."""
    rng = np.random.default_rng(0)
    taus = np.array([lod_maths.sample_tau(rng, 3.0, 64.0) for _ in range(20_000)])
    assert taus.min() >= 3.0 and taus.max() <= 64.0
    unit = (np.log(taus) - math.log(3.0)) / (math.log(64.0) - math.log(3.0))
    np.testing.assert_allclose(np.quantile(unit, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.02)


def _camera_at(eye: list[float], target: list[float], size: int = 64) -> lod_maths.Camera:
    k = np.array([[50.0, 0, size / 2], [0, 50.0, size / 2], [0, 0, 1]])
    return lod_maths.Camera(lod_maths.look_at(np.array(eye), np.array(target)), k, size, size)


def test_the_plan_gives_each_parent_h3dgs_many_selections() -> None:
    tree = small_tree()
    assert pytest.approx(15000 * math.log(2) / math.log(20)) == lod_maths.H3DGS_SELECTIONS
    # Far away: the root's error projects under every tau, so it is always the cut.
    far = [_camera_at([-1000.0, 0.5 + i, 0.5], [2, 0.5, 0.5]) for i in range(3)]
    plan = lod_maths.plan_iterations(tree, far, 3.0, 64.0)
    # Weighted by gaussians: the root (10) always, `a` (5) never.
    assert plan.selection_rate == pytest.approx(10 / 15)
    assert plan.iterations == math.ceil(lod_maths.H3DGS_SELECTIONS / (10 / 15))
    # Looking away from everything: nothing to train on.
    away = [_camera_at([-10.0, 0.5, 0.5], [-20.0, 0.5, 0.5])]
    assert lod_maths.plan_iterations(tree, away, 3.0, 64.0).iterations == 0
    # Never more than H3DGS's own 15,000.
    assert lod_maths.plan_iterations(tree, far, 3.0, 64.0, most=2000).iterations == 2000


def test_a_placed_camera_sees_the_placed_splat_where_the_original_saw_the_original() -> None:
    rng = np.random.default_rng(1)
    angle = 0.7
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    rotation = rotation @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # a y-up model too
    translation, scale = np.array([5.0, -2.0, 1.0]), 3.7
    camera = lod_maths.look_at(np.array([4.0, 1.0, 2.0]), np.zeros(3))
    c2w = np.linalg.inv(camera)
    points = rng.normal(size=(50, 3))
    k = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])

    def project(viewmat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        cam = xyz @ viewmat[:3, :3].T + viewmat[:3, 3]
        uv: np.ndarray = cam @ k.T
        return np.asarray(uv[:, :2] / uv[:, 2:], dtype=np.float64)

    placed_points = scale * points @ rotation.T + translation
    placed_view = np.linalg.inv(lod_maths.placed_camtoworld(c2w, rotation, translation, scale))
    np.testing.assert_allclose(project(placed_view, placed_points), project(camera, points))


def test_the_position_schedule_is_h3dgss_and_the_radius_its_getnerfppnorm() -> None:
    assert lod_maths.position_lr(0, 1000, 2.0) == pytest.approx(2e-5 * 2.0)
    # train_post.py runs 15k of a 30k schedule: it ends at the geometric middle.
    assert lod_maths.position_lr(1000, 1000, 2.0) == pytest.approx(math.sqrt(2e-5 * 2e-7) * 2)
    eyes = np.array([[float(i), 0.0, 0.0] for i in range(11)])
    distances = np.abs(np.arange(11) - 5.0)
    assert lod_maths.spatial_scale(eyes) == pytest.approx(1.1 * np.quantile(distances, 0.9))


def test_a_box_is_in_view_unless_it_is_behind_or_beside_the_camera() -> None:
    camera = _camera_at([0.0, 0.0, 0.0], [0.0, 0.0, 10.0])
    low, high = np.array([-1.0, -1.0, 9.0]), np.array([1.0, 1.0, 11.0])
    args = (camera.viewmat, camera.k, camera.width, camera.height)
    assert lod_maths.in_view(low, high, *args)
    assert not lod_maths.in_view(low - [0, 0, 20], high - [0, 0, 20], *args)  # behind
    beside = np.array([30.0, 0.0, 0.0])
    assert not lod_maths.in_view(low + beside, high + beside, *args)
    # A box the camera is inside is in view.
    assert lod_maths.in_view(np.array([-5.0, -5, -5]), np.array([5.0, 5, 5]), *args)


def test_the_placement_is_places_two_steps_as_one() -> None:
    rng = np.random.default_rng(2)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotation *= np.sign(np.linalg.det(rotation))
    recentred = SimpleNamespace(
        rotation=np.eye(3), translation=np.array([0.3, -0.2, 1.5]), scale=1.0
    )
    xyz = rng.normal(size=(20, 3))
    expected = stages.place_points(xyz, rotation, [1.0, 2.0, 3.0], 2.5, recentred)  # type: ignore[arg-type]
    document = lod_parents.placement(rotation, [1.0, 2.0, 3.0], 2.5, recentred)
    r, t = np.array(document["rotation"]), np.array(document["translation"])
    got = float(document["scale"]) * xyz @ r.T + t  # type: ignore[arg-type]
    np.testing.assert_allclose(got, expected, atol=1e-5)


# --- place writes placement.json ---------------------------------------------------------


def _splat_columns(count: int, seed: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(count, 3)) * [2.0, 2.0, 0.5]
    quat = rng.normal(size=(count, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    columns = {axis: xyz[:, i] for i, axis in enumerate("xyz")}
    columns.update({f"f_dc_{i}": rng.normal(0, 0.5, count) for i in range(3)})
    columns["opacity"] = rng.uniform(0.0, 4.0, count)
    columns.update({f"scale_{i}": rng.normal(-3.5, 0.3, count) for i in range(3)})
    columns.update({f"rot_{i}": quat[:, i] for i in range(4)})
    return {name: np.ascontiguousarray(v, dtype=np.float32) for name, v in columns.items()}


def test_place_writes_the_similarity_it_applied(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    columns = _splat_columns(3000)
    trained = workdir.input_path("trained.ply")
    trained.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(trained, columns)
    angle = math.radians(40.0)
    frame = {
        "source": "test",
        "scale": 1.8,
        "rotation": [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        "translationM": [2.0, 1.0, -0.5],
        "recentre": True,
    }
    workdir.input_path("georef.json").write_text(
        json.dumps({"lat": 1.0, "lon": 2.0, "height": 0.0, "frame": frame})
    )
    recipe = make_recipe(
        [{"id": "place", "impl": "place_splat"}], inputs=["trained.ply", "georef.json"]
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    placed = gaussians.read_splat(workdir.artifact_path("place", "canonical.ply"))
    document = json.loads(workdir.artifact_path("place", "placement.json").read_text())
    r, t = np.array(document["rotation"]), np.array(document["translation"])
    xyz = np.stack([columns[a] for a in "xyz"], 1).astype(np.float64)
    moved = document["scale"] * xyz @ r.T + t
    assert document["scale"] == pytest.approx(1.8)
    np.testing.assert_allclose(moved, placed.xyz, atol=1e-4)


# --- the hand-over to package, with a stand-in optimiser ---------------------------------


def _seed_capture(workdir: Workdir, count: int = 3000) -> Path:
    ply = workdir.input_path("canonical.ply")
    ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.write_ply(ply, _splat_columns(count, seed=6))
    workdir.input_path("placement.json").write_text(
        json.dumps(lod_parents.placement(np.eye(3), None, 1.0, None))
    )
    workdir.input_path("georef.json").write_text(json.dumps({"lat": 1, "lon": 2, "height": 0}))
    frames = workdir.input_path("frames")
    frames.mkdir(parents=True, exist_ok=True)
    (frames / "frame_0000.jpg").write_bytes(b"\xff\xd8\xff not a real jpeg")
    poses = workdir.input_path("poses")
    poses.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (poses / name).write_bytes(b"colmap")
    return ply


def _handover(
    workdir: Workdir, optimise: dict[str, object] | None = None, package_budget: int = BUDGET
) -> None:
    params: dict[str, object] = {
        "python": sys.executable,
        "script": str(STAND_IN),
        "trainer": str(TORCH_STAND_IN / "simple_trainer.py"),
        "tile_gaussians": BUDGET,
        "opacity_min": 0.02,
    }
    params.update(optimise or {})
    recipe = make_recipe(
        [
            {"id": "optimise_lod", "impl": "h3dgs_parents", "params": params},
            {
                "id": "package",
                "impl": "splat_tiles",
                "params": {"tile_gaussians": package_budget, "opacity_min": 0.02},
            },
        ],
        inputs=["canonical.ply", "placement.json", "frames", "poses", "georef.json"],
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))


def _summary_path(workdir: Workdir) -> Path:
    return workdir.artifact_path("optimise_lod", "lod_parents") / "summary.json"


def _parent_z(tiles: Path) -> dict[str, np.ndarray]:
    tileset = json.loads((tiles / "tileset.json").read_text())
    out: dict[str, np.ndarray] = {}
    stack = [tileset["root"]]
    while stack:
        tile = stack.pop()
        if tile.get("children"):
            stack.extend(tile["children"])
            uri = tile["content"]["uri"]
            raw = (tiles / uri).read_bytes()
            json_length = int.from_bytes(raw[12:16], "little")
            binary = raw[20 + json_length + 8 :]
            gltf = json.loads(raw[20 : 20 + json_length])
            view = gltf["bufferViews"][0]
            spz = binary[view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]]
            out[uri] = splat_tiles.unpack_spz(spz)["z"]
    return out


def test_package_draws_the_optimised_parents(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    ply = _seed_capture(workdir)
    _handover(workdir)

    summary = json.loads(
        workdir.artifact_path("optimise_lod", "lod_parents/summary.json").read_text()
    )
    assert summary["status"] == "ok"
    plain = tmp_path / "plain"
    splat_tiles.convert(ply, plain, 1, 2, 0, 0.02, BUDGET)
    before, after = _parent_z(plain), _parent_z(workdir.artifact_path("package", "splat"))
    assert before and set(before) == set(after)
    for uri in before:
        np.testing.assert_allclose(after[uri] - before[uri], 0.01, atol=1.0 / 4096)
    log = workdir.log_path("package").read_text()
    assert "optimised parents" in log


@pytest.mark.parametrize("mode", ["fail", "rejected"])
def test_a_failed_or_rejected_optimisation_leaves_the_merged_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("LOD_STAND_IN", mode)
    workdir = Workdir.create(tmp_path / "run")
    ply = _seed_capture(workdir)
    _handover(workdir)

    folder = workdir.artifact_path("optimise_lod", "lod_parents")
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == ("failed" if mode == "fail" else "rejected")
    assert not (folder / "lod_parents.npz").exists()
    plain = tmp_path / "plain"
    splat_tiles.convert(ply, plain, 1, 2, 0, 0.02, BUDGET)
    assert _parent_z(plain).keys() == _parent_z(workdir.artifact_path("package", "splat")).keys()
    for uri, z in _parent_z(plain).items():
        np.testing.assert_array_equal(z, _parent_z(workdir.artifact_path("package", "splat"))[uri])
    if mode == "fail":
        assert (
            "WARNING: the parents were not optimised"
            in workdir.log_path("optimise_lod").read_text()
        )


def test_parents_for_another_packing_are_not_drawn(tmp_path: Path) -> None:
    """`package` packing a different tile budget than `optimise_lod` optimised for: the
    packer refuses the parents by name and the stage packs the merged ones."""
    workdir = Workdir.create(tmp_path / "run")
    _seed_capture(workdir)
    _handover(workdir, package_budget=500)

    log = workdir.log_path("package").read_text()
    assert "WARNING: the optimised parents were not used" in log and "tiles of 400" in log
    assert (workdir.artifact_path("package", "splat") / "tileset.json").is_file()


def test_it_can_be_turned_off(tmp_path: Path) -> None:
    workdir = Workdir.create(tmp_path / "run")
    _seed_capture(workdir)
    _handover(workdir, {"enabled": False})
    summary = json.loads(
        workdir.artifact_path("optimise_lod", "lod_parents/summary.json").read_text()
    )
    assert summary["status"] == "off"


def test_the_shipped_recipe_optimises_between_place_and_package_on_an_l4() -> None:
    recipe = load_recipe("photo-reconstruct")
    ids = [stage.id for stage in recipe.stages]
    assert ids.index("place") < ids.index("optimise_lod") == ids.index("package") - 1
    optimise = recipe.stages[ids.index("optimise_lod")]
    package = recipe.stages[ids.index("package")]
    assert optimise.impl == "h3dgs_parents"
    assert optimise.gpu is not None and optimise.gpu.tier == "l4"
    assert optimise.params["enabled"] is True
    for name in ("tile_gaussians", "opacity_min"):
        assert optimise.params[name] == package.params[name], name


# --- the torch half, on a CPU ------------------------------------------------------------


def _torch_python() -> str | None:
    candidate = os.environ.get("LOD_TORCH_PYTHON")
    if not candidate:
        return None
    probe = subprocess.run([candidate, "-c", "import torch"], capture_output=True, check=False)
    return candidate if probe.returncode == 0 else None


@pytest.mark.skipif(_torch_python() is None, reason="no $LOD_TORCH_PYTHON with torch")
def test_the_real_script_optimises_a_tiny_scene_on_a_cpu(tmp_path: Path) -> None:
    """`lod_optimise.py` end to end with the torch stand-in rasteriser: the parents it keeps
    lower the held-out loss at the cut, stay inside their boxes and at most 0.99 opaque,
    and the packer accepts them for the very PLY they were optimised on."""
    python = _torch_python()
    assert python is not None
    scene = tmp_path / "scene"
    subprocess.run([python, str(TORCH_STAND_IN / "make_scene.py"), str(scene)], check=True)
    out = tmp_path / "out"
    subprocess.run(
        [
            python,
            str(HERE.parent / "lod_optimise.py"),
            "--ply",
            str(scene / "canonical.ply"),
            "--data_dir",
            str(scene / "data"),
            "--placement",
            str(scene / "placement.json"),
            "--out",
            str(out),
            "--trainer",
            str(TORCH_STAND_IN / "simple_trainer.py"),
            "--tile_gaussians",
            str(BUDGET),
            "--iterations",
            "120",
            "--device",
            "cpu",
            "--no-lpips",
            "--switch-size",
            "48",
            "36",
        ],
        check=True,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "ok", summary["reason"]
    assert summary["judgedOn"] == "held-out frames"
    assert summary["after"]["loss"] < summary["before"]["loss"]
    assert summary["iterationsRun"] == 120 and summary["iterationsSkipped"] == 0
    assert 1000 <= summary["plan"]["iterations"] <= 15000
    overrides = splat_tiles.ParentOverrides.load(out / "lod_parents.npz")
    opacity = 1.0 / (1.0 + np.exp(-overrides.gaussians.opacity_logit.astype(np.float64)))
    assert opacity.max() <= splat_tiles.MERGED_OPACITY_MAX + 1e-6
    emitted: dict[str, Any] = {}

    def emit(tile: Any, gaussians_: Any, keys: Any) -> None:
        emitted[tile.uri] = tile

    splat_tiles.hierarchy(scene / "canonical.ply", 0.02, BUDGET, tmp_path, emit)
    for index, uri in enumerate(overrides.uris):
        tile = emitted[uri]
        start, stop = overrides.offsets[index], overrides.offsets[index + 1]
        xyz = overrides.gaussians.xyz[start:stop]
        assert np.all(xyz >= tile.low - 1e-5) and np.all(xyz <= tile.high + 1e-5)
    stats = splat_tiles.convert(
        scene / "canonical.ply", tmp_path / "tiles", 0, 0, 0, 0.02, BUDGET, parents=overrides
    )
    assert stats["optimised_parent_gaussians"] == summary["parentGaussians"]


def test_the_packed_tileset_and_the_optimisers_tree_cut_alike(tmp_path: Path) -> None:
    """`experiments/lod_compare.py` reads the tree back from `tileset.json`; the optimiser
    builds it in memory (`splat_tiles.hierarchy`). Same tiles, same boxes, same errors --
    so the same cut from anywhere."""
    from experiments.lod_compare import Tileset, summarise

    ply = tmp_path / "scene.ply"
    gaussians.write_ply(ply, _splat_columns(3000, seed=8))
    splat_tiles.convert(ply, tmp_path / "splat", 0, 0, 0, 0.02, BUDGET)
    packed = Tileset(tmp_path / "splat")

    def emit(tile: Any, gaussians_: Any, keys: Any) -> None:
        pass

    walked = splat_tiles.hierarchy(ply, 0.02, BUDGET, tmp_path, emit).walk()
    tree = lod_maths.tree_from(walked, [tile.count for tile in walked])
    assert sorted(packed.uris) == sorted(tree.uris)
    order = [packed.uris.index(uri) for uri in tree.uris]
    np.testing.assert_allclose(packed.tree.error[order], tree.error)
    np.testing.assert_allclose(packed.tree.low[order], tree.low, atol=1e-9)
    rng = np.random.default_rng(3)
    for _ in range(50):
        eye = rng.normal(size=3) * 20
        tau = lod_maths.sample_tau(rng, 3.0, 64.0)
        a = {packed.uris[t] for t in lod_maths.cut(packed.tree, eye, 500.0, tau)}
        b = {tree.uris[t] for t in lod_maths.cut(tree, eye, 500.0, tau)}
        assert a == b
    rows = [
        {"test": "root-at-switch", "tau": 16.0, "tileset": name, "lpips": v, "psnr": 20.0,
         "ssim": 0.9, "holes": 0.1}
        for name, v in (("merged", 0.3), ("merged", 0.32), ("optimised", 0.2))
    ]  # fmt: skip
    lines = summarise(rows)
    assert len(lines) == 2 and "lpips 0.310" in lines[0] and "optimised" in lines[1]
