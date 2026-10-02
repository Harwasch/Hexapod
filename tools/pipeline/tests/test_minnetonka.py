"""The Minnetonka tree's pure parts, and its driver end to end on a local mirror.

No network, no bucket, no GPU: the dataset is stood in for by a Hub-shaped directory of
tiny JPEGs carrying DJI XMP, and the train step by the test suite's stand-in trainer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import sfm
from experiments import minnetonka as tree
from recipe import load_recipe

REPO = Path(__file__).resolve().parents[3]


# --- which files ----------------------------------------------------------------------------


def listing_for(group: str, count: int, total: int) -> list[dict[str, Any]]:
    """A Hub `tree` listing of `count` images of `group` adding up to `total` bytes."""
    each = total // count
    sizes = [each] * (count - 1) + [total - each * (count - 1)]
    return [
        {
            "type": "file",
            "path": f"images/{group}-{number}.jpg",
            "size": size,
            "lfs": {"oid": hashlib.sha256(f"{group}{number}".encode()).hexdigest(), "size": size},
        }
        for number, size in zip(range(1, count + 1), sizes, strict=True)
    ]


def test_the_plan_is_the_pinned_group_in_capture_order() -> None:
    group = tree.GROUPS["The_Tree"]
    entries = listing_for("The_Tree", group.images, group.bytes)
    entries += listing_for("Original_low", 85, tree.GROUPS["Original_low"].bytes)
    planned = tree.plan_download(tree.parse_listing(entries), ["The_Tree"])
    assert len(planned) == 659
    assert tree.planned_bytes(planned) == 13_961_267_697
    # Numeric, not lexicographic: 2 before 10 before 100.
    assert [f.name for f in planned[:3]] == ["The_Tree-1.jpg", "The_Tree-2.jpg", "The_Tree-3.jpg"]
    assert planned[9].name == "The_Tree-10.jpg"
    assert planned[0].url.endswith(f"/resolve/{tree.HF_REVISION}/images/The_Tree-1.jpg")


def test_a_listing_that_is_not_the_pinned_revision_is_refused() -> None:
    group = tree.GROUPS["The_Tree"]
    entries = listing_for("The_Tree", group.images, group.bytes)
    with pytest.raises(ValueError, match="658 images"):
        tree.plan_download(tree.parse_listing(entries[1:]), ["The_Tree"])
    entries[5] = {**entries[5], "size": entries[5]["size"] + 1}
    with pytest.raises(ValueError, match="bytes"):
        tree.plan_download(tree.parse_listing(entries), ["The_Tree"])
    with pytest.raises(ValueError, match="groups must be"):
        tree.plan_download([], ["Somewhere_else"])


def test_an_image_without_an_lfs_digest_is_refused() -> None:
    with pytest.raises(ValueError, match="no LFS sha256"):
        tree.parse_listing([{"type": "file", "path": "images/The_Tree-1.jpg", "size": 1}])
    assert tree.parse_listing([{"type": "directory", "path": "images"}]) == []


def test_pagination_follows_the_link_header() -> None:
    link = '<https://huggingface.co/api/datasets/x/tree/main/images?cursor=abc>; rel="next"'
    assert (
        tree.next_page(link) == "https://huggingface.co/api/datasets/x/tree/main/images?cursor=abc"
    )
    assert tree.next_page(None) is None
    assert tree.next_page('<https://a>; rel="prev"') is None


def test_frame_names_sort_in_capture_order() -> None:
    names = ["The_Tree-10.jpg", "The_Tree-2.jpg", "The_Tree-100.jpg", "The_Tree-1.jpg"]
    assert sorted(tree.frame_name(n) for n in names) == [
        "The_Tree-0001.jpg",
        "The_Tree-0002.jpg",
        "The_Tree-0010.jpg",
        "The_Tree-0100.jpg",
    ]
    with pytest.raises(ValueError):
        tree.frame_name("DJI_0001.JPG")


def test_the_suspects_are_the_four_the_dataset_names() -> None:
    assert tree.SUSPECT_CAMERAS == (
        "The_Tree-88.jpg",
        "The_Tree-137.jpg",
        "The_Tree-223.jpg",
        "The_Tree-235.jpg",
    )


# --- metadata -------------------------------------------------------------------------------

ATTRIBUTE_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
    'xmp:CreateDate="2020-07-20T19:41:59" drone-dji:GpsLatitude="+44.9446257" '
    'drone-dji:GpsLongitude="-93.4259758" drone-dji:AbsoluteAltitude="+287.00" '
    'drone-dji:RelativeAltitude="+4.90" drone-dji:GimbalPitchDegree="-9.90" '
    'drone-dji:GimbalYawDegree="+143.90"/></rdf:RDF></x:xmpmeta>'
)
ELEMENT_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description>'
    "<drone-dji:RelativeAltitude>+0.50</drone-dji:RelativeAltitude>"
    "<drone-dji:GimbalPitchDegree>+4.20</drone-dji:GimbalPitchDegree>"
    "<drone-dji:GimbalYawDegree>-40.10</drone-dji:GimbalYawDegree>"
    "<xmp:CreateDate>2020-07-18T07:56:08</xmp:CreateDate>"
    "</rdf:Description></rdf:RDF></x:xmpmeta>"
)


def jpeg_with_xmp(xmp: str, size: tuple[int, int] = (64, 48), seed: int = 0) -> bytes:
    """A JPEG with an XMP APP1 segment after SOI, the way a camera writes one."""
    pixels = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
    body = buffer.getvalue()
    payload = b"http://ns.adobe.com/xap/1.0/\0" + xmp.encode()
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    return body[:2] + segment + body[2:]


def test_drone_metadata_is_read_in_both_forms_the_dataset_carries() -> None:
    meta = tree.read_drone_meta(jpeg_with_xmp(ATTRIBUTE_XMP))
    assert meta.relative_altitude_m == pytest.approx(4.9)
    assert meta.gimbal_pitch_deg == pytest.approx(-9.9)
    assert meta.gimbal_yaw_deg == pytest.approx(143.9)
    assert meta.latitude == pytest.approx(44.9446257)
    assert meta.taken == "2020-07-20T19:41:59"
    low = tree.read_drone_meta(jpeg_with_xmp(ELEMENT_XMP))
    assert (low.relative_altitude_m, low.gimbal_pitch_deg) == (0.5, 4.2)
    assert low.seconds() is not None and meta.seconds() is not None
    assert tree.read_drone_meta(b"\xff\xd8 no xmp here") == tree.DroneMeta()
    assert tree.DroneMeta.from_dict(meta.to_dict()) == meta


def test_frames_are_shrunk_to_the_long_side_and_small_ones_kept(tmp_path: Path) -> None:
    big = jpeg_with_xmp(ATTRIBUTE_XMP, size=(3000, 2000))
    assert tree.shrink_jpeg(big, tmp_path / "a.jpg", 1600) == (1600, 1067)
    with Image.open(tmp_path / "a.jpg") as image:
        assert image.size == (1600, 1067)
    assert tree.shrink_jpeg(jpeg_with_xmp(ATTRIBUTE_XMP), tmp_path / "b.jpg", 1600) == (64, 48)


# --- the published model --------------------------------------------------------------------


def write_text_model(directory: Path, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """cameras.txt / images.txt / points3D.txt as the dataset ships them: one PINHOLE
    camera per image, empty POINTS2D lines, tracks empty."""
    directory.mkdir(parents=True, exist_ok=True)
    cameras = ["# Camera list", "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    images = ["# Image list with two lines of data per image:", "#   POINTS2D[]"]
    for index, (name, (rotation, centre)) in enumerate(poses.items(), start=1):
        cameras.append(f"{index} PINHOLE 5464 3640 4008.5 4008.5 2732 1820")
        q = quat_of(rotation)
        t = -rotation @ centre
        images.append(f"{index} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {index} {name}")
        images.append("")
    (directory / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (directory / "images.txt").write_text("\n".join(images) + "\n")
    points = ["# 3D point list", "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    rng = np.random.default_rng(3)
    for index in range(1, 201):
        x, y, z = rng.normal(0, 2, 3)
        points.append(f"{index} {x:.6f} {y:.6f} {z:.6f} 100 120 90 0")
    (directory / "points3D.txt").write_text("\n".join(points) + "\n")


def quat_of(rotation: np.ndarray) -> tuple[float, float, float, float]:
    m = rotation
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return (w, x, y, z)


def test_the_published_model_is_filtered_renamed_and_written_as_colmap_binary(
    tmp_path: Path,
) -> None:
    scene = Orbit(per_tier=30)
    poses = {
        name: (scene.rotation_model(i), scene.centre_model(i)) for i, name in enumerate(scene.names)
    }
    poses["Original_low-3.jpg"] = poses.pop(scene.names[-1])
    write_text_model(tmp_path / "text", poses)
    model = tree.read_text_model(tmp_path / "text")
    assert len(model.images) == len(poses) and len(model.points) == 200
    wanted = [n for n in model.images if n.name.startswith("The_Tree-")]
    rename = {i.name: tree.frame_name(i.name) for i in wanted if i.name not in tree.SUSPECT_CAMERAS}
    chosen = tree.select_images(model, rename)
    tree.write_binary_model(chosen, tmp_path / "bin")
    read = sfm.read_model(tmp_path / "bin")
    names = [image.name for image in read.images]
    assert names == sorted(rename.values())
    assert "The_Tree-0088.jpg" not in names  # a suspect, dropped
    assert not any(n.startswith("Original_low") for n in names)
    assert len(read.cameras) == len(names)  # the dropped images' cameras went with them
    assert read.points3d == 200
    first = read.by_name()["The_Tree-0001.jpg"]
    np.testing.assert_allclose(first.centre, scene.centre_model(0), atol=1e-5)
    assert read.cameras[0].model == "PINHOLE" and read.cameras[0].focal_px == 4008.5


# --- the pose gate --------------------------------------------------------------------------


class Orbit:
    """A drone orbiting a tree in three tiers, in a world frame (metres, x east, y north,
    z up above take-off), and the same poses in a model frame 2.5 units to the metre,
    turned and moved by an arbitrary similarity -- what SfM would hand back."""

    TIERS = ((4.9, -11.0, 9.0), (6.4, -38.0, 9.5), (1.4, 15.0, 7.0))

    def __init__(self, per_tier: int = 60, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.units_per_metre = 2.5
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = 1.1
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        self.q = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
        self.b = np.array([3.0, -7.0, 11.0])
        self.names: list[str] = []
        self.world: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.metas: dict[str, tree.DroneMeta] = {}
        number = 0
        for height, pitch, radius in self.TIERS:
            for step in range(per_tier):
                number += 1
                azimuth = 2 * math.pi * step / per_tier
                heading = math.degrees(azimuth) + 180.0
                centre = np.array([radius * math.sin(azimuth), radius * math.cos(azimuth), height])
                h, p = math.radians(heading), math.radians(pitch)
                view = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
                up = np.array([-math.sin(h) * math.sin(p), -math.cos(h) * math.sin(p), math.cos(p)])
                self.world.append((centre, view, up))
                name = f"The_Tree-{number}.jpg"
                self.names.append(name)
                self.metas[tree.frame_name(name)] = tree.DroneMeta(
                    relative_altitude_m=round(height + rng.normal(0, 0.08), 1),
                    gimbal_pitch_deg=pitch + rng.normal(0, 0.4),
                    gimbal_yaw_deg=float((heading + rng.normal(0, 2.0) + 180) % 360 - 180),
                    latitude=44.9446,
                    longitude=-93.4259,
                    absolute_altitude_m=287.0 + height - 4.9,
                    taken=f"2020-07-20T19:{40 + number // 60:02d}:{number % 60:02d}",
                )

    def centre_model(self, index: int) -> np.ndarray:
        return np.asarray(self.units_per_metre * self.q @ self.world[index][0] + self.b)

    def rotation_model(self, index: int) -> np.ndarray:
        _, view, up = self.world[index]
        right = np.cross(view, up)
        world_to_camera = np.stack([right, -up, view])  # x right, y down, z forward
        return np.asarray(world_to_camera @ self.q.T)

    def views(self) -> list[tree.PosedView]:
        return [
            tree.PosedView(
                name=tree.frame_name(name),
                centre=self.centre_model(i),
                view=self.q @ self.world[i][1],
                up=self.q @ self.world[i][2],
            )
            for i, name in enumerate(self.names)
        ]


def test_consistent_poses_pass_and_the_frame_is_metres_east_north_up() -> None:
    scene = Orbit()
    fit = tree.fit_frame(scene.views(), scene.metas)
    assert fit.verdict == "pass", fit.to_dict()
    assert fit.metres_per_unit == pytest.approx(1 / scene.units_per_metre, rel=0.01)
    assert fit.up_agreement_deg < 1.0
    assert fit.residuals["pitchMedianAbsDeg"] < 1.0
    assert fit.residuals["headingMedianAbsDeg"] < 3.0
    # Model-frame camera centres land back on the world ones: north is +y, up is +z,
    # z = 0 at take-off, and the origin on the axis the orbit looks at.
    placed = fit.apply(np.stack([scene.centre_model(i) for i in range(len(scene.names))]))
    truth = np.stack([w[0] for w in scene.world])
    assert np.abs(placed - truth).max() < 0.35
    roi = fit.roi(12.0)
    assert roi["radius"] == pytest.approx(12.0 * scene.units_per_metre, rel=0.01)


def test_poses_attached_to_the_wrong_photos_fail_the_gate_by_name() -> None:
    """What the published model does: the poses are real, the names are not theirs."""
    scene = Orbit()
    names = [tree.frame_name(n) for n in scene.names]
    rng = np.random.default_rng(7)
    shuffled = dict(scene.metas)
    moved = rng.permutation(len(names))[: len(names) // 2]
    for a, b in zip(moved, np.roll(moved, 1), strict=True):
        shuffled[names[a]] = scene.metas[names[b]]
    fit = tree.fit_frame(scene.views(), shuffled)
    assert fit.verdict == "fail"
    assert {"pitch", "height", "heading"} <= set(fit.failed)
    assert fit.worst and "pitchResidualDeg" in fit.worst[0]


def test_too_little_metadata_is_a_missing_input_not_a_verdict() -> None:
    scene = Orbit(per_tier=3)
    with pytest.raises(ValueError, match="cannot fit a frame"):
        tree.fit_frame(scene.views(), {})


def test_a_second_take_off_gets_its_own_barometric_zero() -> None:
    assert tree.flights([0, 5, 10, 400, 405, 30]) == [0, 0, 0, 1, 1, 0]


# --- what the stages are asked for -----------------------------------------------------------


def test_the_stages_run_photo_reconstructs_own_settings_unchanged() -> None:
    """Nothing is fixed for this capture: the recipe's Standard train params as they are
    (cap_max auto, converge, blocks auto; no budget_max, no memory number), and the
    recipe's frame-size rule and train tier."""
    recipe = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "train")
    train = tree.train_params({"roi": {"center": [0, 0, 0], "radius": 5}})
    assert {k: v for k, v in train.items() if k != "roi"} == dict(recipe.params)
    assert train["iterations"] == 30_000
    assert train["strategy"] == "mcmc" and train["cap_max"] == "auto" and train["converge"] is True
    assert "blocks" not in train  # auto: past one GPU's memory, blocks
    assert "budget_max" not in train and "gpu_memory_gb" not in train
    assert train["roi"]["radius"] == 5
    assert "sh_degree" not in train  # gsplat's default, 3
    assert recipe.gpu is not None and tree.train_tier() == recipe.gpu.tier
    normalize = next(s for s in load_recipe("photo-reconstruct").stages if s.id == "normalize")
    assert tree.frame_size_params() == (
        normalize.params["auto_base"],
        normalize.params["auto_ceiling"],
    )
    pose = tree.pose_params()
    assert pose["matcher"] == "sequential" and pose["colmap"] == "4.2"
    assert pose["max_image_size"] == 1600


# --- capture sessions ---------------------------------------------------------------------------


def test_sessions_are_days_and_flights_on_the_cameras_clock() -> None:
    metas = {
        "The_Tree-0001.jpg": tree.DroneMeta(taken="2020-07-18T10:00:01"),
        "The_Tree-0002.jpg": tree.DroneMeta(taken="2020-07-18T10:00:06"),
        "The_Tree-0003.jpg": tree.DroneMeta(taken="2020-07-20T19:40:00"),
        "The_Tree-0004.jpg": tree.DroneMeta(taken="2020-07-20T19:40:05"),
        # A battery change later that evening: a new flight, the same day.
        "The_Tree-0005.jpg": tree.DroneMeta(taken="2020-07-20T20:10:00"),
        "The_Tree-0006.jpg": tree.DroneMeta(),
    }
    assert tree.session_counts(metas) == {
        "2020-07-18/0": 2,
        "2020-07-20/1": 2,
        "2020-07-20/2": 1,
        "unknown": 1,
    }
    assert tree.days(tree.session_counts(metas)) == {
        "2020-07-18": 2,
        "2020-07-20": 3,
        "unknown": 1,
    }
    assert tree.names_in(metas, ["2020-07-20"]) == {
        "The_Tree-0003.jpg",
        "The_Tree-0004.jpg",
        "The_Tree-0005.jpg",
    }
    assert tree.names_in(metas, ["2020-07-20/2", "2020-07-18"]) == {
        "The_Tree-0001.jpg",
        "The_Tree-0002.jpg",
        "The_Tree-0005.jpg",
    }
    with pytest.raises(ValueError, match="2020-07-19"):
        tree.names_in(metas, ["2020-07-19"])
    with pytest.raises(ValueError, match="2020-07-2"):
        tree.names_in(metas, ["2020-07-2"])  # a prefix of a day is not a day


# --- the driver, end to end on a mirror ------------------------------------------------------


def _driver() -> Any:
    spec = importlib.util.spec_from_file_location(
        "minnetonka_driver", REPO / "infra" / "modal" / "minnetonka.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["minnetonka_driver"] = module
    spec.loader.exec_module(module)
    return module


def build_mirror(root: Path, scene: Orbit, *, poses_consistent: bool = True) -> dict[str, Any]:
    """A Hub-shaped directory: images with DJI XMP, colmap/*.txt, and the listings."""
    listing: list[dict[str, Any]] = []
    for i, name in enumerate(scene.names):
        meta = scene.metas[tree.frame_name(name)]
        xmp = (
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
            f'xmp:CreateDate="{meta.taken}" '
            f'drone-dji:RelativeAltitude="{meta.relative_altitude_m}" '
            f'drone-dji:AbsoluteAltitude="{meta.absolute_altitude_m}" '
            f'drone-dji:GimbalPitchDegree="{meta.gimbal_pitch_deg}" '
            f'drone-dji:GimbalYawDegree="{meta.gimbal_yaw_deg}" '
            f'drone-dji:GpsLatitude="{meta.latitude}" drone-dji:GpsLongitude="{meta.longitude}"/>'
            "</rdf:RDF></x:xmpmeta>"
        )
        data = jpeg_with_xmp(xmp, size=(96, 64), seed=i)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "images" / name).write_bytes(data)
        listing.append(
            {
                "type": "file",
                "path": f"images/{name}",
                "size": len(data),
                "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)},
            }
        )
    order = list(range(len(scene.names)))
    if not poses_consistent:
        order = list(np.random.default_rng(5).permutation(order))
    poses = {
        name: (scene.rotation_model(j), scene.centre_model(j))
        for name, j in zip(scene.names, order, strict=True)
    }
    write_text_model(root / "colmap", poses)
    (root / "_listing").mkdir()
    (root / "_listing" / "images.json").write_text(json.dumps(listing))
    return {
        "group": tree.Group(
            images=len(listing), bytes=sum(e["size"] for e in listing), date="2020-07-20"
        ),
        "colmap": {
            name: (
                (root / "colmap" / name).stat().st_size,
                hashlib.sha256((root / "colmap" / name).read_bytes()).hexdigest(),
            )
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
    }


@pytest.fixture
def mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    scene = Orbit(per_tier=10)
    pins = build_mirror(tmp_path / "hub", scene)
    monkeypatch.setattr(tree, "GROUPS", {"The_Tree": pins["group"]})
    monkeypatch.setattr(tree, "COLMAP_FILES", pins["colmap"])
    return tmp_path / "hub", _driver()


def test_check_reads_only_metadata_and_gates_the_published_poses(
    tmp_path: Path, mirror: tuple[Path, Any]
) -> None:
    hub, driver = mirror
    work = tmp_path / "work"
    assert driver.main(["--work", str(work), "check", "--mirror", str(hub), "--workers", "2"]) == 0
    gate = json.loads((work / "check" / "gate.json").read_text())
    assert gate["verdict"] == "pass"
    assert gate["withMeta"] == gate["posed"] == 30
    assert len(gate["camerasM"]) == gate["posed"]
    assert gate["focalPx"] == pytest.approx(4008.5)
    assert gate["takeoffMslM"] == pytest.approx(282.1, abs=0.2)


def test_a_scrambled_published_model_stops_prepare_before_anything_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = Orbit(per_tier=10)
    pins = build_mirror(tmp_path / "hub", scene, poses_consistent=False)
    monkeypatch.setattr(tree, "GROUPS", {"The_Tree": pins["group"]})
    monkeypatch.setattr(tree, "COLMAP_FILES", pins["colmap"])
    driver = _driver()
    work = tmp_path / "work"
    argv = ["--work", str(work), "--rehearse", "prepare", "--mirror", str(tmp_path / "hub")]
    assert driver.main([*argv, "--poses", "published"]) == 1
    kept = work / "bucket" / "experiments" / "minnetonka-tree" / "m0"
    assert (kept / "frames").is_dir()  # the frames are still worth keeping
    assert not (kept / "poses").exists() and not (kept / "frame.json").exists()


def test_a_tampered_image_is_refused(tmp_path: Path, mirror: tuple[Path, Any]) -> None:
    hub, driver = mirror
    target = hub / "images" / "The_Tree-3.jpg"
    target.write_bytes(target.read_bytes()[:-1] + b"\0")
    work = tmp_path / "work"
    with pytest.raises(ValueError, match=r"The_Tree-3\.jpg"):
        driver.main(["--work", str(work), "--rehearse", "prepare", "--mirror", str(hub)])


def test_prepare_then_train_rehearse_end_to_end(tmp_path: Path, mirror: tuple[Path, Any]) -> None:
    """Download, verify, shrink, gate, keep; then the train stage through the cloud
    runner over a local bucket with the stand-in trainer. Proves the plumbing only."""
    hub, driver = mirror
    work = tmp_path / "work"
    base = ["--work", str(work), "--rehearse"]
    prepare = [*base, "prepare", "--mirror", str(hub), "--poses", "published", "--workers", "2"]
    assert driver.main(prepare) == 0
    kept = work / "bucket" / "experiments" / "minnetonka-tree" / "m0"
    frames = sorted(p.name for p in (kept / "frames").iterdir())
    assert frames[0] == "The_Tree-0001.jpg" and len(frames) == 30
    assert json.loads((kept / "frame.json").read_text())["verdict"] == "pass"
    assert (kept / "poses" / "images.bin").is_file()
    # The frames were sized by photo-reconstruct's own rule, not a number of this path's:
    # 96 px photos are too small to be measured, so they keep their size.
    prepared = json.loads((kept / "prepare.json").read_text())
    assert prepared["frameSize"]["rule"] == "auto" and prepared["frameSizes"] == [[96, 64]]
    assert prepared["sessions"] == {"2020-07-20/0": 30}
    # A fresh job: only the bucket carries anything across.
    (work / "prepare").rename(work / "prepare-elsewhere")
    assert driver.main([*base, "train", "--cap-max", "2000"]) == 0
    assert (kept / "train" / "trained.ply").stat().st_size > 0
    report = json.loads((kept / "train" / "train.json").read_text())
    assert report["rehearsal"] is True
    # The recipe's params as they are: no ROI, no block count, no budget ceiling.
    assert not {"roi", "blocks", "budget_max"} & set(report["params"])
    assert report["cost"]["tier"] == tree.train_tier()
    assert report["registered"] == 30 and report["frameSize"] == [96, 64]
    assert report["days"] == {"2020-07-20": 30}
    into = tmp_path / "fetched"
    assert (
        driver.main([*base, "fetch", "train/trained.ply", "frame.json", "--into", str(into)]) == 0
    )
    assert (into / "train" / "trained.ply").is_file() and (into / "frame.json").is_file()


def test_poses_adopted_across_tags_and_one_session_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second tag re-prepares the same photos at another size, adopts the first tag's
    gated poses instead of solving them again, and trains on one capture session."""
    scene = Orbit(per_tier=10)
    for name in scene.names[20:]:  # the low hover tier, flown two days earlier
        frame = tree.frame_name(name)
        meta = scene.metas[frame]
        scene.metas[frame] = dataclasses.replace(
            meta, taken=str(meta.taken).replace("2020-07-20", "2020-07-18")
        )
    pins = build_mirror(tmp_path / "hub", scene)
    monkeypatch.setattr(tree, "GROUPS", {"The_Tree": pins["group"]})
    monkeypatch.setattr(tree, "COLMAP_FILES", pins["colmap"])
    driver = _driver()
    work = tmp_path / "work"
    hub = ["--mirror", str(tmp_path / "hub"), "--workers", "2"]
    base = ["--work", str(work), "--rehearse"]
    bucket = work / "bucket" / "experiments" / "minnetonka-tree"
    assert driver.main([*base, "prepare", *hub, "--poses", "published", "--max-side", "48"]) == 0
    assert json.loads((bucket / "m0" / "prepare.json").read_text())["frameSizes"] == [[48, 32]]
    assert driver.main([*base, "--tag", "m1", "prepare", *hub]) == 0
    assert not (bucket / "m1" / "poses").exists()  # colmap poses: nothing kept by prepare

    assert driver.main([*base, "--tag", "m1", "pose", "--from-tag", "m0"]) == 0
    pose = json.loads((bucket / "m1" / "pose.json").read_text())
    assert pose["adoptedFrom"] == "m0" and pose["gate"] == "pass"
    assert pose["frameSize"] == [96, 64] and pose["registered"] == 30
    kept_frame = (bucket / "m1" / "frame.json").read_bytes()
    assert kept_frame == (bucket / "m0" / "frame.json").read_bytes()
    with pytest.raises(SystemExit, match="this tag"):
        driver.main([*base, "--tag", "m1", "pose", "--from-tag", "m1"])

    train = [*base, "--tag", "m1", "train", "--cap-max", "2000", "--sessions", "2020-07-20"]
    assert driver.main(train) == 0
    report = json.loads((bucket / "m1" / "train" / "train.json").read_text())
    assert report["registered"] == 20 and report["frameSize"] == [96, 64]
    assert report["days"] == {"2020-07-20": 20}
    assert report["sessionSelection"]["images"] == 20
    assert report["sessionSelection"]["imagesBefore"] == 30
    with pytest.raises(ValueError, match="2020-07-19"):
        driver.main([*base, "--tag", "m1", "train", "--sessions", "2020-07-19"])

    assert driver.main(["describe", "--out", str(tmp_path / "capture.json")]) == 0
    capture = json.loads((tmp_path / "capture.json").read_text())
    assert capture["slug"] == "minnetonka-tree" and capture["licenseName"] == "CC-BY-4.0"
    assert (capture["latitude"], capture["longitude"]) == (tree.LATITUDE, tree.LONGITUDE)
