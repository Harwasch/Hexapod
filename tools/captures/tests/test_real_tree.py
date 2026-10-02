"""real_tree.py: a trained splat of a real tree into metres, one tree, a rig and site.json.

The end-to-end case is the synthetic tree standing on a lawn, with a hedge and some sky
floaters, turned into an arbitrary "pose set frame" by a known similarity -- the job the
real capture's frame.json undoes. Everything the step measures is checked against what
built the scene.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import real_tree
import rig_tiles
import skeleton
import splat_tiles
from synthetic_tree import generate_splats, synthetic_tree_rig, write_ply

REPO = Path(__file__).resolve().parents[3]


def random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def splat_dict(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`write_ply`'s input as the column dict `read_ply` returns."""
    sh = (np.asarray(data["rgb"]) - 0.5) / splat_tiles.SH_C0
    out = {
        "x": data["position"][:, 0],
        "y": data["position"][:, 1],
        "z": data["position"][:, 2],
        "opacity": data["opacity_logit"],
    }
    for i in range(3):
        out[f"f_dc_{i}"] = sh[:, i]
        out[f"scale_{i}"] = data["log_scale"][:, i]
    for i in range(4):
        out[f"rot_{i}"] = data["quat_wxyz"][:, i]
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def covariance(data: dict[str, np.ndarray], i: int) -> np.ndarray:
    w, x, y, z = (float(data[f"rot_{k}"][i]) for k in range(4))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    s = np.diag(np.exp([float(data[f"scale_{k}"][i]) for k in range(3)]))
    return r @ s @ s @ r.T


def test_a_gaussian_moves_as_an_ellipsoid_not_a_point() -> None:
    rng = np.random.default_rng(0)
    count = 5
    quats = rng.normal(size=(count, 4))
    data = splat_dict(
        {
            "position": rng.normal(size=(count, 3)),
            "rgb": np.full((count, 3), 0.5),
            "opacity_logit": np.zeros(count),
            "log_scale": np.log(rng.uniform(0.01, 0.5, (count, 3))),
            "quat_wxyz": quats,
        }
    )
    rotation, scale, shift = random_rotation(3), 0.4, np.array([1.0, -2.0, 3.0])
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = shift
    moved = real_tree.transform_splats(data, matrix)
    for i in range(count):
        np.testing.assert_allclose(
            covariance(moved, i),
            scale**2 * rotation @ covariance(data, i) @ rotation.T,
            rtol=1e-4,
            atol=1e-7,
        )
    xyz = real_tree.positions(data)
    np.testing.assert_allclose(
        real_tree.positions(moved), xyz @ (scale * rotation).T + shift, atol=1e-5
    )
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="reflection"):
        real_tree.transform_splats(data, mirror)


def ring(
    count: int, radius: float, centre: tuple[float, float], z: tuple[float, float], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * math.pi, count)
    r = radius * (1 + rng.normal(0, 0.04, count))
    return np.column_stack(
        [centre[0] + r * np.cos(angle), centre[1] + r * np.sin(angle), rng.uniform(*z, count)]
    )


def test_the_trunk_is_the_ring_most_splats_lie_on_not_the_post_beside_it() -> None:
    trunk = ring(600, 0.18, (0.4, -0.3), (0.3, 1.2), seed=1)
    post = ring(80, 0.04, (1.6, 0.9), (0.0, 1.2), seed=2)  # the marker post by the trunk
    haze = np.random.default_rng(3).uniform([-2.5, -2.5, 0.0], [2.5, 2.5, 1.2], (150, 3))
    xyz = np.vstack([trunk, post, haze])
    opacity = np.ones(len(xyz))
    measured = real_tree.measure_trunk(xyz, opacity, 0.0)
    assert measured["diameterM"] == pytest.approx(0.36, rel=0.06)
    assert np.hypot(measured["centre"][0] - 0.4, measured["centre"][1] + 0.3) < 0.02
    assert len(measured["slabs"]) == 3 and measured["centreSpreadM"] < 0.03
    with pytest.raises(ValueError, match="no trunk-shaped ring"):
        real_tree.measure_trunk(haze, np.ones(len(haze)), 0.0)


def test_the_ground_is_the_lawn_not_the_lowest_splat() -> None:
    rng = np.random.default_rng(4)
    lawn = np.column_stack([rng.uniform(-4, 4, (3000, 2)), rng.normal(0.62, 0.015, 3000)])
    below = np.column_stack([rng.uniform(-4, 4, (40, 2)), rng.uniform(-1.5, 0.5, 40)])
    above = np.column_stack([rng.uniform(-2, 2, (2000, 2)), rng.uniform(0.7, 7, 2000)])
    xyz = np.vstack([lawn, below, above])
    assert real_tree.ground_height(xyz, np.ones(len(xyz)), 5.0) == pytest.approx(0.62, abs=0.03)


def test_the_ground_is_below_the_lowest_camera_even_under_a_thick_trunk() -> None:
    """m0: a 1 m-thick trunk out-counted the lawn 3.5 m up it. The drone hovered 1.1 m
    above the lawn, so the lowest camera rules that layer out."""
    rng = np.random.default_rng(5)
    lawn = np.column_stack([rng.uniform(-5, 5, (3000, 2)), rng.normal(0.0, 0.03, 3000)])
    angle = rng.uniform(0, 2 * np.pi, 40000)
    height = np.where(
        rng.uniform(size=40000) < 0.5, rng.normal(3.5, 0.02, 40000), rng.uniform(0, 5, 40000)
    )
    trunk = np.column_stack([0.5 * np.cos(angle), 0.5 * np.sin(angle), height])
    xyz = np.vstack([lawn, trunk])
    opacity = np.ones(len(xyz))
    assert real_tree.ground_height(xyz, opacity, 5.0) == pytest.approx(3.5, abs=0.05)
    assert real_tree.ground_height(xyz, opacity, 5.0, below=1.1) == pytest.approx(0.0, abs=0.05)


def test_the_floater_rule_is_run_until_the_packer_would_drop_nothing(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    body = rng.normal(0, 1.0, (4000, 3))
    floaters = rng.normal(0, 1.0, (60, 3)) * np.array([[8.0, 8.0, 8.0]])
    xyz = np.vstack([body, floaters])
    keep = real_tree.floater_fixpoint(xyz)
    assert (~keep).sum() > 0
    kept = xyz[keep].astype(np.float32)
    count = len(kept)
    ply = tmp_path / "kept.ply"
    write_ply(
        ply,
        {
            "position": kept,
            "rgb": np.full((count, 3), 0.5),
            "opacity_logit": np.full(count, 2.0),
            "log_scale": np.full((count, 3), -4.0),
            "quat_wxyz": np.tile([1.0, 0, 0, 0], (count, 1)),
        },
    )
    stats = splat_tiles.convert(ply, tmp_path / "tiles", 0.0, 0.0, 0.0, tile_gaussians=None)
    assert stats["dropped"] == 0


def test_there_is_no_splat_count_to_cut_to() -> None:
    """m0 cut 1.07M isolated splats to 400k by opacity x area, which kept the big ones and
    dropped the fine ones. The count is now the viewer's, spent on tiles at runtime."""
    assert not hasattr(real_tree, "cap_splats") and not hasattr(real_tree, "DEFAULT_MAX_SPLATS")


def test_a_capture_descriptor_is_refused_by_what_it_lacks() -> None:
    assert real_tree.check_capture(dict(CAPTURE)) == CAPTURE
    lacking = {k: v for k, v in CAPTURE.items() if k not in ("latitude", "licenseName")}
    with pytest.raises(ValueError, match="latitude, licenseName"):
        real_tree.check_capture(lacking)


def test_the_days_the_site_says_it_was_flown() -> None:
    both = {"2020-07-18": 204, "2020-07-20": 451}
    assert real_tree.flight_days(both, "2020-07-20") == ("on 18 and 20 July 2020", "2020-07-20")
    assert real_tree.flight_days({"2020-07-18": 204}, "2020-07-20") == (
        "on 18 July 2020",
        "2020-07-18",
    )
    assert real_tree.flight_days(None, "2020-07-20") == ("on 20 July 2020", "2020-07-20")
    assert real_tree.flight_days({"2020-06-30": 1, "2020-07-01": 1}, "")[0] == (
        "on 30 June 2020 and 1 July 2020"
    )
    meta = {
        "The_Tree-0001.jpg": {"taken": "2020-07-18T10:00:00"},
        "The_Tree-0002.jpg": {"taken": "2020-07-20T09:00:00"},
        "The_Tree-0003.jpg": {"taken": "2020-07-20T09:00:05"},
        "The_Tree-0004.jpg": {"taken": None},
    }
    assert real_tree.days_of(meta) == {"2020-07-18": 1, "2020-07-20": 2}
    assert real_tree.days_of({}) is None and real_tree.days_of(None) is None


def tree_on_a_lawn(seed: int = 11) -> tuple[dict[str, np.ndarray], dict, float]:
    """The synthetic tree (metres, z up, trunk at the origin) with a lawn under it, a hedge
    at the edge of the crown's reach, a street tree outside the orbit, and sky floaters."""
    rig = synthetic_tree_rig(height_m=6.0)
    tree = generate_splats(rig, 9000, seed, 6.0)
    rng = np.random.default_rng(seed)
    parts = [{k: tree[k] for k in ("position", "rgb", "opacity_logit", "log_scale", "quat_wxyz")}]

    def blob(xyz: np.ndarray, colour: tuple[float, float, float], log_scale: float) -> dict:
        n = len(xyz)
        return {
            "position": xyz,
            "rgb": np.tile(colour, (n, 1)),
            "opacity_logit": np.full(n, 3.0),
            "log_scale": np.full((n, 3), log_scale),
            "quat_wxyz": np.tile([1.0, 0, 0, 0], (n, 1)),
        }

    lawn_xy = rng.uniform(-10, 10, (20000, 2))
    parts.append(
        blob(np.column_stack([lawn_xy, rng.normal(0.0, 0.01, 20000)]), (0.2, 0.5, 0.2), -3.2)
    )
    hedge = np.column_stack(
        [rng.uniform(-6, 6, 1500), rng.normal(-6.5, 0.3, 1500), rng.uniform(0, 1.6, 1500)]
    )
    parts.append(blob(hedge, (0.1, 0.4, 0.1), -3.0))
    street = np.column_stack(
        [rng.normal(14, 1.5, 2000), rng.normal(3, 1.5, 2000), rng.uniform(1, 8, 2000)]
    )
    parts.append(blob(street, (0.1, 0.3, 0.1), -3.0))
    sky = rng.uniform(-60, 60, (40, 3)) + np.array([0, 0, 60.0])
    parts.append(blob(sky, (0.8, 0.8, 0.9), -1.0))
    merged = {
        key: np.concatenate([np.asarray(p[key], dtype=np.float64) for p in parts])
        for key in parts[0]
    }
    trunk_radius = float(rig["nodes"][0]["radius"])
    return merged, rig, trunk_radius


def into_model_frame(metric: dict[str, np.ndarray], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """What a trainer hands back: the scene in the pose set's frame (the inverse of the
    similarity frame.json records)."""
    return real_tree.transform_splats(splat_dict(metric), np.linalg.inv(matrix))


#: A capture descriptor as `real_tree.py --capture` reads it: everything about the capture
#: that is not in its splat or its poses. (The Minnetonka tree's is
#: `infra/modal/minnetonka.py describe`; nothing in real_tree.py is about it.)
CAPTURE = {
    "slug": "a-street-tree",
    "name": "A street tree (real capture)",
    "subject": "one street tree in a test",
    "photos": "drone photographs",
    "capturedBy": "A. Pilot",
    "conditions": "in still air",
    "dataset": "A Test Dataset, CC BY 4.0",
    "latitude": 44.944565,
    "longitude": -93.425903,
    "geoidUndulationM": -27.6,
    "attribution": "A. Pilot",
    "attributionUrl": "https://example.org/pilot",
    "licenseName": "CC-BY-4.0",
    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
    "sourceUrl": "https://example.org/dataset",
    "captured": "2020-07-20",
}


def frame_for(matrix: np.ndarray, metres_per_unit: float) -> dict:
    cameras = []
    for tier, (height, radius) in enumerate(((4.9, 9.0), (6.4, 9.5), (1.4, 7.0))):
        for step in range(36):
            a = 2 * math.pi * (step + tier / 3) / 36
            cameras.append([radius * math.sin(a), radius * math.cos(a), height])
    return {
        "verdict": "pass",
        "failed": [],
        "matrix": matrix.tolist(),
        "metresPerUnit": metres_per_unit,
        "camerasM": cameras,
        "focalPx": 1174.0,
        "imageSize": [1600, 1066],
        "takeoffMslM": 282.0,
    }


def write_model_ply(path: Path, data: dict[str, np.ndarray]) -> None:
    write_ply(
        path,
        {
            "position": real_tree.positions(data).astype(np.float32),
            "rgb": np.stack([data[f"f_dc_{i}"] for i in range(3)], axis=1) * splat_tiles.SH_C0
            + 0.5,
            "opacity_logit": data["opacity"],
            "log_scale": np.stack([data[f"scale_{i}"] for i in range(3)], axis=1),
            "quat_wxyz": np.stack([data[f"rot_{i}"] for i in range(4)], axis=1),
        },
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, float]:
    tmp = tmp_path_factory.mktemp("real-tree")
    metric, _, trunk_radius = tree_on_a_lawn()
    metres_per_unit = 0.41
    matrix = np.eye(4)
    matrix[:3, :3] = metres_per_unit * random_rotation(9)
    ply = tmp / "trained.ply"
    write_model_ply(ply, into_model_frame(metric, matrix))
    # The frame's origin is the axis the orbit looked at, 36 cm off the trunk.
    matrix[:3, 3] = [0.3, -0.2, 0.0]
    out = tmp / "minnetonka-tree"
    report = real_tree.build(
        ply,
        out,
        frame_for(matrix, metres_per_unit),
        CAPTURE,
        # m1's shape: trained at twice the posed frames' size, on two days' photos.
        train={
            "psnr": 24.3,
            "registered": 108,
            "frameSize": [3200, 2132],
            "days": {"2020-07-18": 40, "2020-07-20": 68},
        },
        pose={"registered": 120},
        # Small leaves, so ~8k splats make a real hierarchy (the default is 100k a leaf).
        tile_gaussians=1500,
    )
    return out, report, trunk_radius


def test_the_tree_is_measured_isolated_and_stands_up(built: tuple[Path, dict, float]) -> None:
    _, report, trunk_radius = built
    assert report["groundM"] == pytest.approx(0.0, abs=0.03)
    # The origin went to the trunk, not to where the orbit's axis happened to be.
    assert np.hypot(*report["trunk"]["centre"]) == pytest.approx(math.hypot(0.3, 0.2), abs=0.04)
    assert report["trunk"]["diameterM"] == pytest.approx(2 * trunk_radius, rel=0.2)
    assert report["uprightness"]["upright"] and report["uprightness"]["crownToBase"] >= 3
    assert report["crown"]["radiusM"] < 6.0  # the hedge at 6.5 m and the street tree are out
    iso = report["isolation"]
    assert 7000 < iso["kept"] <= 9000  # the tree, less its bottom 20 cm and haze
    assert report["rig"]["splats"] == iso["kept"]  # every isolated splat, no cap
    assert report["rig"]["nodes"] > 10
    assert report["gsdM"] is not None and 0.001 < report["gsdM"] < 0.01


def test_the_gsd_is_at_the_size_the_splat_trained_at(tmp_path: Path) -> None:
    """frame.json's focal is the posed frames' (1600 px); trained on frames twice that
    size, a training pixel is half as big, and the site says which size it means."""
    metric, _, _ = tree_on_a_lawn(seed=14)
    matrix = np.eye(4)
    matrix[:3, :3] = 0.41 * random_rotation(6)
    ply = tmp_path / "trained.ply"
    write_model_ply(ply, into_model_frame(metric, matrix))
    frame = frame_for(matrix, 0.41)
    posed = real_tree.build(ply, tmp_path / "a", frame, CAPTURE, tile=False)
    doubled = real_tree.build(
        ply, tmp_path / "b", frame, CAPTURE, train={"frameSize": [3200, 2132]}, tile=False
    )
    assert (posed["trainFramePx"], doubled["trainFramePx"]) == (1600, 3200)
    assert doubled["gsdM"] == pytest.approx(posed["gsdM"] / 2, rel=1e-6)
    site = json.loads((tmp_path / "b" / "site.json").read_text())
    assert "of the 3200 px training frames" in site["assets"][0]["description"]


def _sorted_rows(positions: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(positions, dtype="<f4").view("<u4").reshape(-1, 3)
    return bits[np.lexsort(bits.T[::-1])]


def test_every_isolated_splat_is_a_leaf_of_a_stamped_lod_tileset(
    built: tuple[Path, dict, float],
) -> None:
    """LOD by default: the leaves hold every isolated splat once, merged parents above
    them, and the rig carries every tile's checksum with its motion sidecar beside it --
    what apps/web/src/cesium/splatTiles.ts needs to move a multi-tile tileset."""
    out, report, _ = built
    tileset = json.loads((out / "splat" / "tileset.json").read_text())
    assert tileset["root"].get("children"), "a hierarchy, not one tile"
    assert report["tiles"]["tiles"] == len(rig_tiles.tile_uris(tileset)) > 1
    assert report["tiles"]["parentGaussians"] > 0
    rig = json.loads((out / "source" / "rig.json").read_text())
    assert skeleton.rig_issues(rig) == []
    assert rig["tileChecksums"] == rig_tiles.tile_checksums(out / "splat")
    assert (out / "source" / rig["motion"]).is_file()
    motion = json.loads((out / "source" / rig["motion"]).read_text())
    assert motion["rigChecksum"] == rig["canonicalChecksum"]
    positions = np.frombuffer((out / "source" / "positions.f32").read_bytes(), dtype="<f4")
    canonical = positions.reshape(-1, 3)
    assert len(canonical) == report["rig"]["splats"]
    leaves: list[np.ndarray] = []

    def walk(tile: dict) -> None:
        if not tile.get("children"):
            leaves.append(rig_tiles.tile_positions(out / "splat" / tile["content"]["uri"]))
        for child in tile.get("children", []):
            walk(child)

    walk(tileset["root"])
    assert np.array_equal(_sorted_rows(np.concatenate(leaves)), _sorted_rows(canonical))
    assert "{nodes}" not in rig["sourceNote"] and CAPTURE["sourceUrl"] in rig["sourceNote"]
    assert real_tree.uprightness(canonical.astype(np.float64))["upright"]
    assert not (out / "source" / "isolated.ply").exists()


def test_one_tile_is_still_a_choice(tmp_path: Path) -> None:
    metric, _, _ = tree_on_a_lawn(seed=13)
    matrix = np.eye(4)
    matrix[:3, :3] = 0.41 * random_rotation(5)
    ply = tmp_path / "trained.ply"
    write_model_ply(ply, into_model_frame(metric, matrix))
    out = tmp_path / "out"
    report = real_tree.build(ply, out, frame_for(matrix, 0.41), CAPTURE, tile_gaussians=None)
    tileset = json.loads((out / "splat" / "tileset.json").read_text())
    assert "children" not in tileset["root"] and report["tiles"]["tiles"] == 1
    rig = json.loads((out / "source" / "rig.json").read_text())
    assert rig["tileChecksums"] == [rig["canonicalChecksum"]]


def asset_description(site: dict) -> str:
    return str(site["assets"][0]["description"])


def test_site_json_is_the_capture_shape_the_seeder_reads(built: tuple[Path, dict, float]) -> None:
    out, _, _ = built
    site = json.loads((out / "site.json").read_text())
    legacy = json.loads(
        (REPO / "apps" / "api" / "app" / "seed" / "legacy_captures.json").read_text()
    )
    entries = legacy["captures"]
    allowed = set().union(*(e.keys() for e in entries))
    assert set(site) <= allowed
    asset_keys = set().union(*(a.keys() for e in entries for a in e["assets"])) | {"rig"}
    (asset,) = site["assets"]
    assert set(asset) <= asset_keys | {"point_spacing_m", "ground_sample_distance_m"}
    assert site["slug"] == CAPTURE["slug"] and site["name"] == CAPTURE["name"]
    assert site["attribution"] == CAPTURE["attribution"]
    assert site["attribution_url"] == CAPTURE["attributionUrl"]
    assert site["license_name"] == "CC-BY-4.0"
    assert site["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert site["source_url"] == CAPTURE["sourceUrl"]
    assert site["captured"] == "2020-07-20"
    assert site["images"] == 108  # what it trained on, not the 120 the poses registered
    assert "photogrammetric reconstruction of a real tree" in site["description"]
    assert "motion is simulated" in site["description"]
    assert "on 18 and 20 July 2020 in still air" in site["description"]
    assert "one street tree in a test" in site["description"]
    assert "level-of-detail tileset" in asset_description(site)
    assert "3200 px training frames" in asset_description(site)
    assert asset["representation"] == "gaussian-splat"
    assert asset["path"] == "splat/tileset.json" and asset["rig"] == "../source/rig.json"
    assert asset["ground_sample_distance_m"] > 0 and asset["clamp_to_ground"] is True
    assert site["boundary"][0] == site["boundary"][-1] and len(site["boundary"]) == 17
    lon, lat, height = site["center"]
    assert (lat, lon) == pytest.approx((44.944565, -93.425903))
    assert height == pytest.approx(282.0 + CAPTURE["geoidUndulationM"], abs=0.1)
    assert "24.30 dB" in site["pipeline"]


def test_a_measured_trunk_can_set_the_scale(tmp_path: Path) -> None:
    metric, _, trunk_radius = tree_on_a_lawn(seed=12)
    wrong = 0.5  # the frame's scale is 22 % high; the trunk says by how much
    matrix = np.eye(4)
    matrix[:3, :3] = 0.41 * random_rotation(4)
    ply = tmp_path / "trained.ply"
    write_model_ply(ply, into_model_frame(metric, matrix))
    matrix[:3, :3] *= wrong / 0.41
    frame = frame_for(matrix, wrong)
    frame["camerasM"] = (np.asarray(frame["camerasM"]) * wrong / 0.41).tolist()
    report = real_tree.build(
        ply, tmp_path / "out", frame, CAPTURE, trunk_diameter_m=2 * trunk_radius, tile=False
    )
    assert report["scale"]["method"] == "trunk"
    assert report["scale"]["metresPerUnit"] == pytest.approx(0.41, rel=0.2)
    assert report["trunk"]["diameterScaledM"] == pytest.approx(2 * trunk_radius, abs=1e-3)
    with pytest.raises(ValueError, match="not both"):
        real_tree.build(
            ply, tmp_path / "x", frame, CAPTURE, metres_per_unit=0.4, trunk_diameter_m=0.3
        )
