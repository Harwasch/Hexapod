"""Candidate A, ground first (`segment_ground_first`), with masks from ground truth: a spool
whose bottom flange is a board on the ground (what the ground pass alone calls ground, and
the refine pass must give back to the spool), and the synthetic yard (things above, ground
classes below)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import ground_pass as gp
import segment_ground_first as sgf
import segment_scene as ss
import synthetic_yard
from splat_render import Splats

VOCABULARY = ["tree", "shrub", "dead tree", "house", "grass", "dirt", "path", "spool"]


class ColourEmbedder:
    """Green images are grass, the rest gravel: texts by the cover word they hold."""

    name = "colour"
    dim = 4

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images), self.dim))
        for k, image in enumerate(images):
            rgb = np.asarray(image, np.float64).reshape(-1, 3)
            rgb = rgb[rgb.sum(axis=1) > 30]
            mean = rgb.mean(axis=0) if len(rgb) else np.zeros(3)
            out[k, 0 if mean[1] > max(mean[0], mean[2]) * 1.05 else 1] = 1.0
        return out

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for k, text in enumerate(texts):
            if "lawn" in text or "short grass" in text or text == "a photo of grass":
                out[k, 0] = 1.0
            elif "gravel" in text:
                out[k, 1] = 1.0
            else:
                out[k, 2 + k % 2] = 1.0
        return out


class StubNamer:
    name = "stub"

    def __init__(self) -> None:
        self.asked = 0

    def name_objects(self, crops):
        self.asked += len(crops)
        return [{"name": "thing", "whole": True, "material": "wood", "movable": True}] * len(crops)


def _splats(points: np.ndarray, colours: np.ndarray, scale: float = 0.012) -> Splats:
    n = len(points)
    rotations = np.zeros((n, 4))
    rotations[:, 0] = 1.0
    return Splats(points, rotations, np.full((n, 3), scale), colours, np.full(n, 0.9))


def _spool(seed: int = 4) -> tuple[Splats, np.ndarray, np.ndarray]:
    """A cable spool standing on its bottom flange, a 2 cm board on grass: per splat its
    part (0 bottom flange, 1 drum, 2 top flange, -1 ground)."""
    rng = np.random.default_rng(seed)
    parts, part = [], []

    def disc(r: float, z: float, n: int) -> np.ndarray:
        a, d = rng.random(n) * 2 * np.pi, r * np.sqrt(rng.random(n))
        return np.stack([d * np.cos(a), d * np.sin(a), np.full(n, z)], 1)

    def ring(r: float, z0: float, z1: float, n: int) -> np.ndarray:
        a = rng.random(n) * 2 * np.pi
        return np.stack([r * np.cos(a), r * np.sin(a), z0 + (z1 - z0) * rng.random(n)], 1)

    n = 60000
    x, y = rng.random(n) * 5 - 2.5, rng.random(n) * 5 - 2.5
    out = np.hypot(x, y) > 0.62
    ground = np.stack([x[out], y[out], 0.012 * rng.random(int(out.sum()))], 1)
    parts.append(ground)
    part.append(np.full(len(ground), -1))
    for k, chunk in enumerate(
        [
            np.concatenate([disc(0.6, 0.02, 3000), ring(0.6, 0.0, 0.02, 400)]),
            ring(0.25, 0.02, 0.62, 5000),
            np.concatenate(
                [disc(0.6, 0.64, 3000), disc(0.6, 0.62, 2500), ring(0.6, 0.62, 0.64, 400)]
            ),
        ]
    ):
        parts.append(chunk)
        part.append(np.full(len(chunk), k))
    points = np.concatenate(parts)
    labels = np.concatenate(part)
    colours = np.where(
        (labels < 0)[:, None],
        np.array([0.3, 0.55, 0.2]) * (0.8 + 0.4 * rng.random((len(points), 1))),
        np.array([0.62, 0.52, 0.38]) * (0.85 + 0.3 * rng.random((len(points), 1))),
    )
    colours[labels == 1] *= 0.8  # the drum a little darker than the flanges
    return _splats(points, np.clip(colours, 0, 1)), labels, points


@pytest.fixture(scope="module")
def spool() -> dict:
    splats, part, _ = _spool()
    objects = np.where(part >= 0, 0, 1)
    levels = [objects, np.where(part >= 0, part, 9)]
    common = {
        "view_count": 12,
        "workers": 1,
        "coverage_rounds": 0,
        "source_factory": lambda cameras: ss.OracleMasks(splats, levels, cameras),
    }
    namer = StubNamer()
    refined = sgf.run(
        splats, None, ColourEmbedder(), VOCABULARY,
        box_factory=lambda cameras: sgf.OracleBoxMasks(splats, objects, cameras),
        namer=namer, **common,
    )  # fmt: skip
    plain = sgf.run(splats, None, ColourEmbedder(), VOCABULARY, **common)
    return {"splats": splats, "part": part, "refined": refined, "plain": plain, "namer": namer}


def _top_ids(result: sgf.GroundFirst) -> np.ndarray:
    parent = np.array([i.parent or 0 for i in result.instances], np.int64)
    return ss.top_level(parent)[result.splat_id]


def test_the_ground_pass_alone_calls_the_flange_ground(spool) -> None:
    result = spool["plain"]
    flange = spool["part"] == 0
    assert (result.ground.label[flange] == gp.GROUND).mean() > 0.8


def test_the_refine_pass_gives_the_flange_back_to_the_spool(spool) -> None:
    result, part = spool["refined"], spool["part"]
    top = _top_ids(result)
    spool_id = np.bincount(top[part == 1]).argmax()
    assert spool_id > 0
    assert result.extra[int(spool_id)]["kind"] == "thing"
    for k in (0, 1, 2):
        assert (top[part == k] == spool_id).mean() > 0.85, k
    # ... and none of the ground goes with it.
    assert (top[part < 0] == spool_id).mean() < 0.01
    assert result.stats["refine"]["claimedCells"] > 0
    # Without the refine pass the flange stays ground.
    plain_top = _top_ids(spool["plain"])
    plain_id = np.bincount(plain_top[part == 1]).argmax()
    assert (plain_top[part == 0] == plain_id).mean() < 0.5


def test_the_ground_is_a_named_class_and_things_are_named(spool) -> None:
    result, part = spool["refined"], spool["part"]
    ground = [i for i in result.instances if result.extra[i.id]["kind"] == "ground"]
    assert ground
    assert all(i.category == "ground" for i in ground)
    assert {result.extra[i.id]["name"] for i in ground if i.parent is None} == {"Grass"}
    top = _top_ids(result)
    grass = next(i.id for i in ground if i.parent is None)
    assert (top[part < 0] == grass).mean() > 0.95
    things = [i for i in result.instances if result.extra[i.id]["kind"] == "thing"]
    named = [i for i in things if result.extra[i.id].get("nameSource") == "vlm"]
    assert named and spool["namer"].asked >= 1
    assert result.extra[named[0].id]["name"] == "Thing"
    assert result.extra[named[0].id]["material"] == "wood"


@pytest.fixture(scope="module")
def yard() -> dict:
    data, klass, inst, instances = synthetic_yard.generate_yard()
    p = data["position"].astype(np.float64)
    splats = Splats(
        p,
        data["quat_wxyz"],
        np.exp(data["log_scale"]),
        np.clip(data["rgb"], 0, 1),
        1 / (1 + np.exp(-data["opacity_logit"])),
    )
    truth = {
        "classes": list(synthetic_yard.CLASSES),
        "instances": instances,
        "class": klass.tolist(),
        "instance": inst.tolist(),
    }
    levels = ss.truth_levels(truth, np.arange(len(p)), splats.colours)
    result = sgf.run(
        splats, None, ColourEmbedder(), VOCABULARY,
        box_factory=lambda cameras: sgf.OracleBoxMasks(splats, levels[0], cameras),
        source_factory=lambda cameras: ss.OracleMasks(splats, levels, cameras),
        view_count=12, workers=1, coverage_rounds=0, refine_objects=12,
    )  # fmt: skip
    return {"result": result, "klass": klass, "inst": inst}


def test_the_yard(yard) -> None:
    result, klass, inst = yard["result"], yard["klass"], yard["inst"]
    names = synthetic_yard.CLASSES
    lawn = klass == names.index("grass/low")
    path = klass == names.index("ground")
    top = _top_ids(result)
    kind = np.array(["none"] + [result.extra[i.id]["kind"] for i in result.instances])
    name = np.array([""] + [result.extra[i.id].get("name", "") for i in result.instances])
    assert (kind[top[lawn | path]] == "ground").mean() > 0.9
    assert (name[top[lawn]] == "Grass").mean() > 0.8
    assert (name[top[path]] == "Gravel").mean() > 0.6, result.stats["cover"]
    # No thing holds more than 5% ground; each plant is mostly one thing.
    for t in np.unique(top[kind[top] == "thing"]):
        members = top == t
        assert (lawn | path)[members].mean() < 0.05
    for k in range(len(set(inst[inst >= 0].tolist()))):
        mine = inst == k
        ids, counts = np.unique(top[mine], return_counts=True)
        assert counts.max() / mine.sum() > 0.6
        assert kind[ids[counts.argmax()]] == "thing"


def test_the_document(yard, tmp_path) -> None:
    result = yard["result"]
    doc = sgf.document(result, {}, ColourEmbedder(), len(VOCABULARY), None)
    assert doc["format"] == "hexapod.instances" and doc["version"] == 1
    records = {r["id"]: r for r in doc["instances"]}
    for i in result.instances:
        record = records[i.id]
        assert record["kind"] in ("thing", "ground")
        if record["kind"] == "ground":
            assert record["category"] == "ground" and record["name"]
            assert record["tags"][0]["label"] == record["name"].lower()
        if record["parent"] is not None:
            assert record["parent"] < record["id"]
    assert doc["ground"]["cover"] and doc["variant"]["name"] == "ground-first"
    json.dumps(doc)
