"""The Living Mode motion sidecar, from a rig — the known one and an extracted one.

``motion_params.py`` is mirrored by ``deriveMotionSidecar`` in packages/world; the TypeScript
test re-derives the committed sidecar and compares it field by field. What is checked here is
the Python side on its own terms, and — the part only Python can do — what the derivation makes
of a skeleton ``skeleton.py`` *extracted*, which is what a real capture will hand it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import motion_params
import skeleton
import splat_tiles

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "data" / "tiles" / "synthetic-tree" / "source"


def _rig() -> dict:
    return json.loads((FIXTURE / "rig.json").read_text(encoding="utf-8"))


def _sidecar() -> dict:
    return json.loads((FIXTURE / "motion.json").read_text(encoding="utf-8"))


def _branch_shares(sidecar: dict) -> dict[int, float]:
    sums: dict[int, float] = defaultdict(float)
    for i, (base, share) in enumerate(
        zip(sidecar["nodes"]["branch"], sidecar["nodes"]["share"], strict=True)
    ):
        if i:
            sums[base] += share
    return sums


def test_the_committed_rig_points_at_its_sidecar() -> None:
    rig = _rig()
    sidecar = _sidecar()
    assert rig["motion"] == "motion.json"
    assert sidecar["rigChecksum"] == rig["canonicalChecksum"]
    assert sidecar["nodeCount"] == len(rig["nodes"])
    assert sidecar["motionEvidence"] == "allometric"
    assert sidecar["provenance"] == motion_params.ALLOMETRIC_PROVENANCE


def test_the_committed_sidecar_is_what_the_generator_writes() -> None:
    rig = {k: v for k, v in _rig().items() if k != "motion"}
    data = splat_tiles.read_ply(FIXTURE / "splat.ply")
    positions = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
    assert motion_params.sidecar_for(rig, positions, scales) == _sidecar()


def test_every_branch_bend_sums_to_one_and_the_trunk_is_the_pendulum() -> None:
    sidecar = _sidecar()
    for total in _branch_shares(sidecar).values():
        assert total == pytest.approx(1, abs=1e-4)
    trunk = [i for i, mode in enumerate(sidecar["nodes"]["mode"]) if i and mode == 0]
    assert trunk, "the whole-tree mode must own at least one joint"
    law = motion_params.tree_frequency_hz(sidecar["treeHeightM"])
    for i in trunk:
        assert sidecar["nodes"]["frequencyHz"][i] == pytest.approx(law, abs=1e-4)
        assert sidecar["nodes"]["damping"][i] == motion_params.TREE_DAMPING_SUMMER


def test_coder_law_and_rounding_match_javascript() -> None:
    assert motion_params.branch_frequency_hz(1.0) == pytest.approx(2.55)
    assert motion_params.branch_frequency_hz(0.0) == pytest.approx(2.55 * 0.05**-0.59)
    # Math.round is half-up; Python's round() is banker's. The sidecar must use the former.
    assert motion_params.round_to(0.00005, 4) == 0.0001
    assert motion_params.round_to(2.5, 0) == 3


@pytest.fixture(scope="module")
def extracted(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("extracted")
    skeleton.extract(FIXTURE / "splat.ply", out, lat=28.0389, lon=-82.6966, tile=False)
    return {
        "rig": json.loads((out / "source" / "rig.json").read_text(encoding="utf-8")),
        "sidecar": json.loads((out / "source" / "motion.json").read_text(encoding="utf-8")),
    }


def test_an_extracted_skeleton_gets_a_usable_sidecar(extracted: dict) -> None:
    """The case a real capture is: joints inferred from geometry, radii mostly unmeasurable.
    Nothing in the derivation reads a radius, so the sidecar is as good as the topology."""
    rig, sidecar = extracted["rig"], extracted["sidecar"]
    assert rig["motion"] == "motion.json"
    assert sidecar["rigChecksum"] == rig["canonicalChecksum"]
    assert sidecar["nodeCount"] == len(rig["nodes"])
    for total in _branch_shares(sidecar).values():
        assert total == pytest.approx(1, abs=1e-4)
    frequencies = sidecar["nodes"]["frequencyHz"][1:]
    assert min(frequencies) > 0.2 and max(frequencies) < 15
    truth = _sidecar()
    branches = len(set(sidecar["nodes"]["branch"][1:]))
    true_branches = len(set(truth["nodes"]["branch"][1:]))
    print(
        f"MEASURED extracted skeleton: {len(rig['nodes'])} joints, {branches} branches "
        f"(truth {true_branches}); branch frequency median "
        f"{np.median(frequencies):.2f} Hz (truth {np.median(truth['nodes']['frequencyHz'][1:]):.2f}),"
        f" tree {sidecar['treeHeightM']} m / {truth['treeHeightM']} m, leaf "
        f"{sidecar['leafSizeM']} m / {truth['leafSizeM']} m"
    )
    # Height comes from the splats' extent, so extraction moves it only by what isolation
    # trims off the bottom (ground removal): 8.5 % here, 4 % in the pendulum frequency. A real
    # capture should pass a measured height (``--tree-height``) when it has one.
    assert sidecar["treeHeightM"] == pytest.approx(truth["treeHeightM"], rel=0.1)
    assert branches >= 10
