"""`rescan_gate`: the fill gate reads the API's fill targets exactly, finds points in their
cells in the tileset frame, refuses what does not add up, and holds a published layer to them.

The cross-language half: `data/tiles/synthetic-yard-epochs/fill-targets.json` is what the
API's endpoint answers over the committed yard grids (apps/api tests/test_fill_targets_api.py
writes and checks it, reading its mask with `rescan_gate.unpack`); here the same file is read
with this project's own reader and held to a brute force over the same grids.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import gate_fixtures as gf
import numpy as np
import pytest

import cell_grid
import rescan_gate as rg

REPO = Path(__file__).resolve().parents[3]
EPOCHS = REPO / "data" / "tiles" / "synthetic-yard-epochs"
PAYLOAD = EPOCHS / "fill-targets.json"
FAR = REPO / "data" / "tiles" / "synthetic-far"


def _fixture(name: str) -> cell_grid.CellGrid:
    document = json.loads((EPOCHS / "grids.json").read_text(encoding="utf-8"))
    return cell_grid.decode(
        document["epochs"][name]["descriptor"], (EPOCHS / name / "confidence.bin").read_bytes()
    )


def _committed() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    return payload


def _brute_force(level: int) -> dict[tuple[int, int], bool]:
    """Every cell of e2's plane at `level`: weak and covered in e2, and weak in e1."""
    planes = {name: cell_grid.plane_at(_fixture(name), level) for name in ("e1", "e2")}

    def at(name: str, i: int, j: int) -> tuple[int, int]:
        plane = planes[name]
        if not (plane.i0 <= i < plane.i0 + plane.ni and plane.j0 <= j < plane.j0 + plane.nj):
            return 0, 0
        cell = plane.data[j - plane.j0, i - plane.i0]
        return int(cell[0]), int(cell[1])

    def weak(name: str, i: int, j: int) -> bool:
        score, reasons = at(name, i, j)
        return 0 < score < 128 or bool(reasons & 16)

    b = planes["e2"]
    return {
        (i, j): weak("e2", i, j) and weak("e1", i, j)
        for j in range(b.j0, b.j0 + b.nj)
        for i in range(b.i0, b.i0 + b.ni)
    }


def test_the_api_s_answer_reads_as_the_fixture_grids_say() -> None:
    payload = _committed()
    gate = rg.RescanGate.from_json(PAYLOAD)
    expected = _brute_force(payload["level"])
    found = {
        (gate.i0 + di, gate.j0 + dj): bool(gate.targets[dj, di])
        for dj in range(gate.nj)
        for di in range(gate.ni)
    }
    assert found == expected
    assert gate.cells == payload["cells"] == sum(expected.values()) > 0
    assert (gate.level, gate.cell_m) == (2, 1.0)
    assert gate.epoch_id == payload["epochId"] and gate.weak_in == tuple(payload["weakIn"])
    assert gate.mask_sha256 == hashlib.sha256(base64.b64decode(payload["mask"])).hexdigest()


def test_the_test_oracle_is_the_api_s_answer() -> None:
    """`gate_fixtures.still_weak_payload`, which the teacher_fill tests gate with, writes
    exactly what the API wrote for the same grids (its ids aside)."""
    committed = _committed()
    committed.pop("fixture")
    made = gf.still_weak_payload(
        _fixture("e2"),
        [_fixture("e1")],
        site=committed["siteId"],
        epoch=committed["epochId"],
        earlier_ids=committed["comparedWith"],
    )
    assert made == committed


def _payload(ni: int, nj: int, rows: list[bytes], **overrides: Any) -> dict[str, Any]:
    cells = sum(byte.bit_count() for row in rows for byte in row)
    out: dict[str, Any] = {
        "policy": "rescan-gate-v1",
        "siteId": gf.SITE,
        "epochId": gf.B,
        "weakIn": [gf.A] if cells else [],
        "comparedWith": [gf.A],
        "frame": "tileset",
        "cellM": 2.0,
        "level": 3,
        "i0": -3,
        "j0": 5,
        "ni": ni,
        "nj": nj,
        "rowBytes": (ni + 7) // 8,
        "bitOrder": "little",
        "mask": base64.b64encode(b"".join(rows)).decode("ascii"),
        "cells": cells,
        "reason": "test",
    }
    out.update(overrides)
    return out


def test_rows_run_south_to_north_and_bits_least_significant_first() -> None:
    """Ten cells a row (two bytes, six padding bits), three rows. Row 0 is j0 (south); in
    it bit 0 of byte 0 is i0, bit 1 of byte 1 is i0 + 9."""
    rows = [bytes([0b0000_0001, 0b0000_0010]), bytes([0, 0]), bytes([0b1000_0000, 0])]
    gate = rg.RescanGate.from_payload(_payload(10, 3, rows))
    on = {(gate.i0 + int(di), gate.j0 + int(dj)) for dj, di in zip(*np.nonzero(gate.targets))}
    assert on == {(-3, 5), (-3 + 9, 5), (-3 + 7, 5 + 2)}
    # The same cells by position: cell (i, j) of 2 m cells holds x in [2i, 2i + 2).
    centres = np.array([[2 * i + 1.0, 2 * j + 1.0, 0.0] for i, j in sorted(on)])
    assert gate.mask(centres).all()


def test_a_point_is_in_the_cell_its_floor_says_at_every_edge() -> None:
    """One target cell, i = -3 (x in [-6, -4)) and j = 5 (y in [10, 12)): its west and
    south edges belong to it, its east and north edges to its neighbours; the window's
    own edges, negative indices, and points that are not finite."""
    gate = rg.RescanGate.from_payload(_payload(10, 3, [bytes([1, 0]), bytes(2), bytes(2)]))
    below = np.nextafter(-4.0, -np.inf)
    points = np.array(
        [
            [-6.0, 10.0, 0.0],  # the south-west corner: in
            [below, np.nextafter(12.0, -np.inf), 50.0],  # just inside the north-east: in
            [-4.0, 10.5, 0.0],  # the east edge is the next cell's
            [-5.0, 12.0, 0.0],  # the north edge is the next row's
            [np.nextafter(-6.0, -np.inf), 10.5, 0.0],  # west of the window
            [-5.0, np.nextafter(10.0, -np.inf), 0.0],  # south of the window
            [np.nan, 10.5, 0.0],
            [np.inf, np.inf, 0.0],
        ]
    )
    assert gate.mask(points).tolist() == [True, True, False, False, False, False, False, False]
    # The last cell of the window and the first past it.
    far = rg.RescanGate.from_payload(_payload(10, 3, [bytes(2), bytes(2), bytes([0, 0b10])]))
    assert far.mask([[2 * 6 + 1.0, 2 * 7 + 1.0, 0], [2 * 7 + 1.0, 2 * 7 + 1.0, 0]]).tolist() == [
        True,
        False,
    ]
    assert far.mask(np.zeros((0, 3))).shape == (0,)


def test_inside_keeps_a_point_only_when_it_stays_in_target_cells() -> None:
    """Two target cells side by side, i = -3 and -2 (x in [-6, -2)): the edge between them
    is no edge, the targets' own edges are held `INSET_M` inside."""
    gate = rg.RescanGate.from_payload(_payload(10, 3, [bytes([0b11, 0]), bytes(2), bytes(2)]))
    inset = rg.INSET_M
    points = np.array(
        [[-4.0, 11.0, 0], [-6.0 + 2 * inset, 11.0, 0], [-6.0 + inset / 2, 11.0, 0], [-3, 10.0, 0]]
    )
    assert gate.mask(points).tolist() == [True, True, True, True]
    assert gate.inside(points).tolist() == [True, True, False, False]


def test_tiles_with_their_own_origins_are_gated_in_the_site_frame() -> None:
    """The synthetic far site's tiles sit at origins up to 4.5 km out. `load_tileset`
    composes them, so a gate on 4 m cells around the north-east plot (2.6 km east, 1.8 km
    north) selects exactly the scene's gaussians there -- and the tile-local positions,
    were they used, would select none of them."""
    import synthetic_far
    from rig_tiles import glb_spz
    from splat_render import load_tileset
    from splat_tiles import unpack_spz

    targets = np.zeros((20, 20), dtype=bool)
    targets[8:12, 8:12] = True  # i 648..651, j 448..451: x in [2592, 2608), y in [1792, 1808)
    rows = np.packbits(targets.astype(np.uint8), axis=1, bitorder="little")
    gate = rg.RescanGate.from_payload(
        _payload(20, 20, [row.tobytes() for row in rows], level=4, cellM=4.0, i0=640, j0=440)
    )
    truth = synthetic_far.scene().positions.astype(np.float64)
    i, j = np.floor(truth[:, 0] / 4.0), np.floor(truth[:, 1] / 4.0)
    expected = int(((i >= 648) & (i <= 651) & (j >= 448) & (j <= 451)).sum())
    assert expected > 100
    loaded = load_tileset(FAR / "splat" / "tileset.json").positions
    assert int(gate.mask(loaded).sum()) == expected
    document = json.loads((FAR / "splat" / "tileset.json").read_text(encoding="utf-8"))
    stack, local = [document["root"]], []
    while stack:
        tile = stack.pop()
        stack.extend(tile.get("children", []))
        if not tile.get("children"):
            spz = unpack_spz(glb_spz(FAR / "splat" / tile["content"]["uri"]))
            local.append(np.column_stack([spz["x"], spz["y"], spz["z"]]))
    assert int(gate.mask(np.concatenate(local)).sum()) == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"policy": "weak-v1"}, "policy"),
        ({"frame": "site"}, "frame"),
        ({"bitOrder": "big"}, "bitOrder"),
        ({"cellM": 1.0}, "cellM"),
        ({"level": 31}, "level"),
        ({"level": True}, "integer"),
        ({"rowBytes": 1}, "rowBytes"),
        ({"mask": "AQ=="}, "bytes"),
        ({"mask": "!!"}, "base64"),
        ({"mask": base64.b64encode(bytes([1, 0b0100_0000]) + bytes(4)).decode()}, "past ni"),
        ({"cells": 2}, "cells"),
        ({"weakIn": []}, "weakIn"),
        ({"weakIn": [gf.B]}, "weakIn"),
        ({"comparedWith": []}, "no earlier"),
        ({"comparedWith": [gf.A, gf.B]}, "itself"),
        ({"epochId": "e2"}, "UUID"),
        ({"ni": -1}, "window"),
    ],
)
def test_a_payload_that_does_not_add_up_is_refused(change: dict[str, Any], message: str) -> None:
    payload = _payload(10, 3, [bytes([1, 0]), bytes(2), bytes(2)])
    rg.RescanGate.from_payload(payload)
    with pytest.raises(rg.RescanGateError, match=message):
        rg.RescanGate.from_payload({**payload, **change})


def test_capped_is_read_and_refuses_nothing() -> None:
    """`capped` (more earlier epochs than the API compares with) says nothing about the
    cells: the gate reads the same either way, and an answer without it is still read."""
    payload = _payload(10, 3, [bytes([1, 0]), bytes(2), bytes(2)])
    plain = rg.RescanGate.from_payload(payload)
    for capped in (True, False):
        gate = rg.RescanGate.from_payload({**payload, "capped": capped})
        assert np.array_equal(gate.targets, plain.targets)
        assert gate.mask_sha256 == plain.mask_sha256


def test_a_missing_field_or_file_is_refused(tmp_path: Path) -> None:
    payload = _payload(10, 3, [bytes([1, 0]), bytes(2), bytes(2)])
    del payload["bitOrder"]
    with pytest.raises(rg.RescanGateError, match="missing bitOrder"):
        rg.RescanGate.from_payload(payload)
    with pytest.raises(rg.RescanGateError, match="JSON object"):
        rg.RescanGate.from_payload([payload])
    with pytest.raises(rg.RescanGateError):
        rg.RescanGate.from_json(tmp_path / "absent.json")
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(rg.RescanGateError):
        rg.RescanGate.from_json(tmp_path / "broken.json")


def test_no_targets_is_a_gate_that_selects_nothing() -> None:
    gate = rg.RescanGate.from_payload(_payload(10, 3, [bytes(2), bytes(2), bytes(2)]))
    assert gate.cells == 0 and gate.weak_in == ()
    assert not gate.mask([[-5.0, 11.0, 0.0]]).any()
    empty = rg.RescanGate.from_payload(_payload(0, 0, [], cellM=2.0))
    assert (empty.ni, empty.nj, empty.cells) == (0, 0, 0)


def test_a_tileset_is_held_to_its_own_confidence_grid(tmp_path: Path) -> None:
    """The yard's answer binds to a tileset carrying e2's grid; one with none, a grid on
    another window, or one where the target cells are not weak, is refused."""
    gate = rg.RescanGate.from_json(PAYLOAD)
    e2 = _fixture("e2")
    sha = gate.check_tileset(gf.bare_tileset(tmp_path / "e2", e2))
    descriptor = json.loads((tmp_path / "e2" / "tileset.json").read_text())["root"]["extras"]
    assert sha == descriptor["confidence"]["sha256"]
    with pytest.raises(rg.RescanGateError, match="no confidence grid"):
        gate.check_tileset(gf.bare_tileset(tmp_path / "none", None))
    # Another epoch's grid, one cell further east: another window.
    level1 = e2.planes[0]
    shifted = gf.confidence_grid(
        1, level1.i0 + 2, level1.j0, level1.data[..., 0], level1.data[..., 1]
    )
    with pytest.raises(rg.RescanGateError, match="window"):
        gate.check_tileset(gf.bare_tileset(tmp_path / "shifted", shifted))
    # Same window, every cell strong: not where these targets came from.
    strong = gf.confidence_grid(
        1,
        level1.i0,
        level1.j0,
        np.full(level1.data.shape[:2], 200),
        np.zeros(level1.data.shape[:2]),
    )
    with pytest.raises(rg.RescanGateError, match="not weak"):
        gate.check_tileset(gf.bare_tileset(tmp_path / "strong", strong))
    # A grid finer than nothing the gate can be read at: the gate's level is finer.
    coarse = gf.confidence_grid(3, 0, 0, [[60]], [[1]])
    with pytest.raises(rg.RescanGateError, match="finer"):
        gate.check_tileset(gf.bare_tileset(tmp_path / "coarse", coarse))


def test_the_evidence_names_the_targets_and_the_grid_it_was_checked_against() -> None:
    gate = rg.RescanGate.from_json(PAYLOAD)
    evidence = gate.evidence(confidence_sha256="ab" * 32)
    payload = _committed()
    assert evidence == {
        "policy": "rescan-gate-v1",
        "siteId": payload["siteId"],
        "epochId": payload["epochId"],
        "weakIn": payload["weakIn"],
        "comparedWith": payload["comparedWith"],
        "level": 2,
        "cellM": 1.0,
        "i0": payload["i0"],
        "j0": payload["j0"],
        "ni": payload["ni"],
        "nj": payload["nj"],
        "cells": payload["cells"],
        "maskSha256": gate.mask_sha256,
        "confidenceSha256": "ab" * 32,
    }
    assert "confidenceSha256" not in gate.evidence()


# --- publish: verify_layer ---------------------------------------------------------------


def _publish_case() -> dict[str, Any]:
    payload = _committed()
    gate = rg.RescanGate.from_payload(payload)
    e2_sha = json.loads((EPOCHS / "grids.json").read_text())["epochs"]["e2"]["descriptor"]["sha256"]
    cells = np.argwhere(gate.targets)
    centres = np.array(
        [
            [(gate.i0 + i + 0.5) * gate.cell_m, (gate.j0 + j + 0.5) * gate.cell_m, 1.0]
            for j, i in cells
        ]
    )
    return {
        "evidence": {"kind": "inferred", "rescanGate": gate.evidence(confidence_sha256=e2_sha)},
        "asset": {"epochId": payload["epochId"], "siteId": payload["siteId"], "validTo": None},
        "tileset": {"root": {"extras": {"confidence": {"sha256": e2_sha}}}},
        "current": lambda site, epoch: payload,
        "positions": centres,
    }


def test_a_gated_layer_on_its_epoch_s_current_tiles_is_published() -> None:
    case = _publish_case()
    asked: list[tuple[str, str]] = []
    case["current"] = lambda site, epoch: asked.append((site, epoch)) or _committed()
    fresh = rg.verify_layer(**case)
    assert fresh.cells == _committed()["cells"]
    # The targets are fetched for the asset's own site and epoch, once.
    assert asked == [(_committed()["siteId"], _committed()["epochId"])]


def test_an_ungated_layer_is_refused_before_anything_is_fetched() -> None:
    case = _publish_case()
    case["evidence"] = {"kind": "inferred", "filler": "fixer"}

    def fetch(site: str, epoch: str) -> object:
        raise AssertionError("fetched for an ungated layer")

    case["current"] = fetch
    with pytest.raises(rg.RescanGateError, match="no rescanGate"):
        rg.verify_layer(**case)


def _broken(**changes: Any) -> dict[str, Any]:
    case = copy.deepcopy(_publish_case())
    for path, value in changes.items():
        where, _, key = path.partition("__")
        if key:
            case[where][key] = value
        else:
            case[where] = value
    return case


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"evidence": {"kind": "inferred"}}, "no rescanGate"),
        ({"asset__epochId": None}, "no epoch"),
        ({"asset__epochId": gf.A}, "revision of epoch"),
        ({"asset__validTo": "2026-10-20T00:00:00Z"}, "superseded"),
        ({"asset__siteId": gf.B}, "site"),
        ({"tileset": {"root": {"extras": {}}}}, "carries no confidence grid"),
        ({"tileset": {"root": {"extras": {"confidence": {"sha256": "0" * 64}}}}}, "not the one"),
        ({"current": lambda site, epoch: {**_committed(), "cells": 0}}, "cells"),
        ({"positions": np.array([[-100.0, -100.0, 0.0]])}, "outside"),
        ({"positions": np.zeros((0, 3))}, "no gaussians"),
    ],
)
def test_publish_refuses_an_ungated_stale_or_misplaced_layer(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(rg.RescanGateError, match=message):
        rg.verify_layer(**_broken(**changes))


def test_publish_refuses_a_layer_filled_for_targets_that_changed_since() -> None:
    """The epoch's targets now hold one cell fewer (a re-run, an earlier epoch measured
    since): the layer's own cells may all still be targets, but it is filled again."""
    case = _publish_case()
    payload = _committed()
    gate = rg.RescanGate.from_payload(payload)
    targets = gate.targets.copy()
    last = tuple(np.argwhere(targets)[-1])
    targets[last] = False
    rows = np.packbits(targets.astype(np.uint8), axis=1, bitorder="little")
    changed = {
        **payload,
        "mask": base64.b64encode(rows.tobytes()).decode(),
        "cells": payload["cells"] - 1,
    }
    case["current"] = lambda site, epoch: changed
    case["positions"] = case["positions"][:-1]
    with pytest.raises(rg.RescanGateError, match="changed since"):
        rg.verify_layer(**case)
