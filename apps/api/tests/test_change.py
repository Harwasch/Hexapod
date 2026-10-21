"""The change between two epochs from their height surfaces (`app/analysis/change.py`, A5).

Synthetic surface planes built here (a crate on flat ground, noise, holes, an unaligned
epoch, weak confidence), and the committed fixture (data/tiles/synthetic-yard-surfaces) for
P5's own reference: its demo's `coarsen_surface` and `surface_change` over the same grids.
The rule's mechanics are tested on the top surface with no motion term (`change_planes`
here); the two layers, ground and vegetation, after them.
"""

from __future__ import annotations

import hashlib
import math
import tracemalloc
from typing import Any

import numpy as np
import pytest

from app.analysis import cell_grid, change
from app.analysis.change import Surface, change_areas, coarsen_surface
from app.analysis.change import change_map as _change_map
from app.analysis.change import change_planes as _change_planes
from app.analysis.weak_areas import site_projection
from tests.confidence_fixtures import blob, descriptor
from tests.surface_fixtures import DOCUMENT, SURFACES, surface_blob, surface_descriptor

NO = cell_grid.NO_HEIGHT
PROJECTION = site_projection(44.7965, -111.1346, 1520.0)


def flat(
    level: int = 2,
    i0: int = 0,
    j0: int = 0,
    ni: int = 20,
    nj: int = 20,
    height_cm: int = 1000,
    sigma_cm: int = 2,
    ground_sigma_cm: int = 4,
) -> dict[str, Any]:
    return {
        "level": level,
        "i0": i0,
        "j0": j0,
        "ni": ni,
        "nj": nj,
        "dsm": np.full((nj, ni), height_cm, dtype=np.int16),
        "ground": np.full((nj, ni), height_cm, dtype=np.int16),
        "sigma": np.full((nj, ni), sigma_cm, dtype=np.uint8),
        "support": np.full((nj, ni), 3 | (4 << 4), dtype=np.uint8),
        "ground_sigma": np.full((nj, ni), ground_sigma_cm, dtype=np.uint8),
    }


def surface(fields: dict[str, Any]) -> Surface:
    return Surface(**fields)


def change_planes(a: Surface, b: Surface, ua: float, ub: float, **kwargs: Any) -> Any:
    """The top surface's change with no motion term: the rule's own mechanics."""
    kwargs.setdefault("layer", "vegetation")
    kwargs.setdefault("motion_m", 0.0)
    return _change_planes(a, b, ua, ub, **kwargs)


def change_map(a: Surface, b: Surface, ua: float, ub: float, *args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("layer", "vegetation")
    return _change_map(a, b, ua, ub, *args, **kwargs)


def crate(fields: dict[str, Any], rows: slice, columns: slice, tall_cm: int) -> dict[str, Any]:
    out = {**fields, "dsm": fields["dsm"].copy()}
    out["dsm"][rows, columns] += tall_cm
    return out


def hole(fields: dict[str, Any], rows: slice, columns: slice) -> dict[str, Any]:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in fields.items()}
    out["dsm"][rows, columns] = NO
    out["ground"][rows, columns] = NO
    out["sigma"][rows, columns] = 0
    out["support"][rows, columns] = 0
    out["ground_sigma"][rows, columns] = 0
    return out


def confidence(
    level: int, i0: int, j0: int, score: np.ndarray, reasons: np.ndarray | None = None
) -> cell_grid.Grid:
    nj, ni = score.shape
    plane = cell_grid.Plane(
        level,
        i0,
        j0,
        ni,
        nj,
        {
            "score": score.astype(np.uint8),
            "reasons": (reasons if reasons is not None else np.zeros_like(score)).astype(np.uint8),
            "relief": np.zeros_like(score, dtype=np.uint8),
        },
    )
    return cell_grid.Grid(kind="confidence", planes=(plane,), descriptor={})


def strong(fields: dict[str, Any]) -> cell_grid.Grid:
    return confidence(
        fields["level"],
        fields["i0"],
        fields["j0"],
        np.full((fields["nj"], fields["ni"]), 200),
    )


def classes_at(planes: change.ChangePlanes, name: str) -> set[tuple[int, int]]:
    rows, columns = np.nonzero(planes.classes == change.CLASSES.index(name))
    return {(int(planes.i0 + c), int(planes.j0 + r)) for r, c in zip(rows, columns, strict=True)}


def test_a_crate_is_raised_and_nothing_else_changed() -> None:
    a = flat()
    b = crate(flat(), slice(5, 8), slice(10, 13), 150)
    planes = change_planes(
        surface(a), surface(b), 0.0, 0.062, confidence_a=strong(a), confidence_b=strong(b)
    )
    assert planes.level == 2 and planes.cell_m == 1.0
    assert classes_at(planes, "raised") == {(i, j) for i in range(10, 13) for j in range(5, 8)}
    assert planes.counts() == {"unchanged": 391, "raised": 9, "lowered": 0, "insufficient": 0}
    assert planes.low_counts() == {"unchanged": 0, "raised": 0, "lowered": 0}
    # 3 sqrt(2 * 0.02^2 + 0.062^2): the least change seen here.
    assert planes.min_detectable_m == pytest.approx(3 * math.sqrt(2 * 0.02**2 + 0.062**2))

    [raised] = change_areas(planes, PROJECTION)  # unchanged cells are counted, not outlined
    assert (raised.change, raised.cells, raised.low_confidence) == ("raised", 9, False)
    assert raised.metric.area == pytest.approx(9.0)
    assert raised.mean_delta_m == pytest.approx(1.5) and raised.max_abs_delta_m == 1.5


def test_a_lowered_cell_and_its_sign() -> None:
    a = crate(flat(), slice(2, 4), slice(2, 4), 120)
    b = flat()
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert classes_at(planes, "lowered") == {(2, 2), (3, 2), (2, 3), (3, 3)}
    [lowered, *_] = change_areas(planes, PROJECTION)
    assert lowered.change == "lowered" and lowered.mean_delta_m == pytest.approx(-1.2)


def test_noise_within_three_sigma_is_unchanged() -> None:
    rng = np.random.default_rng(5)
    a, b = flat(sigma_cm=4), flat(sigma_cm=4)
    # Up to 15 cm of difference against t = 3 sqrt(2 * 0.04^2) = 0.17 m.
    a["dsm"] = a["dsm"] + rng.integers(-7, 8, a["dsm"].shape).astype(np.int16)
    b["dsm"] = b["dsm"] + rng.integers(-8, 9, b["dsm"].shape).astype(np.int16)
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert planes.counts()["unchanged"] == 400
    assert int(np.abs(planes.delta_cm).max()) >= 10  # the noise is there, and within t


def test_one_side_without_a_surface_is_insufficient() -> None:
    a = hole(flat(), slice(0, 20), slice(0, 4))
    b = hole(crate(flat(), slice(0, 20), slice(0, 2), 300), slice(15, 20), slice(15, 20))
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    insufficient = classes_at(planes, "insufficient")
    expected = {(i, j) for i in range(4) for j in range(20)} | {
        (i, j) for i in range(15, 20) for j in range(15, 20)
    }
    assert insufficient == expected
    # The 3 m in b's west strip is where a has no surface: insufficient, never raised.
    assert planes.counts()["raised"] == 0


def test_neither_side_has_no_class_and_a_window_s_outside_is_insufficient() -> None:
    a = flat(ni=10)
    b = flat(i0=5, ni=10)
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert (planes.i0, planes.ni) == (0, 15)
    counts = planes.counts()
    assert counts["insufficient"] == 10 * 20 and counts["unchanged"] == 5 * 20
    both_holes = hole(flat(), slice(0, 2), slice(0, 2))
    planes = change_planes(surface(both_holes), surface(both_holes), 0.0, 0.0)
    assert int(np.count_nonzero(planes.classes == change.NONE)) == 4


def test_an_unaligned_epoch_cannot_see_a_crate_and_says_how_much_it_could() -> None:
    """An epoch placed by GPS alone at +/-10 m: the threshold is ~30 m, so a 1.5 m crate is
    not a change it can see -- and nothing is called changed."""
    a = flat()
    b = crate(flat(), slice(5, 8), slice(10, 13), 150)
    planes = change_planes(surface(a), surface(b), 0.0, 10.0)
    assert planes.min_detectable_m == pytest.approx(30.0, abs=0.01)
    assert planes.counts() == {"unchanged": 400, "raised": 0, "lowered": 0, "insufficient": 0}
    assert planes.uncertainty_to_m == 10.0


def test_weak_cells_are_flagged_low_confidence_not_hidden() -> None:
    a = flat()
    b = crate(flat(), slice(5, 8), slice(10, 13), 150)
    score = np.full((20, 20), 200)
    score[5:8, 10:12] = 60  # two columns of the crate weak in a's grid
    reasons = np.zeros((20, 20))
    reasons[0, 0] = cell_grid.UNSEEN  # an unseen cell: not vouched for either
    score[0, 0] = 0
    planes = change_planes(
        surface(a),
        surface(b),
        0.0,
        0.0,
        confidence_a=confidence(2, 0, 0, score, reasons),
        confidence_b=strong(b),
    )
    assert planes.counts()["raised"] == 9  # still classified
    assert planes.low_counts() == {"unchanged": 1, "raised": 6, "lowered": 0}
    areas = change_areas(planes, PROJECTION)
    raised = [area for area in areas if area.change == "raised"]
    assert sorted((area.cells, area.low_confidence) for area in raised) == [(3, False), (6, True)]
    assert {str(area.change) for area in areas} == {"raised"}  # no unchanged area


def test_without_a_confidence_grid_nothing_is_vouched_for() -> None:
    planes = change_planes(surface(flat()), surface(flat()), 0.0, 0.0, confidence_a=None)
    assert planes.low_counts()["unchanged"] == 400


def test_a_saturated_sigma_is_insufficient_never_a_change() -> None:
    """255 cm means at least 2.55 m, so a threshold made of it is a lower bound only: two
    30 m floaters in a cell (P5's review) read as raised against 7.65 m when the true
    spread was past 45 m. Such a cell is not classified (it was, and flagged)."""
    a, b = flat(), flat()
    b["sigma"] = b["sigma"].copy()
    b["dsm"] = b["dsm"].copy()
    b["sigma"][3, 3], b["dsm"][3, 3] = 255, 4000
    b["sigma"][3, 4], b["dsm"][3, 4] = 254, 4000
    planes = change_planes(
        surface(a), surface(b), 0.0, 0.0, confidence_a=strong(a), confidence_b=strong(b)
    )
    assert classes_at(planes, "insufficient") == {(3, 3)}
    assert classes_at(planes, "raised") == {(4, 3)}  # 254 cm is a measurement
    assert int(planes.low.sum()) == 0


def test_heights_far_apart_do_not_wrap() -> None:
    """Two i2 heights 640 m apart: an i2 difference would wrap to -1.36 m and call it
    lowered; the change is computed wide."""
    a = flat(ni=2, nj=1, height_cm=-32000)
    b = flat(ni=2, nj=1, height_cm=32000)
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert planes.counts()["raised"] == 2
    assert int(planes.delta_cm[0, 0]) == 64000
    [area] = change_areas(planes, PROJECTION)
    assert area.mean_delta_m == pytest.approx(640.0)


def test_the_no_surface_sentinel_is_never_a_height() -> None:
    """A cell with NO_HEIGHT but a sigma (no grid P5 writes) is not believed: no surface."""
    a, b = flat(ni=3, nj=1), flat(ni=3, nj=1)
    b["dsm"] = b["dsm"].copy()
    b["dsm"][0, 1] = NO  # sigma left at 2 cm
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert classes_at(planes, "insufficient") == {(1, 0)}


def test_two_levels_compare_at_the_coarser_with_dsm_max_ground_min_sigma_max() -> None:
    """a at 1 m cells, b at 0.5 m: compared at 1 m. One 0.5 m child 2 m taller raises its
    1 m cell (dsm by max); a child's larger sigma is the coarse cell's (sigma by max)."""
    a = flat(level=2, ni=4, nj=4)
    b = flat(level=1, ni=8, nj=8)
    b["dsm"] = b["dsm"].copy()
    b["dsm"][0, 0] += 200
    b["ground"] = b["ground"].copy()
    b["ground"][1, 1] -= 50
    b["sigma"] = b["sigma"].copy()
    b["sigma"][7, 7] = 40
    coarse = coarsen_surface(surface(b))
    assert (coarse.level, coarse.ni, coarse.nj) == (2, 4, 4)
    assert int(coarse.dsm[0, 0]) == 1200 and int(coarse.ground[0, 0]) == 950
    assert int(coarse.sigma[3, 3]) == 40
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert (planes.level, planes.native_level) == (2, 2)
    assert classes_at(planes, "raised") == {(0, 0)}


def only(fields: dict[str, Any], measured: set[tuple[int, int]]) -> dict[str, Any]:
    """`fields` with a surface only in the cells `measured` ((row, column) of its plane)."""
    nj, ni = fields["dsm"].shape
    out = hole(fields, slice(0, 0), slice(0, 0))
    for r in range(nj):
        for c in range(ni):
            if (r, c) not in measured:
                out["dsm"][r, c], out["ground"][r, c], out["sigma"][r, c] = NO, NO, 0
                out["ground_sigma"][r, c] = 0
    return out


def test_a_coarse_cell_compares_only_the_children_both_epochs_measured() -> None:
    """The review's case: a measured all four children of a 1 m cell, one of them a 2 m
    crate; b measured one ground child. At 0.5 m that is insufficient where only a has a
    surface. At 1 m, b's one child must not stand for the whole cell against a's crate,
    and the one child both measured is not the cell either: fewer shared children than a
    measured is insufficient (review B5a; it was compared and only flagged)."""
    a = crate(flat(level=0, ni=2, nj=2), slice(0, 1), slice(0, 1), 200)
    b = only(flat(level=0, ni=2, nj=2), {(1, 1)})
    native = change_planes(surface(a), surface(b), 0.0, 0.0, confidence_a=strong(a))
    assert native.counts() == {"unchanged": 1, "raised": 0, "lowered": 0, "insufficient": 3}
    coarse = change_planes(
        surface(a),
        surface(b),
        0.0,
        0.0,
        confidence_a=strong(a),
        confidence_b=strong(flat(level=0, ni=2, nj=2)),
        level=1,
    )
    assert coarse.counts() == {"unchanged": 0, "raised": 0, "lowered": 0, "insufficient": 1}
    # Every child both measured: compared, the crate child the cell's top in both.
    whole = change_planes(
        surface(a), surface(a), 0.0, 0.0, confidence_a=strong(a), confidence_b=strong(a), level=1
    )
    assert whole.counts()["unchanged"] == 1 and whole.low_counts()["unchanged"] == 0


def test_a_coarse_cell_whose_children_neither_epoch_shares_is_insufficient() -> None:
    """On a slope, a's west children against b's east children read as a change of the
    slope's height; they share no child, so there is nothing to compare."""
    a, b = flat(level=0, ni=2, nj=2), flat(level=0, ni=2, nj=2)
    for f in (a, b):
        f["dsm"] = f["dsm"].copy()
        f["dsm"][:, 1] += 60  # 60 cm up the slope from the west column to the east
        f["ground"] = f["dsm"].copy()
    a = only(a, {(0, 0), (1, 0)})
    b = only(b, {(0, 1), (1, 1)})
    coarse = change_planes(surface(a), surface(b), 0.0, 0.0, level=1)
    assert coarse.counts() == {"unchanged": 0, "raised": 0, "lowered": 0, "insufficient": 1}


def test_on_a_slope_a_partial_child_set_is_insufficient() -> None:
    """a measured the whole 1 m cell on a slope, b only its lowest child: the one child is
    not the cell, so it is insufficient rather than compared like for like."""
    a, b = flat(level=0, ni=2, nj=2), flat(level=0, ni=2, nj=2)
    for f in (a, b):
        f["dsm"] = f["dsm"].copy()
        f["dsm"][:, 1] += 60
        f["dsm"][1, :] += 60
        f["ground"] = f["dsm"].copy()
    b = only(b, {(0, 0)})
    coarse = change_planes(surface(a), surface(b), 0.0, 0.0, level=1)
    assert coarse.counts() == {"unchanged": 0, "raised": 0, "lowered": 0, "insufficient": 1}


def test_a_coarser_epoch_against_part_of_its_cell_is_insufficient() -> None:
    """Review B5a: a at 1 m (one cell, its top a crate somewhere in it), b at 0.5 m
    measuring one ground child of it. a has nothing finer to compare, and b's one child is
    not the whole cell: insufficient, never "lowered" (it was classified, and flagged)."""
    a = crate(flat(level=1, ni=1, nj=1), slice(0, 1), slice(0, 1), 200)
    b = only(flat(level=0, ni=2, nj=2), {(1, 1)})
    planes = change_planes(
        surface(a),
        surface(b),
        0.0,
        0.0,
        confidence_a=strong(a),
        confidence_b=strong(flat(level=0, ni=2, nj=2)),
    )
    assert planes.level == 1
    assert planes.counts() == {"unchanged": 0, "raised": 0, "lowered": 0, "insufficient": 1}
    whole = change_planes(
        surface(a),
        surface(flat(level=0, ni=2, nj=2)),
        0.0,
        0.0,
        confidence_a=strong(a),
        confidence_b=strong(flat(level=0, ni=2, nj=2)),
    )
    assert whole.counts()["lowered"] == 1 and whole.low_counts()["lowered"] == 0


def test_windows_too_large_together_are_compared_coarser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(change, "MAX_CHANGE_CELLS", 100)
    a = flat(ni=8, nj=8)
    b = flat(i0=40, ni=8, nj=8)
    planes = change_planes(surface(a), surface(b), 0.0, 0.0)
    assert planes.native_level == 2 and planes.level > 2
    assert planes.ni * planes.nj <= 100


def test_a_noisy_map_is_drawn_coarser_within_the_outline_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkerboard of raised cells is one area per cell: over the edge bound it is
    classified again a level up and drawn there; the counts stay the compared level's."""
    monkeypatch.setattr(change, "MAX_ZONE_EDGES", 400)
    a = flat(level=0, ni=32, nj=32)
    b = {**a, "dsm": a["dsm"].copy()}
    jj, ii = np.indices(b["dsm"].shape)
    b["dsm"][(ii + jj) % 2 == 0] += 300  # past vegetation's own motion term (1.06 m)
    found = change_map(surface(a), surface(b), 0.0, 0.0, PROJECTION)
    assert found.planes.level == 0 and found.planes.counts()["raised"] == 512
    assert found.drawn.level > 0
    assert change.outline_edges(change.group_codes(found.drawn)) <= 400
    assert all(area.change == "raised" for area in found.areas)


def test_an_area_lies_over_its_cells_on_the_globe() -> None:
    a = flat(i0=-10, j0=-10)
    b = crate(flat(i0=-10, j0=-10), slice(5, 8), slice(10, 13), 150)
    [raised] = change_areas(change_planes(surface(a), surface(b), 0.0, 0.0), PROJECTION)
    [polygon] = raised.footprint.geoms
    lon, lat = np.asarray(polygon.exterior.coords).T
    x, y = PROJECTION.to_metric(lon, lat)
    corners = {(round(float(p), 6), round(float(q), 6)) for p, q in zip(x, y, strict=True)}
    # Cells i 0..2, j -5..-3 at 1 m: the square [0, 3] x [-5, -2], four corners only.
    assert corners == {(0.0, -5.0), (3.0, -5.0), (3.0, -2.0), (0.0, -2.0)}


@pytest.mark.parametrize("seed", range(25))
def test_every_outlined_cell_is_in_one_area_and_every_outline_is_valid(seed: int) -> None:
    """Random maps (holes, raised and lowered specks, random confidence, any window): the
    areas' cells add up to the changed and insufficient cells (unchanged ones are counted,
    not outlined), and every footprint is valid GeoJSON as the API answers it."""
    from app.services.change_maps import _footprint

    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 40))
    a = flat(
        level=int(rng.integers(0, 3)),
        i0=int(rng.integers(-50, 50)),
        j0=int(rng.integers(-50, 50)),
        ni=n,
        nj=n,
    )
    b = {**a, "dsm": (a["dsm"] + rng.choice([0, 0, 100, -100], size=(n, n))).astype(np.int16)}
    a = hole(a, slice(0, 0), slice(0, 0))
    gone = rng.random((n, n)) < 0.15
    a["dsm"][gone], a["ground"][gone], a["sigma"][gone] = NO, NO, 0
    a["ground_sigma"][gone] = 0
    grid = confidence(a["level"], a["i0"], a["j0"], rng.integers(0, 256, (n, n)))
    found = change_map(
        surface(a), surface(b), 0.0, 0.0, PROJECTION, confidence_a=grid, confidence_b=grid
    )
    outlined = (found.drawn.classes >= 0) & (found.drawn.classes != change.UNCHANGED)
    assert sum(area.cells for area in found.areas) == int(outlined.sum())
    assert {str(area.change) for area in found.areas} <= {"raised", "lowered", "insufficient"}
    for area in found.areas:
        assert _footprint(area.footprint).type == "MultiPolygon"


@pytest.mark.parametrize(("layer", "bound"), [("vegetation", 40), ("ground", 56)])
def test_change_planes_memory_is_bounded(layer: change.Layer, bound: int) -> None:
    """Bytes a cell at the peak, not the window held in float64 per array: the top surface
    with vegetation's 5-cell neighbourhoods measured 36.3 (2026-10-20, this machine:
    both sides over their shared cells, the outputs, what each side measured, the
    neighbourhoods); the ground, with its 1-cell neighbourhood and its local rise for the
    drift bound (a 3.5 m reach), 50.9. Each bound is its measurement plus about 10%."""
    side = 1024
    reach = 10 * 0.25 * math.sqrt(2.0)
    a = flat(level=0, ni=side, nj=side)
    b = crate(flat(level=0, ni=side, nj=side), slice(0, 100), slice(0, 100), 150)
    sa, sb = Surface(**a, ground_reach_m=reach), Surface(**b, ground_reach_m=reach)
    grid = strong(a)
    tracemalloc.start()
    _change_planes(sa, sb, 0.03, 0.062, layer=layer, confidence_a=grid, confidence_b=grid)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / (side * side) < bound, peak / (side * side)


# --- P5's reference, over the committed fixture -----------------------------------------


def fixture_surface(epoch: str) -> Surface:
    grid = cell_grid.decode(surface_descriptor(epoch), surface_blob(epoch))
    return Surface.of(grid.finest)


def fixture_confidence(epoch: str) -> cell_grid.Grid:
    return cell_grid.decode(descriptor(epoch), blob(epoch))


def test_the_api_coarsens_as_p5_s_demo_does() -> None:
    coarse = coarsen_surface(fixture_surface("e2"))
    expected = DOCUMENT["coarsened"]
    window = [coarse.i0, coarse.j0, coarse.ni, coarse.nj]
    assert (coarse.level, window) == (expected["level"], expected["window"])
    names = {"dsm": "dsm", "ground": "ground", "sigma": "sigma", "support": "support"}
    names["groundSigma"] = "ground_sigma"
    record = np.dtype(
        [
            (name, cell_grid.TYPES[kind])
            for name, kind in cell_grid.REQUIRED_CHANNELS["surface"].items()
        ]
    )
    rows = np.zeros((coarse.nj, coarse.ni), dtype=record)
    for name, field in names.items():
        rows[name] = getattr(coarse, field)
    assert hashlib.sha256(rows.tobytes()).hexdigest() == expected["sha256"]


def test_the_api_classifies_as_p5_s_demo_does() -> None:
    """Every class count and every changed cell (its class and delta) as P5's demo found
    them on the same grids; the cells the demo calls weak are all low confidence here. The
    demo compares cell for cell, so this does too (`horizontal=False`)."""
    reference = DOCUMENT["change"]
    ua = DOCUMENT["epochs"]["e1"]["uncertaintyM"]
    ub = DOCUMENT["epochs"]["e2"]["uncertaintyM"]
    planes = change_planes(
        fixture_surface("e1"),
        fixture_surface("e2"),
        ua,
        ub,
        confidence_a=fixture_confidence("e1"),
        confidence_b=fixture_confidence("e2"),
        horizontal=False,  # and no motion term (`change_planes` here): the demo's own rule
    )
    assert (planes.level, planes.cell_m) == (reference["level"], reference["cellM"])
    assert planes.counts() == reference["counts"]
    found = {}
    for name in ("raised", "lowered"):
        for i, j in classes_at(planes, name):
            delta = int(planes.delta_cm[j - planes.j0, i - planes.i0]) / 100
            found[(i, j)] = (name, round(delta, 2), bool(planes.low[j - planes.j0, i - planes.i0]))
    expected = {
        (r["i"], r["j"]): (r["class"], r["deltaM"], r["weak"]) for r in reference["changed"]
    }
    assert {k: v[:2] for k, v in found.items()} == {k: v[:2] for k, v in expected.items()}
    assert all(found[k][2] for k, v in expected.items() if v[2])
    # The shed and the spoil were seen well by both: not low confidence.
    assert planes.low_counts()["raised"] == reference["weakChanged"]


# --- horizontal uncertainty (review B5a) ---------------------------------------------------


def roof(columns: slice, level: int = 0, tall_cm: int = 400) -> dict[str, Any]:
    """Flat ground at 0 with a roof `tall_cm` up over rows 5..14 and `columns`, 0.25 m cells."""
    fields = flat(level=level, height_cm=0)
    fields["dsm"] = fields["dsm"].copy()
    fields["dsm"][5:15, columns] = tall_cm
    return fields


def test_a_roof_edge_one_cell_off_is_not_a_change_within_the_horizontal_uncertainty() -> None:
    """Review B5a: an unchanged 4 m roof, the re-scan placed one 25 cm cell east of the
    first, with 25 cm of declared uncertainty. Cell for cell its west edge reads as
    lowered 4 m and its east edge as raised; against the other epoch's neighbourhood
    within the horizontal radius (3 sigma: 0.75 m, 3 cells) nothing changed."""
    a, b = roof(slice(5, 12)), roof(slice(6, 13))
    planes = change_planes(surface(a), surface(b), 0.0, 0.25)
    assert planes.horizontal_cells == 3
    assert planes.counts()["raised"] == planes.counts()["lowered"] == 0
    # The same pair cell for cell (P5's demo rule): the edges are "change".
    naive = change_planes(surface(a), surface(b), 0.0, 0.25, horizontal=False)
    assert classes_at(naive, "lowered") == {(5, j) for j in range(5, 15)}
    assert classes_at(naive, "raised") == {(12, j) for j in range(5, 15)}


def test_a_post_put_up_or_taken_away_is_still_a_change_under_horizontal_uncertainty() -> None:
    """The neighbourhood explains a shift, never a new height: a 2 m post where the other
    epoch had only ground anywhere near is raised (put up) or lowered (taken away)."""
    ground = flat(level=0, height_cm=0)
    post = {**ground, "dsm": ground["dsm"].copy()}
    post["dsm"][10, 10] = 200
    up = change_planes(surface(ground), surface(post), 0.0, 0.25)
    assert classes_at(up, "raised") == {(10, 10)} and up.counts()["lowered"] == 0
    down = change_planes(surface(post), surface(ground), 0.0, 0.25)
    assert classes_at(down, "lowered") == {(10, 10)} and down.counts()["raised"] == 0
    # A roof raised by a metre is raised everywhere, its edges included.
    higher = change_planes(
        surface(roof(slice(5, 12))),
        surface(roof(slice(5, 12), tall_cm=500)),
        0.0,
        0.25,
    )
    assert classes_at(higher, "raised") == {(i, j) for i in range(5, 12) for j in range(5, 15)}


# --- the two layers (review B5a's ruling) ------------------------------------------------

#: The ground's reach at 1 m cells: P5's 5 m window is 5 cells, centres 2 m off.
REACH_1M = 2.0 * math.sqrt(2.0)


def canopy(
    fields: dict[str, Any], rows: slice, columns: slice, tall_cm: int, *, reach: float = REACH_1M
) -> Surface:
    """`fields` (dsm and ground at the same height) with a canopy `tall_cm` above its
    ground in the given cells: its top moves, its ground is the ground's."""
    out: dict[str, Any] = {
        k: v.copy() if isinstance(v, np.ndarray) else v for k, v in fields.items()
    }
    out["dsm"][rows, columns] = out["ground"][rows, columns] + tall_cm
    return Surface(**out, ground_reach_m=reach)


def test_wind_in_a_crown_is_never_ground_change_and_growth_is_vegetation_change() -> None:
    """The ruling: a crown that sways (its top 40 cm lower, its ground the same) changes
    neither layer; one that grew 2 m is vegetation raised, and the ground under it still
    unchanged. Vegetation's threshold carries its own motion: about 1.06 m at least."""
    a = canopy(flat(), slice(5, 9), slice(5, 9), 600)
    swayed = canopy(flat(), slice(5, 9), slice(5, 9), 560)
    grown = canopy(flat(), slice(5, 9), slice(5, 9), 800)
    for b, vegetation in ((swayed, 0), (grown, 16)):
        top = _change_planes(a, b, 0.0, 0.0, layer="vegetation")
        ground = _change_planes(a, b, 0.0, 0.0, layer="ground")
        assert top.counts()["raised"] == vegetation and top.counts()["lowered"] == 0
        assert ground.counts() == {"unchanged": 400, "raised": 0, "lowered": 0, "insufficient": 0}
    top = _change_planes(a, swayed, 0.0, 0.0, layer="vegetation")
    assert top.min_detectable_m == pytest.approx(3 * math.sqrt(2 * 0.02**2 + 2 * 0.25**2))
    assert top.motion_m == change.VEGETATION_MOTION_M and top.horizontal_cells == 2
    ground = _change_planes(a, grown, 0.0, 0.0, layer="ground")
    assert ground.min_detectable_m == pytest.approx(3 * math.sqrt(2 * 0.04**2))
    assert ground.motion_m == 0.0 and ground.horizontal_cells == 0


def test_ground_never_observed_is_insufficient_for_the_ground_and_compared_for_vegetation() -> None:
    """P5 writes no ground (NO_HEIGHT, groundSigma 0) under a surface where its window saw
    only canopy: the ground layer cannot compare there; the top surface still can."""
    a = Surface(**flat(), ground_reach_m=REACH_1M)
    fields = flat()
    fields["ground"] = fields["ground"].copy()
    fields["ground_sigma"] = fields["ground_sigma"].copy()
    fields["ground"][0:2, 0:3], fields["ground_sigma"][0:2, 0:3] = NO, 0
    b = Surface(**fields, ground_reach_m=REACH_1M)
    ground = _change_planes(a, b, 0.0, 0.0, layer="ground")
    assert ground.counts()["insufficient"] == 6 and ground.counts()["unchanged"] == 394
    assert _change_planes(a, b, 0.0, 0.0, layer="vegetation").counts()["unchanged"] == 400


def test_the_ground_envelope_drifting_down_a_slope_is_not_ground_change() -> None:
    """On a slope the ground is the least 2nd-lowest over its window, so it reads up to
    g R low, R the window's half-diagonal, and which cell it reads moves with what stands
    in the window: half a metre lower over 10 x 10 m on a 1 in 5 slope (R = 2.83 m) is
    within its drift. The same drop on flat ground is a change, away from its rim (where
    the step itself is a grade of 1 in 4 over the window's half-width, so drift too)."""
    for grade_cm, expected in ((20, 0), (0, 36)):
        fields = flat()
        slope = np.arange(20, dtype=np.int16)[None, :] * grade_cm
        fields["dsm"] = (fields["dsm"] + slope).astype(np.int16)
        fields["ground"] = fields["dsm"].copy()
        a = Surface(**fields, ground_reach_m=REACH_1M)
        drifted: dict[str, Any] = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in fields.items()
        }
        drifted["ground"][6:16, 6:16] -= 50
        b = Surface(**drifted, ground_reach_m=REACH_1M)
        ground = _change_planes(a, b, 0.0, 0.0, layer="ground")
        assert ground.counts()["lowered"] == expected, grade_cm
        assert ground.drift_reach_m == pytest.approx(REACH_1M)


def test_ground_lowered_over_an_area_is_ground_change() -> None:
    """Earth moved: the ground a metre lower over 8 x 8 m (wider than the window, so the
    envelope follows it), its top with it."""
    a = Surface(**flat(), ground_reach_m=REACH_1M)
    fields = flat()
    for name in ("dsm", "ground"):
        fields[name] = fields[name].copy()
        fields[name][6:14, 6:14] -= 100
    b = Surface(**fields, ground_reach_m=REACH_1M)
    ground = _change_planes(a, b, 0.0, 0.0, layer="ground")
    lowered = classes_at(ground, "lowered")
    # Its rim reads the drift of the step itself (a grade of 1 m over the window's span).
    assert {(i, j) for i in range(8, 12) for j in range(8, 12)} <= lowered
    assert ground.counts()["raised"] == 0


def test_the_grounds_reach_is_its_windows_half_diagonal() -> None:
    assert change.ground_reach_m({"estimators": {"groundWindowCells": 21}}, 0) == pytest.approx(
        10 * 0.25 * math.sqrt(2.0)
    )
    assert change.ground_reach_m({"estimators": {"groundWindowCells": 1}}, 4) == 0.0
    # Not stated: P5's own 5 m window at that level (centres within 2.5 m).
    assert change.ground_reach_m({}, 2) == pytest.approx(2 * 1.0 * math.sqrt(2.0))
    assert change.ground_reach_m({}, 4) == 0.0


# --- the no-change control, on P5's own bytes (review B5a) -------------------------------


def control_surface(name: str) -> Surface:
    scan = DOCUMENT["control"]["scans"][name]
    data = (SURFACES / scan["file"]).read_bytes()
    grid = cell_grid.decode(scan["descriptor"], data)
    reach = change.ground_reach_m(scan["descriptor"], grid.finest.level)
    return Surface.of(grid.finest, ground_reach_m=reach)


@pytest.mark.parametrize("layer", change.LAYERS)
@pytest.mark.parametrize(("pair", "ub"), [(("a", "b"), 0.0), (("c", "d"), 0.03)])
def test_no_change_control_nothing_changes_whatever_the_samples_and_noise(
    layer: change.Layer, pair: tuple[str, str], ub: float
) -> None:
    """Review B5a's ruling: one unchanged scene (flat ground, a 1.5 m box), scanned by P5's
    own estimator with 20 against 2,000 samples a cell (10 cm of noise each), and 1 cm
    against 10 cm of noise (100 samples each). Neither layer may read a change between
    them: the bias of P5's 2nd highest and of its lower envelope is in its sigmas. The
    3 sigma rule allows about one cell in 370 of Gaussian noise: 2 of these 1,024 at most."""
    a, b = (control_surface(name) for name in pair)
    planes = _change_planes(a, b, 0.0, ub, layer=layer)
    counts = planes.counts()
    assert counts["raised"] + counts["lowered"] <= 2, counts
    assert counts["unchanged"] > 900, counts
