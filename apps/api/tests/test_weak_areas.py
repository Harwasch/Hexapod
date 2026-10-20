"""Weak cells become smoothed patches on the globe, each with a way to fly it (A4).

Synthetic planes with known answers: where the patches are, how large, what a budget
keeps, and which pattern each reason asks for. The projection is checked against an
independent east/north/up -> ECEF -> WGS84 computation, not against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Polygon, box

from app.analysis import weak_areas
from app.analysis.cell_grid import REASON_BITS, Plane, weak_mask
from app.analysis.weak_areas import cell_patches, pattern_for, site_projection

LAT, LON, HEIGHT = 40.2, -111.6, 1500.0
BITS = REASON_BITS


def a_plane(level: int = 2, i0: int = -10, j0: int = -10, ni: int = 20, nj: int = 20) -> Plane:
    """Strong everywhere (score 200, no reasons, flat)."""
    return Plane(
        level,
        i0,
        j0,
        ni,
        nj,
        {
            "score": np.full((nj, ni), 200, dtype=np.uint8),
            "reasons": np.zeros((nj, ni), dtype=np.uint8),
            "relief": np.zeros((nj, ni), dtype=np.uint8),
        },
    )


def weaken(
    plane: Plane, i: tuple[int, int], j: tuple[int, int], *, score: int, bits: int, relief: int = 0
) -> None:
    """Cells `i[0] <= i < i[1]`, `j[0] <= j < j[1]` (lattice indices) made weak."""
    rows = slice(j[0] - plane.j0, j[1] - plane.j0)
    columns = slice(i[0] - plane.i0, i[1] - plane.i0)
    plane.channels["score"][rows, columns] = score
    plane.channels["reasons"][rows, columns] = bits
    plane.channels["relief"][rows, columns] = relief


def enu_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Independently of the module: rotate (x, y, 0) out of the frame's ENU into ECEF
    about the origin, then ECEF to WGS84."""
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    ox, oy, oz = to_ecef.transform(LON, LAT, HEIGHT)
    phi, lam = math.radians(LAT), math.radians(LON)
    east = np.array([-math.sin(lam), math.cos(lam), 0.0])
    north = np.array(
        [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)]
    )
    point = np.array([ox, oy, oz]) + x * east + y * north
    lon, lat, _ = to_geo.transform(float(point[0]), float(point[1]), float(point[2]))
    return float(lon), float(lat)


PROJECTION = site_projection(LAT, LON, HEIGHT)


def test_the_projection_is_the_frame_s_own_tangent_plane() -> None:
    for x, y in ((0.0, 0.0), (1000.0, 0.0), (-300.0, 700.0), (1500.0, -1500.0)):
        lon, lat = PROJECTION.to_geographic(x, y)
        want = enu_to_lonlat(x, y)
        assert float(lon) == pytest.approx(want[0], abs=1e-9)
        assert float(lat) == pytest.approx(want[1], abs=1e-9)
        back = PROJECTION.to_metric(lon, lat)
        assert math.hypot(float(back[0]) - x, float(back[1]) - y) < 1e-3


def test_an_azimuthal_projection_would_put_the_patch_off_its_cells() -> None:
    """Why the module does not use an azimuthal-equidistant projection at the origin: at a
    1,500 m origin it measures on the ellipsoid, and 1 km east lands 23 cm short."""
    aeqd = Transformer.from_crs(
        "EPSG:4326", f"+proj=aeqd +lat_0={LAT} +lon_0={LON} +datum=WGS84 +units=m", always_xy=True
    )
    x, _ = aeqd.transform(*enu_to_lonlat(1000.0, 0.0))
    assert 1000.0 - x > 0.2


def test_two_weak_blocks_are_two_patches_over_their_cells() -> None:
    plane = a_plane()
    # A 4 m x 6 m block that stands 4 m tall, seen from too few directions ...
    weaken(plane, (0, 4), (0, 6), score=50, bits=BITS["views"] | BITS["spread"], relief=16)
    # ... and a 10 m x 2 m strip seen too coarsely.
    weaken(plane, (-10, 0), (-8, -6), score=90, bits=BITS["gsd"])
    mask = weak_mask(plane)

    found = cell_patches(plane, mask, PROJECTION, median_gsd_mm=12.0)

    assert not found.truncated_by_budget and found.weak_cells == 44
    big, small = found.patches
    assert big.metric.symmetric_difference(box(0, 0, 4, 6)).area < 1e-9
    assert big.area_m2 == pytest.approx(24.0) and big.cells == 24
    assert small.metric.symmetric_difference(box(-10, -8, 0, -6)).area < 1e-9
    assert small.area_m2 == pytest.approx(20.0) and small.cells == 20
    # The tall compact block: an orbit, sideways camera.
    assert big.pattern == "orbit" and big.gimbal_pitch_deg == -30.0
    assert big.relief_max_m == pytest.approx(4.0) and big.diameter_m == pytest.approx(6.0)
    assert big.reasons["views"] == 24 and big.reasons["gsd"] == 0
    assert big.mean_score == pytest.approx(50.0)
    assert "rule of thumb" in big.hint
    # The strip: nadir, lower, for the capture's median GSD.
    assert small.pattern == "nadir" and small.target_gsd_m == pytest.approx(0.012)
    assert "1.2 cm" in small.hint
    # On the globe, each corner is where the independent computation puts it.
    lon, lat = big.footprint.exterior.coords[0]
    corners = [enu_to_lonlat(x, y) for x, y in box(0, 0, 4, 6).exterior.coords]
    assert min(math.dist((lon, lat), c) for c in corners) < 1e-9
    assert big.footprint.exterior.is_ccw


def test_a_flat_or_wide_patch_is_flown_oblique() -> None:
    plane = a_plane()
    weaken(plane, (0, 4), (0, 6), score=50, bits=BITS["spread"], relief=4)
    [patch] = cell_patches(plane, weak_mask(plane), PROJECTION).patches
    assert patch.pattern == "oblique" and patch.gimbal_pitch_deg == -45.0
    assert patch.target_gsd_m is None


def test_pattern_rules() -> None:
    """UX heuristics, stated once: the reason group most cells carry decides."""
    assert pattern_for({"views": 10}, 5.0, 30.0, None)[0] == "orbit"
    assert pattern_for({"views": 10}, 5.0, 41.0, None)[0] == "oblique"
    assert pattern_for({"views": 10}, 2.9, 10.0, None)[0] == "oblique"
    assert pattern_for({"gsd": 10, "views": 3}, 5.0, 10.0, 20.0)[:2] == ("nadir", 0.02)
    assert pattern_for({"gsd": 10}, 0.0, 10.0, None)[:2] == ("nadir", None)
    assert pattern_for({"unseen": 10}, 0.0, 10.0, 20.0)[:2] == ("nadir", None)
    pattern, target, hint = pattern_for({"alignment": 10, "views": 10}, 9.0, 10.0, 20.0)
    assert (pattern, target) == ("nadir", None) and "reference area" in hint
    assert pattern_for({}, 0.0, 0.0, None)[0] == "nadir"


def test_an_unaligned_epoch_is_one_weak_area_to_overlap_the_reference() -> None:
    """Every scored cell capped with the alignment bit: weak everywhere, flown nadir."""
    plane = a_plane()
    plane.channels["score"][:] = 14
    plane.channels["reasons"][:] = BITS["alignment"] | BITS["views"]
    [patch] = cell_patches(plane, weak_mask(plane), PROJECTION).patches
    assert patch.cells == 400 and patch.area_m2 == pytest.approx(400.0)
    assert patch.pattern == "nadir" and "reference area" in patch.hint
    assert patch.reasons["alignment"] == 400


def test_a_zone_too_large_for_the_budget_is_clipped_about_its_weakest_cells_or_the_focus() -> None:
    plane = a_plane()
    # One 10 x 10 m weak block, weaker in its east half.
    weaken(plane, (-5, 0), (-5, 5), score=100, bits=BITS["views"])
    weaken(plane, (0, 5), (-5, 5), score=30, bits=BITS["views"])
    mask = weak_mask(plane)

    found = cell_patches(plane, mask, PROJECTION, budget_m2=30.0)
    assert found.truncated_by_budget
    total = sum(p.area_m2 for p in found.patches)
    assert total <= 30.0 + 1.0
    [patch] = found.patches
    assert patch.metric.centroid.x > 0  # about the weaker east half

    lon, lat = PROJECTION.to_geographic(4.5, 4.5)
    near = cell_patches(plane, mask, PROJECTION, budget_m2=10.0, focus=(float(lon), float(lat)))
    [patch] = near.patches
    assert patch.metric.centroid.x > 2.5 and patch.metric.centroid.y > 2.5

    everything = cell_patches(plane, mask, PROJECTION, budget_m2=1e6)
    assert not everything.truncated_by_budget


def test_max_areas_keeps_the_largest_and_small_specks_are_not_zones() -> None:
    plane = a_plane()
    weaken(plane, (-10, -6), (-10, -6), score=50, bits=BITS["views"])  # 16 m2
    weaken(plane, (0, 3), (0, 3), score=50, bits=BITS["views"])  # 9 m2
    weaken(plane, (6, 8), (6, 8), score=50, bits=BITS["views"])  # 4 m2
    weaken(plane, (-8, -7), (8, 9), score=50, bits=BITS["views"])  # a single cell: dropped
    found = cell_patches(plane, weak_mask(plane), PROJECTION, max_areas=2)
    assert [round(p.area_m2) for p in found.patches] == [16, 9]
    every = cell_patches(plane, weak_mask(plane), PROJECTION, max_areas=8)
    assert [round(p.area_m2) for p in every.patches] == [16, 9, 4]


def test_nothing_weak_is_no_patch() -> None:
    plane = a_plane()
    found = cell_patches(plane, weak_mask(plane), PROJECTION)
    assert found.patches == [] and found.weak_cells == 0


def test_quarter_metre_cells_are_closed_into_zones_a_drone_can_fly() -> None:
    """At the finest level a checkerboard of weak cells is one zone, not hundreds of
    specks: patches close by at least `MIN_SMOOTHING_M` (a UX heuristic)."""
    plane = a_plane(level=0, i0=0, j0=0, ni=40, nj=40)
    for j in range(8, 32):
        for i in range(8, 32):
            if (i + j) % 2 == 0:
                weaken(plane, (i, i + 1), (j, j + 1), score=40, bits=BITS["views"])
    [patch] = cell_patches(plane, weak_mask(plane), PROJECTION).patches
    assert patch.area_m2 == pytest.approx(36.0, abs=2.0)
    assert weak_areas.MIN_SMOOTHING_M == 0.5


def test_a_large_plane_is_drawn_a_level_up_by_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past `MAX_ZONE_CELLS` zones are drawn on coarser cells, weak where at least half of
    the children the epoch speaks for are: a block of weak cells stays one zone the same
    size, a lone weak cell among strong ones does not make its parent weak, and the counts
    are still of the finest cells."""
    monkeypatch.setattr(weak_areas, "MAX_ZONE_CELLS", 100)
    plane = a_plane(level=0, i0=0, j0=0, ni=40, nj=40)
    weaken(plane, (8, 24), (8, 16), score=50, bits=BITS["views"])  # 4 m x 2 m
    weaken(plane, (33, 34), (33, 34), score=50, bits=BITS["views"])  # one stray cell
    found = cell_patches(plane, weak_mask(plane), PROJECTION)
    assert found.zone_level == 2  # 40 x 40 -> 20 x 20 -> 10 x 10 cells
    [patch] = found.patches
    assert patch.metric.symmetric_difference(box(2, 2, 6, 4)).area < 1e-9
    assert patch.cells == 128 and found.weak_cells == 129


def test_a_noisy_plane_at_the_ceiling_is_answered_in_seconds() -> None:
    """2^22 cells of scattered weakness: the zones are drawn a level up, in well under the
    minutes outlining every region took."""
    import time

    rng = np.random.default_rng(1)
    n = 2048
    plane = a_plane(level=0, i0=-n // 2, j0=-n // 2, ni=n, nj=n)
    yy, xx = np.mgrid[0:n, 0:n]
    noisy = np.sin(xx / 37.0) * np.cos(yy / 53.0) + 0.3 * rng.standard_normal((n, n)) > 0.6
    plane.channels["score"][noisy] = 60
    plane.channels["reasons"][noisy] = BITS["views"]
    started = time.perf_counter()
    found = cell_patches(plane, weak_mask(plane), PROJECTION, max_areas=8, budget_m2=2000)
    assert time.perf_counter() - started < 20
    assert found.truncated_by_budget
    assert found.patches and sum(p.area_m2 for p in found.patches) <= 2000 + 1.0


# --- budgets over scattered weakness, and planes past the ceiling (A4 review) -------------


@pytest.mark.parametrize("share", [0.2, 0.4])
@pytest.mark.parametrize("budget", [10.0, 100.0, 1000.0])
def test_scattered_weakness_within_a_budget_still_gives_a_zone(share: float, budget: float) -> None:
    """Closing makes a zone of scattered weak cells larger than its cells; a budget used to
    trim cells until nothing was left (200 x 200, 40 % weak, 1,000 m2: no zone at all). A
    budget is met by choosing and clipping zones, and some zone always comes back."""
    rng = np.random.default_rng(0)
    plane = a_plane(level=0, i0=0, j0=0, ni=200, nj=200)
    weak = rng.random((200, 200)) < share
    plane.channels["score"][weak] = 50
    plane.channels["reasons"][weak] = BITS["views"]
    found = cell_patches(plane, weak_mask(plane), PROJECTION, budget_m2=budget)
    assert found.patches, (share, budget)
    assert found.truncated_by_budget
    assert sum(p.area_m2 for p in found.patches) <= budget + 1.0
    assert all(p.cells > 0 for p in found.patches)


def test_a_budget_keeps_the_largest_zone_that_fits_not_a_speck() -> None:
    """The evidence run kept a 1.75 m2 unseen speck and dropped a 208 m2 zone: cells were
    ranked weakest first. Zones are ranked by size (then score, then focus) and added while
    they fit."""
    plane = a_plane()
    weaken(plane, (-8, 2), (-8, 2), score=90, bits=BITS["gsd"])  # 100 m2
    weaken(plane, (6, 8), (6, 8), score=0, bits=BITS["unseen"])  # 4 m2, unseen
    mask = weak_mask(plane)
    assert [round(p.area_m2) for p in cell_patches(plane, mask, PROJECTION).patches] == [100, 4]
    kept = cell_patches(plane, mask, PROJECTION, budget_m2=100.0)
    assert [round(p.area_m2) for p in kept.patches] == [100] and kept.truncated_by_budget
    both = cell_patches(plane, mask, PROJECTION, budget_m2=104.0)
    assert [round(p.area_m2) for p in both.patches] == [100, 4]
    # Too small for the block: the block is clipped to the budget, not passed over for
    # the speck behind it (the evidence run's 208 m2 zone at 100 m2).
    for budget in (10.0, 50.0):
        [clip] = cell_patches(plane, mask, PROJECTION, budget_m2=budget).patches
        assert clip.metric.within(box(-8, -8, 2, 2).buffer(1e-9))
        assert budget / 2 < clip.area_m2 <= budget + 1.0 and clip.reasons["gsd"] == clip.cells


def test_a_budget_smaller_than_every_zone_clips_the_best_one() -> None:
    plane = a_plane()
    weaken(plane, (-8, 2), (-8, 2), score=90, bits=BITS["gsd"])  # 100 m2, the only zone
    found = cell_patches(plane, weak_mask(plane), PROJECTION, budget_m2=30.0)
    [patch] = found.patches
    assert found.truncated_by_budget and patch.area_m2 <= 30.0 + 1.0
    assert patch.metric.within(box(-8, -8, 2, 2).buffer(1e-9)) and patch.cells > 0


def test_a_coarse_cell_is_weak_by_majority_not_by_any_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drawn a level up, a quarter of the children weak is not a weak coarse cell (with
    'any child' it would be a 16 m2 zone)."""
    monkeypatch.setattr(weak_areas, "MAX_ZONE_CELLS", 100)
    plane = a_plane(level=0, i0=0, j0=0, ni=40, nj=40)
    for j in range(8, 24, 2):
        for i in range(8, 24, 2):
            weaken(plane, (i, i + 1), (j, j + 1), score=50, bits=BITS["views"])
    weaken(plane, (32, 40), (32, 40), score=50, bits=BITS["views"])  # a real 2 m x 2 m block
    found = cell_patches(plane, weak_mask(plane), PROJECTION)
    assert found.zone_level > 0
    [patch] = found.patches
    assert patch.metric.symmetric_difference(box(8, 8, 10, 10)).area < 1e-9


def test_a_dense_noisy_plane_past_the_ceiling_answers_in_seconds() -> None:
    """2048 x 2048 cells, half of them weak at random: one region full of holes, which took
    157 s to outline. Zones are coarsened until their outline is bounded and holes under
    the smallest zone are filled. The bound -- 10 s -- is the budget an interactive request
    can spend (first principles: a person pressed a button), not a measured timing."""
    import time

    rng = np.random.default_rng(0)
    n = 2048
    plane = a_plane(level=2, i0=-n // 2, j0=-n // 2, ni=n, nj=n)
    weak = rng.random((n, n)) < 0.5
    plane.channels["score"][weak] = 100
    plane.channels["reasons"][weak] = BITS["views"]
    started = time.perf_counter()
    found = cell_patches(plane, weak_mask(plane), PROJECTION, max_areas=3)
    assert time.perf_counter() - started < 10.0
    assert found.patches
    smallest = max(weak_areas.MIN_AREA_M2, 2 * math.ldexp(0.25, found.zone_level) ** 2)
    for patch in found.patches:
        assert all(Polygon(ring).area >= smallest for ring in patch.metric.interiors)


def test_a_clip_is_about_the_zone_s_weakest_cells_not_its_middle() -> None:
    plane = a_plane()
    weaken(plane, (-5, 5), (-5, 5), score=100, bits=BITS["views"])  # 100 m2 about the origin
    weaken(plane, (3, 5), (3, 5), score=0, bits=BITS["unseen"])  # its weakest, a corner
    [patch] = cell_patches(plane, weak_mask(plane), PROJECTION, budget_m2=10.0).patches
    assert patch.metric.centroid.x > 2 and patch.metric.centroid.y > 2


def test_with_a_focus_outside_every_zone_the_largest_comes_first() -> None:
    plane = a_plane()
    weaken(plane, (-10, -2), (-10, -2), score=60, bits=BITS["views"])  # 64 m2, far
    weaken(plane, (6, 9), (6, 9), score=60, bits=BITS["views"])  # 9 m2, near the focus
    lon, lat = PROJECTION.to_geographic(9.5, 9.5)
    focus = (float(lon), float(lat))
    found = cell_patches(plane, weak_mask(plane), PROJECTION, focus=focus, max_areas=1)
    assert [round(p.area_m2) for p in found.patches] == [64]
    # Under the focus, the zone it is in comes first, however small.
    lon, lat = PROJECTION.to_geographic(7.5, 7.5)
    inside = cell_patches(
        plane, weak_mask(plane), PROJECTION, focus=(float(lon), float(lat)), max_areas=1
    )
    assert [round(p.area_m2) for p in inside.patches] == [9]


def test_a_clip_at_a_zone_s_edge_still_spends_most_of_the_budget() -> None:
    """The weakest cell at the zone's edge puts most of a square about it outside the zone
    (the evidence run: 36 m2 of a 100 m2 budget); the square about the zone's middle is
    tried too, and the larger piece kept."""
    plane = a_plane()
    weaken(plane, (-10, 10), (-10, 10), score=100, bits=BITS["views"])  # 400 m2
    weaken(plane, (-10, -9), (-10, -9), score=0, bits=BITS["unseen"])  # weakest, a corner
    [patch] = cell_patches(plane, weak_mask(plane), PROJECTION, budget_m2=100.0).patches
    assert 50.0 <= patch.area_m2 <= 101.0
