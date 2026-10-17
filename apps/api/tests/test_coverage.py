from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pyproj import Geod
from shapely.geometry import shape

from app.services.coverage import geodesic_area_m2, measure
from app.services.machine_classes import MACHINE_CLASSES, load

# The fixtures packages/geo/src/coverage.test.ts holds the console's preview to as well.
CASES = json.loads(
    (Path(__file__).resolve().parents[3] / "packages/geo/fixtures/coverage_cases.json").read_text()
)["cases"]


def case(name: str) -> dict[str, Any]:
    return next(c for c in CASES if c["name"] == name)


@pytest.mark.parametrize("fixture", CASES, ids=[c["name"] for c in CASES])
def test_measure_matches_the_shared_fixtures(fixture: dict[str, Any]) -> None:
    expect = fixture["expect"]
    kwargs: dict[str, Any] = {}
    if "maxPasses" in fixture:
        kwargs["max_passes"] = fixture["maxPasses"]
    if "headingDeg" in fixture:
        kwargs["heading_deg"] = fixture["headingDeg"]
    result = measure(fixture["geometry"], fixture["spacingM"], **kwargs)
    assert result is not None and result.kind == "area"
    if "headingDeg" in expect:
        assert result.heading_deg == expect["headingDeg"]
    if "passes" in expect:
        assert result.passes == expect["passes"]
    if "spacingM" in expect:
        assert result.spacing_m == pytest.approx(expect["spacingM"], abs=0.05)
    if "capped" in expect:
        assert result.capped is expect["capped"]
    if "lengthMin" in expect:
        assert expect["lengthMin"] < result.length_m < expect["lengthMax"]
    if "lengthM" in expect:
        assert result.length_m == pytest.approx(expect["lengthM"], abs=expect["lengthTolM"])


def test_rectangle_reports_spans_transit_and_geodesic_area() -> None:
    rect = case("rectangle 200 x 100 m")["geometry"]
    result = measure(rect, 10)
    assert result is not None
    assert result.spans == 10 and result.transit_m == pytest.approx(90)
    assert result.area_m2 == pytest.approx(20_000, rel=0.001)
    assert not result.capped and result.spacing_m == 10
    # The geodesic area is the same one land actions compute.
    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(shape(rect))[0])
    assert result.area_m2 == pytest.approx(area, rel=0.001)


def test_a_hole_shortens_the_passes_that_cross_it() -> None:
    whole = measure(case("rectangle 200 x 100 m")["geometry"], 10)
    holed = measure(case("rectangle with a 50 x 50 m hole")["geometry"], 10)
    assert whole is not None and holed is not None
    # Five passes cross the 50 m hole and lose 50 m each; each becomes two spans.
    assert whole.length_m - holed.length_m == pytest.approx(250, abs=5)
    assert holed.spans == 15 and holed.passes == 10
    assert holed.area_m2 == pytest.approx(17_500, rel=0.002)


def test_a_multipolygon_sums_its_parts() -> None:
    one = measure(case("rectangle 200 x 100 m")["geometry"], 10)
    both = measure(case("two rectangles")["geometry"], 10)
    assert one is not None and both is not None
    assert both.length_m == pytest.approx(2 * one.length_m, rel=0.01)
    assert both.area_m2 == pytest.approx(2 * (one.area_m2 or 0), rel=0.001)
    assert both.spans == 20


def test_lines_points_and_degenerate_input() -> None:
    line = {"type": "LineString", "coordinates": [[-122.139, 47.644], [-122.135, 47.644]]}
    result = measure(line, 10)
    geodesic = Geod(ellps="WGS84").geometry_length(shape(line))
    assert result is not None and result.kind == "line"
    assert result.length_m == pytest.approx(geodesic, rel=0.005)
    assert result.passes == result.spans == 1 and result.area_m2 is None
    assert result.transit_m == 0 and result.spacing_m is None
    assert measure({"type": "Point", "coordinates": [-122.139, 47.644]}, 10) is None
    assert measure(case("rectangle 200 x 100 m")["geometry"], 0) is None


def test_geodesic_area_ignores_lines() -> None:
    assert geodesic_area_m2(shape({"type": "Point", "coordinates": [0, 0]})) == 0


def test_machine_classes_load_and_derive_the_drone_spacing(tmp_path: Path) -> None:
    drone = MACHINE_CLASSES.drone
    assert MACHINE_CLASSES.mower.speed_mps == 1.4 and MACHINE_CLASSES.mower.swath_m == 2.0
    assert MACHINE_CLASSES.seeder.turn_s == 12 and MACHINE_CLASSES.survey.swath_m == 10
    assert all("(assumed)" in c.label for c in (drone, MACHINE_CLASSES.mower))
    # The dress rehearsal's limits, assumed like the rest: a slope cap and a creek buffer.
    assert MACHINE_CLASSES.mower.max_slope_pct == 25 and MACHINE_CLASSES.mower.creek_buffer_m == 15
    assert MACHINE_CLASSES.survey.creek_buffer_m < MACHINE_CLASSES.seeder.creek_buffer_m
    # The wide camera of a Mavic 3 Enterprise class drone: 73.7 degrees across (24 mm
    # equivalent), so 89.9 m of ground across at 60 m and lines 27 m apart at 70 % side overlap.
    assert drone.footprint_width_m() == pytest.approx(89.9, abs=0.1)
    assert drone.line_spacing_m() == pytest.approx(27.0, abs=0.1)
    assert drone.line_spacing_m(120) == pytest.approx(2 * drone.line_spacing_m())
    broken = tmp_path / "classes.json"
    broken.write_text(json.dumps({"mower": {"label": "x"}}))
    with pytest.raises(ValueError):
        load(broken)
