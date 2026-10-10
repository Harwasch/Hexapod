from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from app.analysis.solar import (
    ENDPOINT,
    PARAMETERS,
    SolarResult,
    calculate,
    diffuse_factors,
    horizon_at,
    location,
    retrieve,
)
from app.schemas.geojson import Footprint
from app.schemas.land_solar import SolarRequest

BOUNDARY: Footprint = TypeAdapter(Footprint).validate_python(
    {
        "type": "Polygon",
        "coordinates": [
            [
                [-77.055, 38.886],
                [-77.045, 38.886],
                [-77.045, 38.891],
                [-77.055, 38.891],
                [-77.055, 38.886],
            ]
        ],
    }
)
REQUEST: dict[str, Any] = {
    "arrayZone": BOUNDARY.model_dump(),
    "zoneBasis": "Public National Mall software test fixture",
    "year": 2023,
    "moduleAreaM2": 100,
    "moduleEfficiency": 0.2,
    "tiltDegrees": 0,
    "azimuthDegrees": 180,
    "mounting": "open_rack_glass_glass",
    "temperatureCoefficient": 0,
    "dcAcRatio": 1,
    "inverterEfficiency": 0.96,
    "systemLoss": 0,
    "additionalShadeLoss": 0,
    "albedo": 0,
    "iamB": 0,
    "horizon": [],
    "horizonBasis": "Synthetic unobstructed fixture",
    "assumptions": "Test inputs only",
}


def weather(request: SolarRequest) -> dict[str, Any]:
    lon, lat, _ = location(BOUNDARY, request)
    start, end = (
        datetime(request.year, 1, 1, tzinfo=UTC),
        datetime(request.year + 1, 1, 1, tzinfo=UTC),
    )
    parameters: dict[str, dict[str, float]] = {name: {} for name in PARAMETERS}
    while start < end:
        key = start.strftime("%Y%m%d%H")
        # One diffuse-only hour per day. This supports an independent energy balance.
        values = [500 if start.hour == 17 else 0, 0, 500 if start.hour == 17 else 0, 25, 1]
        for name, value in zip(PARAMETERS, values, strict=True):
            parameters[name][key] = value
        start += timedelta(hours=1)
    return {
        "header": {
            "time_standard": "UTC",
            "start": f"{request.year}0101",
            "end": f"{request.year}1231",
            "fill_value": -999,
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat, 70]},
        "parameters": {name: {"units": unit} for name, unit in PARAMETERS.items()},
        "properties": {"parameter": parameters},
    }


def result(request: SolarRequest, source: dict[str, Any] | None = None) -> SolarResult:
    return calculate(BOUNDARY, request, json.dumps(source or weather(request)).encode(), ENDPOINT)


def test_energy_units_hourly_archive_and_leap_year() -> None:
    request = SolarRequest.model_validate({**REQUEST, "year": 2024})
    raw = json.dumps(weather(request)).encode()
    output = calculate(BOUNDARY, request, raw, ENDPOINT)
    m = output.metadata
    assert m.expected_hours == m.valid_hours == 8784 and m.complete_year
    assert m.plane_irradiation_kwh_m2 == pytest.approx(366 * 0.5)
    # Independent PVWatts inverter equation: rated AC=20kW, pdc0=20/.96kW, input=10kW.
    zeta = 10 / (20 / 0.96)
    expected_ac = 10 * 0.96 / 0.9637 * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
    assert m.annual_generation_kwh == pytest.approx(expected_ac * 366)
    assert m.annual_generation_kwh == pytest.approx(sum(v.generation_kwh or 0 for v in m.monthly))
    assert m.monthly[1].expected_hours == 29 * 24
    with zipfile.ZipFile(io.BytesIO(output.data)) as bundle:
        assert set(bundle.namelist()) == {
            "source.json",
            "hourly.csv",
            "result.json",
            "request.json",
        }
        assert hashlib.sha256(bundle.read("source.json")).hexdigest() == m.source_sha256
        assert bundle.read("source.json") == raw
        rows = list(csv.DictReader(io.StringIO(bundle.read("hourly.csv").decode())))
        assert len(rows) == 8784 and rows[0]["hour_start_utc"] == "2024-01-01T00:00:00+00:00"
        assert sum(float(r["ac_w"]) for r in rows) / 1000 == pytest.approx(m.annual_generation_kwh)


def test_missing_weather_is_not_zero_or_annualized() -> None:
    request = SolarRequest.model_validate(REQUEST)
    source = weather(request)
    source["properties"]["parameter"]["T2M"].pop("2023010117")
    source["properties"]["parameter"]["WS10M"]["2023010217"] = -999
    m = result(request, source).metadata
    assert m.valid_hours == 8758 and not m.complete_year and m.annual_generation_kwh is None
    assert m.invalid_parameter_hours["T2M"] == m.invalid_parameter_hours["WS10M"] == 1
    assert m.plane_irradiation_kwh_m2 == pytest.approx(363 * 0.5)
    for parameter in source["properties"]["parameter"].values():
        parameter.clear()
    empty = result(request, source).metadata
    assert empty.valid_hours == 0 and empty.annual_generation_kwh is None
    assert all(month.generation_kwh is None for month in empty.monthly)


def test_horizon_quadrature_wraparound_shading_and_clipping() -> None:
    request = SolarRequest.model_validate(REQUEST)
    assert diffuse_factors(request) == pytest.approx((1, 1, 1, 1))
    shaded = SolarRequest.model_validate(
        {
            **REQUEST,
            "horizon": [{"azimuthDegrees": az, "elevationDegrees": 45} for az in [270, 0, 180, 90]],
        }
    )
    assert diffuse_factors(shaded)[0] == pytest.approx(0.5, abs=0.0001)
    assert horizon_at(np.array([-1.0, 0.0, 359.0, 360.0]), shaded).tolist() == [45] * 4
    assert result(shaded).metadata.plane_irradiation_kwh_m2 == pytest.approx(365 * 0.25)
    fully_shaded = SolarRequest.model_validate(
        {
            **REQUEST,
            "horizon": [{"azimuthDegrees": az, "elevationDegrees": 90} for az in [0, 90, 180, 270]],
        }
    )
    assert result(fully_shaded).metadata.annual_generation_kwh == 0
    clipped = SolarRequest.model_validate({**REQUEST, "dcAcRatio": 2})
    source = weather(clipped)
    for name in ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DIFF"]:
        source["properties"]["parameter"][name]["2023010117"] = 1000
    assert result(clipped, source).metadata.peak_ac_kw == pytest.approx(10)


def test_reject_wrong_units_period_location_geometry_and_horizon() -> None:
    request = SolarRequest.model_validate(REQUEST)
    source = weather(request)
    source["parameters"]["T2M"]["units"] = "F"
    with pytest.raises(ValueError, match="units"):
        result(request, source)
    source = weather(request)
    source["header"]["time_standard"] = "LST"
    with pytest.raises(ValueError, match="UTC"):
        result(request, source)
    with pytest.raises(ValueError, match="projected module area"):
        location(BOUNDARY, SolarRequest.model_validate({**REQUEST, "moduleAreaM2": 1e7}))
    with pytest.raises(ValidationError, match="full circle"):
        SolarRequest.model_validate(
            {
                **REQUEST,
                "horizon": [{"azimuthDegrees": az, "elevationDegrees": 10} for az in [0, 1, 2, 3]],
            }
        )
    with pytest.raises(ValidationError, match="completed"):
        SolarRequest.model_validate({**REQUEST, "year": datetime.now(UTC).year})


def test_retrieval_fixed_endpoint_units_and_size_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "power.larc.nasa.gov"
        assert request.url.params["time-standard"] == "UTC"
        assert request.url.params["start"] == "20230101"
        return httpx.Response(200, content=b" " * (8 * 1024**2 + 1))

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="8 MiB"),
    ):
        retrieve(client, -77.05, 38.889, 2023)


def test_orientation_incidence_and_temperature_change_physical_output() -> None:
    south = SolarRequest.model_validate(
        {**REQUEST, "tiltDegrees": 60, "temperatureCoefficient": -0.0035}
    )
    source = weather(south)
    for name in ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF"]:
        source["properties"]["parameter"][name] = dict.fromkeys(
            source["properties"]["parameter"][name], 0
        )
    source["properties"]["parameter"]["ALLSKY_SFC_SW_DNI"]["2023012017"] = 800
    source["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]["2023012017"] = 400
    south_result = result(south, source).metadata
    north = south.model_copy(update={"azimuth_degrees": 0})
    assert (
        south_result.plane_irradiation_kwh_m2
        > 3 * result(north, source).metadata.plane_irradiation_kwh_m2
    )
    warm = result(south, source).metadata.modeled_generation_kwh
    source["properties"]["parameter"]["T2M"]["2023012017"] = 5
    cool = result(south, source).metadata.modeled_generation_kwh
    assert cool > warm
    incidence_loss = south.model_copy(update={"iam_b": 0.1})
    assert result(incidence_loss, source).metadata.modeled_generation_kwh < cool
