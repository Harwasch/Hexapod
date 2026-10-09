"""Fixed, auditable hourly solar model. No generated code or arbitrary network locations."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pvlib
from numpy.typing import NDArray
from pydantic import HttpUrl
from shapely.geometry import shape

from app.schemas.geojson import Footprint
from app.schemas.land_solar import SolarMetadata, SolarMonth, SolarRequest
from app.services.land_surveys import area

ENDPOINT = "https://power.larc.nasa.gov/api/temporal/hourly/point"
DOCUMENTATION = "https://power.larc.nasa.gov/docs/services/api/temporal/hourly/"
PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": "Wh/m^2",
    "ALLSKY_SFC_SW_DNI": "Wh/m^2",
    "ALLSKY_SFC_SW_DIFF": "Wh/m^2",
    "T2M": "C",
    "WS10M": "m/s",
}
ATTRIBUTION = (
    "NASA POWER Project, NASA Langley Research Center; MERRA-2 and CERES/SYN1deg "
    "contributing data; pvlib python models."
)
LICENSE = (
    "NASA open data; acknowledge POWER and its contributing data sources. pvlib model "
    "implementation is BSD licensed."
)


@dataclass
class SolarResult:
    data: bytes
    metadata: SolarMetadata


def location(boundary: Footprint, request: SolarRequest) -> tuple[float, float, float]:
    zone = shape(request.array_zone.model_dump())
    if zone.is_empty or not zone.is_valid or zone.area <= 0:
        raise ValueError(
            "The array zone must be a valid nonempty polygon without self-intersections."
        )
    if not shape(boundary.model_dump()).covers(zone):
        raise ValueError(
            "The complete array zone must be inside the pinned land boundary, including its exclusions."
        )
    west, south, east, north = zone.bounds
    if east - west > 0.1 or north - south > 0.1 or max(abs(south), abs(north)) > 85:
        raise ValueError("Use a local array zone within 0.1 degree and below 85 degrees latitude.")
    mapped_area = area(request.array_zone)
    if request.module_area_m2 * math.cos(math.radians(request.tilt_degrees)) > mapped_area * 1.001:
        raise ValueError(
            "The projected module area exceeds the mapped array zone. Revise the module area or zone."
        )
    point = zone.representative_point()
    return point.x, point.y, mapped_area


def retrieve(
    client: httpx.Client, longitude: float, latitude: float, year: int
) -> tuple[bytes, str, dict[str, str]]:
    params: dict[str, str | float] = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
        "time-standard": "UTC",
    }
    started = time.monotonic()
    with client.stream(
        "GET", ENDPOINT, params=params, follow_redirects=False, timeout=45
    ) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_bytes(chunk_size=65536):
            if time.monotonic() - started > 80:
                raise ValueError("The hourly weather source exceeded its retrieval time limit.")
            data.extend(chunk)
            if len(data) > 8 * 1024**2:
                raise ValueError("The hourly weather source exceeded its 8 MiB limit.")
        return bytes(data), str(response.url), dict(response.headers)


def horizon_at(azimuth: NDArray[np.float64], request: SolarRequest) -> NDArray[np.float64]:
    if not request.horizon:
        return np.zeros_like(azimuth)
    return np.asarray(
        np.interp(
            azimuth,
            [p.azimuth_degrees for p in request.horizon],
            [p.elevation_degrees for p in request.horizon],
            period=360,
        ),
        dtype=np.float64,
    )


def diffuse_factors(request: SolarRequest) -> tuple[float, float, float, float]:
    # One-degree midpoint solid-angle quadrature on an isotropic hemisphere. Normalize
    # by the numerical open-sky integral, so an empty horizon is exactly unchanged.
    azimuth = np.arange(0.5, 360, 1.0)[None, :]
    altitude = np.arange(0.5, 90, 1.0)[:, None]
    tilt = math.radians(request.tilt_degrees)
    relative = np.deg2rad(azimuth - request.azimuth_degrees)
    a = np.deg2rad(altitude)
    projection = np.maximum(
        0, np.cos(a) * math.sin(tilt) * np.cos(relative) + np.sin(a) * math.cos(tilt)
    )
    weights = projection * np.cos(a)
    visible = altitude > horizon_at(azimuth, request)
    incidence = np.rad2deg(np.arccos(np.clip(projection, 0, 1)))
    iam = np.asarray(pvlib.iam.ashrae(incidence, b=request.iam_b), dtype=np.float64)
    denominator = float(weights.sum())
    ground_projection = np.maximum(
        0, np.cos(a) * math.sin(tilt) * np.cos(relative) - np.sin(a) * math.cos(tilt)
    )
    ground_weights = ground_projection * np.cos(a)
    ground_iam = pvlib.iam.ashrae(
        np.rad2deg(np.arccos(np.clip(ground_projection, 0, 1))), b=request.iam_b
    )
    ground_factor = (
        float((ground_weights * ground_iam).sum() / ground_weights.sum())
        if ground_weights.sum()
        else 1.0
    )
    return (
        float((weights * visible).sum() / denominator),
        float((weights * visible * iam).sum() / denominator),
        float((weights * iam).sum() / denominator),
        ground_factor,
    )


def calculate(
    boundary: Footprint,
    request: SolarRequest,
    raw: bytes,
    source_url: str,
    headers: dict[str, str] | None = None,
) -> SolarResult:
    longitude, latitude, mapped_area = location(boundary, request)
    source = json.loads(raw)
    header = source.get("header", {})
    if (
        header.get("time_standard") != "UTC"
        or str(header.get("start")) != f"{request.year}0101"
        or str(header.get("end")) != f"{request.year}1231"
    ):
        raise ValueError("The weather source did not return the requested full UTC calendar year.")
    source_geometry = source.get("geometry", {}).get("coordinates", [])
    if len(source_geometry) < 3 or not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in source_geometry[:3]
    ):
        raise ValueError("The weather source omitted its sample location and grid elevation.")
    if abs(source_geometry[0] - longitude) > 0.001 or abs(source_geometry[1] - latitude) > 0.001:
        raise ValueError("The weather source returned a different requested location.")
    elevation = float(source_geometry[2])
    if not -500 <= elevation <= 9000:
        raise ValueError("The weather source returned an unsupported grid elevation.")
    starts = pd.date_range(
        f"{request.year}-01-01", f"{request.year + 1}-01-01", inclusive="left", freq="h", tz="UTC"
    )
    times = starts + pd.Timedelta(minutes=30)
    keys = list(starts.strftime("%Y%m%d%H"))
    expected = set(keys)
    arrays, invalid = {}, {}
    limits = {
        "ALLSKY_SFC_SW_DWN": (0, 2000),
        "ALLSKY_SFC_SW_DNI": (0, 2000),
        "ALLSKY_SFC_SW_DIFF": (0, 2000),
        "T2M": (-100, 80),
        "WS10M": (0, 100),
    }
    for parameter, unit in PARAMETERS.items():
        if source.get("parameters", {}).get(parameter, {}).get("units") != unit:
            raise ValueError(f"The weather source returned unexpected units for {parameter}.")
        values = source.get("properties", {}).get("parameter", {}).get(parameter, {})
        if not isinstance(values, dict) or set(values) - expected:
            raise ValueError(
                "The weather source returned timestamps outside the requested hourly period."
            )
        values_list = [values.get(key) for key in keys]
        data = np.array(
            [
                float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan
                for v in values_list
            ],
            dtype=np.float64,
        )
        low, high = limits[parameter]
        good = (
            np.isfinite(data)
            & (data >= low)
            & (data <= high)
            & (data != header.get("fill_value", -999))
        )
        invalid[parameter] = int((~good).sum())
        arrays[parameter] = np.where(good, data, np.nan)
    valid = np.logical_and.reduce([np.isfinite(v) for v in arrays.values()])
    ghi, dni, dhi, temperature, wind = (arrays[k] for k in PARAMETERS)
    # POWER reports hourly Wh/m² at the start of each one-hour interval. Dividing
    # by one hour gives its average W/m²; solar geometry is approximated at midpoint.
    solar = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude=elevation, pressure=0
    )
    zenith = np.asarray(solar["zenith"], dtype=np.float64)
    sun_azimuth = np.asarray(solar["azimuth"], dtype=np.float64)
    sun_elevation = 90 - zenith
    daylight = sun_elevation > 0
    beam_visible = daylight & (sun_elevation > horizon_at(sun_azimuth, request))
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    sky = pvlib.irradiance.haydavies(
        request.tilt_degrees,
        request.azimuth_degrees,
        dhi,
        dni,
        dni_extra,
        solar_zenith=zenith,
        solar_azimuth=sun_azimuth,
        return_components=True,
    )
    isotropic = np.asarray(sky["poa_isotropic"], dtype=np.float64)
    circumsolar = np.where(daylight, np.asarray(sky["poa_circumsolar"], dtype=np.float64), 0)
    direct = np.where(
        daylight,
        pvlib.irradiance.beam_component(
            request.tilt_degrees, request.azimuth_degrees, zenith, sun_azimuth, dni
        ),
        0,
    )
    ground = np.asarray(
        pvlib.irradiance.get_ground_diffuse(request.tilt_degrees, ghi, request.albedo),
        dtype=np.float64,
    )
    incidence = pvlib.irradiance.aoi(
        request.tilt_degrees, request.azimuth_degrees, zenith, sun_azimuth
    )
    beam_iam = pvlib.iam.ashrae(incidence, b=request.iam_b)
    sky_fraction, sky_effective, sky_open_effective, ground_iam = diffuse_factors(request)
    unshaded_poa = direct + circumsolar + isotropic + ground
    shaded_poa = ((direct + circumsolar) * beam_visible + isotropic * sky_fraction + ground) * (
        1 - request.additional_shade_loss
    )
    effective_open = (
        (direct + circumsolar) * beam_iam + isotropic * sky_open_effective + ground * ground_iam
    )
    effective_shaded = (
        (direct + circumsolar) * beam_visible * beam_iam
        + isotropic * sky_effective
        + ground * ground_iam
    ) * (1 - request.additional_shade_loss)
    capacity = request.module_area_m2 * request.module_efficiency * 1000
    ac_rating = capacity / request.dc_ac_ratio
    thermal = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][request.mounting]

    def generation(
        poa: NDArray[Any], effective: NDArray[Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        cell = np.asarray(
            pvlib.temperature.sapm_cell(poa, temperature, wind, **thermal), dtype=np.float64
        )
        dc = np.maximum(
            0, pvlib.pvsystem.pvwatts_dc(effective, cell, capacity, request.temperature_coefficient)
        ) * (1 - request.system_loss)
        ac = pvlib.inverter.pvwatts(
            dc, ac_rating / request.inverter_efficiency, eta_inv_nom=request.inverter_efficiency
        )
        return cell, np.asarray(dc, dtype=np.float64), np.asarray(ac, dtype=np.float64)

    cell_temperature, dc, ac = generation(shaded_poa, effective_shaded)
    _, _, unshaded_ac = generation(unshaded_poa, effective_open)
    for values in (shaded_poa, unshaded_poa, cell_temperature, dc, ac, unshaded_ac):
        if not np.isfinite(values[valid]).all():
            raise ValueError("The solar model produced an invalid value for valid source hours.")
        values[~valid] = np.nan
    monthly = []
    for month in range(1, 13):
        selected = np.asarray(starts.month == month) & valid
        count = int(selected.sum())
        monthly.append(
            SolarMonth(
                month=month,
                expected_hours=int((starts.month == month).sum()),
                valid_hours=count,
                generation_kwh=float(ac[selected].sum()) / 1000 if count else None,
                unshaded_generation_kwh=float(unshaded_ac[selected].sum()) / 1000
                if count
                else None,
                plane_irradiation_kwh_m2=float(shaded_poa[selected].sum()) / 1000
                if count
                else None,
                unshaded_plane_irradiation_kwh_m2=float(unshaded_poa[selected].sum()) / 1000
                if count
                else None,
            )
        )
    energy = float(np.nansum(ac)) / 1000
    complete = bool(valid.all())
    limitations = [
        "Modeled output for one historical weather year, not a forecast, typical "
        "meteorological year or performance guarantee.",
        "NASA POWER solar and weather grids describe regional conditions, not measured "
        "roof conditions or local obstructions.",
        "Hourly energy inputs use solar position at the UTC interval midpoint. Sub-hour "
        "cloud, sunrise and shadow changes are unresolved.",
        "The array zone is a mapped plan area, not a surveyed roof plane. Module face "
        "area, tilt, mounting and equipment parameters are entered assumptions; panel "
        "layout, setbacks and structural suitability are not solved.",
        "A supplied horizon is linearly interpolated around the full azimuth circle. It "
        "masks direct/circumsolar light and the isotropic sky dome; it is not a "
        "three-dimensional building/tree or module-string electrical shading model.",
        "Diffuse incidence losses and horizon visibility use one-degree hemisphere "
        "quadrature. Ground reflection assumes isotropic constant albedo and no ground "
        "obstruction; no snow or bifacial model is included.",
        "PVWatts DC and inverter models with SAPM temperature and ASHRAE incidence "
        "losses; other DC losses and additional uniform irradiance shading are separate "
        "entered assumptions.",
        "Missing or invalid source hours are excluded, never replaced by zero or scaled "
        "to a full year. A full annual yield is available only when all source hours are "
        "valid.",
    ]
    if not request.horizon:
        limitations.append(
            "No local horizon was supplied. The geometric model assumes an open horizon; "
            "nearby building and tree shadows remain unmodeled."
        )
    if not complete:
        limitations.append(
            f"Only {int(valid.sum())} of {len(starts)} source hours are valid. "
            "Do not use the partial-period sum as annual generation."
        )
    metadata = SolarMetadata(
        algorithm="hourly-solar-pvwatts-haydavies-v1",
        model_version=f"pvlib {pvlib.__version__}; numpy {np.__version__}; pandas {pd.__version__}",
        source_url=HttpUrl(source_url),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        source_header=header,
        source_units=PARAMETERS,
        source_etag=(headers or {}).get("etag"),
        source_last_modified=(headers or {}).get("last-modified"),
        retrieved_at=datetime.now(UTC),
        attribution=ATTRIBUTION,
        license=LICENSE,
        documentation_url=HttpUrl(DOCUMENTATION),
        longitude=longitude,
        latitude=latitude,
        source_elevation_m=elevation,
        mapped_zone_area_m2=mapped_area,
        capacity_kw_dc=capacity / 1000,
        inverter_kw_ac=ac_rating / 1000,
        expected_hours=len(starts),
        valid_hours=int(valid.sum()),
        complete_year=complete,
        invalid_parameter_hours=invalid,
        modeled_generation_kwh=energy,
        annual_generation_kwh=energy if complete else None,
        unshaded_generation_kwh=float(np.nansum(unshaded_ac)) / 1000,
        plane_irradiation_kwh_m2=float(np.nansum(shaded_poa)) / 1000,
        unshaded_plane_irradiation_kwh_m2=float(np.nansum(unshaded_poa)) / 1000,
        horizon_sky_fraction=sky_fraction,
        peak_ac_kw=float(np.nanmax(ac)) / 1000 if valid.any() else 0,
        monthly=monthly,
        limitations=limitations,
    )
    hourly = io.StringIO(newline="")
    writer = csv.writer(hourly)
    writer.writerow(
        [
            "hour_start_utc",
            "ghi_wh_m2",
            "dni_wh_m2",
            "dhi_wh_m2",
            "air_temperature_c",
            "wind_10m_m_s",
            "sun_azimuth_degrees",
            "sun_elevation_degrees",
            "beam_visible",
            "poa_w_m2",
            "cell_temperature_c",
            "dc_w",
            "ac_w",
            "unshaded_ac_w",
            "valid",
        ]
    )
    for index, start in enumerate(starts):
        values = [
            ghi[index],
            dni[index],
            dhi[index],
            temperature[index],
            wind[index],
            sun_azimuth[index],
            sun_elevation[index],
            float(beam_visible[index]),
            shaded_poa[index],
            cell_temperature[index],
            dc[index],
            ac[index],
            unshaded_ac[index],
        ]
        writer.writerow(
            [
                start.isoformat(),
                *[float(v) if np.isfinite(v) else "" for v in values],
                bool(valid[index]),
            ]
        )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in {
            "request.json": request.model_dump_json().encode(),
            "result.json": metadata.model_dump_json().encode(),
            "source.json": raw,
            "hourly.csv": hourly.getvalue().encode(),
        }.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)
    archive_bytes = archive.getvalue()
    if len(archive_bytes) > 16 * 1024**2:
        raise ValueError("The solar output exceeded its 16 MiB storage limit.")
    return SolarResult(archive_bytes, metadata)


def analyze(boundary: Footprint, request: SolarRequest, client: httpx.Client) -> SolarResult:
    longitude, latitude, _ = location(boundary, request)
    raw, url, headers = retrieve(client, longitude, latitude, request.year)
    return calculate(boundary, request, raw, url, headers)
