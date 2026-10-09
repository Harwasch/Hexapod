from __future__ import annotations

from math import cos, expm1, isfinite, log1p, radians
from typing import Any

from app.schemas.scenarios import RestorationInputs, ScenarioInputs, ScenarioResult, SolarInputs
from app.services.errors import InvalidInputError


def _solar_case(
    p: SolarInputs,
    yield_factor: float = 1,
    cost_factor: float = 1,
    annual_generation: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capacity = p.usable_roof_area_m2 * p.module_efficiency  # STC is 1 kW/m².
    generation = (
        capacity
        * p.annual_plane_irradiation_kwh_m2
        * (1 - p.shade_loss)
        * (1 - p.system_loss)
        * yield_factor
    )
    if annual_generation is not None:
        generation = annual_generation * yield_factor
    installed = p.installed_cost * cost_factor
    net_cost = max(0, installed - p.upfront_incentive)
    principal = net_cost * p.financed_fraction
    r = p.loan_interest_rate
    payment = (
        principal / p.loan_years if r == 0 else principal * r / -expm1(-p.loan_years * log1p(r))
    )
    equity = net_cost - principal
    cumulative = -equity
    discounted = -equity
    rows = [
        {
            "year": 0,
            "generationKwh": 0,
            "energyValue": 0,
            "maintenanceCost": 0,
            "debtService": 0,
            "replacementCost": 0,
            "netCashFlow": -equity,
            "cumulativeCashFlow": -equity,
            "discountedCashFlow": -equity,
        }
    ]
    for year in range(1, p.years + 1):
        energy = generation * (1 - p.degradation_per_year) ** (year - 1)
        tariff = (
            p.purchase_rate_per_kwh * p.self_consumption_fraction
            + p.export_rate_per_kwh * (1 - p.self_consumption_fraction)
        ) * (1 + p.tariff_escalation) ** (year - 1)
        value = energy * tariff
        maintenance = p.annual_maintenance_cost * (1 + p.maintenance_escalation) ** (year - 1)
        debt = payment if year <= p.loan_years else 0
        replacement = p.replacement_cost if year == p.replacement_year else 0
        cash = value - maintenance - debt - replacement
        cumulative += cash
        present = cash / (1 + p.discount_rate) ** year
        discounted += present
        rows.append(
            {
                "year": year,
                "generationKwh": energy,
                "energyValue": value,
                "maintenanceCost": maintenance,
                "debtService": debt,
                "replacementCost": replacement,
                "netCashFlow": cash,
                "cumulativeCashFlow": cumulative,
                "discountedCashFlow": present,
            }
        )
    # Report sustained cash recovery: a later replacement cost may undo a first crossing.
    payback = next(
        (
            row["year"]
            for index, row in enumerate(rows)
            if row["cumulativeCashFlow"] >= 0
            and all(later["cumulativeCashFlow"] >= 0 for later in rows[index:])
        ),
        None,
    )
    return rows, {
        "capacityKwDc": capacity,
        "firstYearGenerationKwh": generation,
        "netPresentValue": discounted,
        "initialEquity": equity,
        "annualLoanPayment": payment,
        "paybackYear": payback,
        "lifetimeNetCashFlow": cumulative,
        "currency": p.currency,
    }


def solar(
    p: SolarInputs, land_area_m2: float, annual_generation: float | None = None
) -> ScenarioResult:
    if annual_generation is not None and (not isfinite(annual_generation) or annual_generation < 0):
        raise InvalidInputError("Annual generation must be a finite nonnegative value.")
    projected_area = p.usable_roof_area_m2 * (
        cos(radians(p.tilt_degrees)) if annual_generation is not None else 1
    )
    if projected_area > land_area_m2 * 1.001:
        raise InvalidInputError(
            "Usable roof area exceeds the pinned land area. Select the correct land or revise the area."
        )
    rows, summary = _solar_case(p, annual_generation=annual_generation)
    sensitivity = []
    for yield_factor, cost_factor, label in [
        (0.8, 1, "Generation -20%"),
        (1.2, 1, "Generation +20%"),
        (1, 0.8, "Capital cost -20%"),
        (1, 1.2, "Capital cost +20%"),
    ]:
        _, case = _solar_case(p, yield_factor, cost_factor, annual_generation)
        sensitivity.append({"case": label, "netPresentValue": case["netPresentValue"]})
    return ScenarioResult(
        algorithm="solar-hourly-cash-flow/1"
        if annual_generation is not None
        else "solar-cash-flow/1",
        summary=summary,
        rows=rows,
        sensitivity=sensitivity,
        limitations=(
            [
                "Generation uses a pinned complete historical hourly assessment including orientation, "
                "temperature, shading and inverter effects. Those losses are not applied a second time.",
                "One historical weather year is repeated with entered degradation; it is not a forecast "
                "or an interannual uncertainty model. Read the assessment's physical limitations.",
                "Self-consumption and tariffs are annual entered assumptions, not an hourly load or tariff simulation.",
            ]
            if annual_generation is not None
            else [
                "Scenario estimate from supplied assumptions, not a measured or engineered roof design.",
                "Plane-of-array irradiation must already account for tilt and azimuth. Those angles are "
                "recorded, not used to transform a horizontal resource.",
                "Shade, usable roof area and self-consumption are supplied assumptions; no roof obstruction, "
                "structural or hourly load model has been run.",
            ]
        )
        + [
            "Annual end-of-year cash flows; tax effects, salvage value and additional incentives are "
            "excluded. Tariffs, financing and maintenance follow the entered assumptions.",
        ],
    )


def restoration(p: RestorationInputs, land_area_m2: float) -> ScenarioResult:
    area_ha = land_area_m2 / 10_000
    if any(treatment.area_ha > area_ha * 1.001 for treatment in p.treatments):
        raise InvalidInputError("A treatment area exceeds the pinned land area.")
    years: dict[int, float] = {}
    for treatment in p.treatments:
        years[treatment.year] = (
            years.get(treatment.year, 0) + treatment.area_ha * treatment.cost_per_ha
        )
    for year in p.monitoring_years:
        years[year] = years.get(year, 0) + p.monitoring_cost_per_visit
    total = sum(years.values()) * (1 + p.contingency_fraction)
    discounted = sum(
        cost * (1 + p.contingency_fraction) / (1 + p.discount_rate) ** year
        for year, cost in years.items()
    )
    rows = [
        {
            "class": row.name,
            "baselinePercent": row.baseline_percent,
            "targetPercent": row.target_percent,
            "baselineHa": row.baseline_percent / 100 * area_ha,
            "targetHa": row.target_percent / 100 * area_ha,
            "changeHa": (row.target_percent - row.baseline_percent) / 100 * area_ha,
        }
        for row in p.cover
    ]
    return ScenarioResult(
        algorithm="restoration-cover-and-cost/1",
        summary={
            "areaHa": area_ha,
            "unclassifiedBaselinePercent": max(
                0, 100 - sum(row.baseline_percent for row in p.cover)
            ),
            "totalCost": total,
            "presentValueCost": discounted,
            "currency": p.currency,
        },
        rows=rows,
        sensitivity=[
            {"case": f"Year {year}", "cost": cost * (1 + p.contingency_fraction)}
            for year, cost in sorted(years.items())
        ],
        limitations=[
            "Cover is supplied survey/estimate data. Occurrence records alone do not establish species coverage.",
            "Targets are a proposed future condition, not a prediction of establishment or ecological success.",
            "Classes are mutually exclusive cover categories; layered vegetation surveys require a "
            "separate classification.",
            "Treatment areas may overlap across activities and years; costs count each specified activity.",
        ],
    )


def analyze(
    inputs: ScenarioInputs, land_area_m2: float, annual_generation: float | None = None
) -> ScenarioResult:
    return (
        solar(inputs, land_area_m2, annual_generation)
        if isinstance(inputs, SolarInputs)
        else restoration(inputs, land_area_m2)
    )
