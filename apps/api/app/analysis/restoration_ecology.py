from __future__ import annotations

import uuid
from math import expm1

from app.schemas.land_surveys import SurveyRead
from app.schemas.restoration_ecology import (
    CoverProjection,
    CoverResponse,
    RestorationEcology,
    RestorationEcologyResult,
    SpeciesTargetResult,
)
from app.services.errors import InvalidInputError


def trajectory(baseline: float, response: CoverResponse, end_year: int) -> list[CoverProjection]:
    """Exact envelope for independent bounded parameters in C=A+(C0-A)*exp(-r*t)."""
    rows = []
    for year in range(end_year + 1):
        elapsed = max(0, year - response.start_year)
        values = [
            baseline + (asymptote - baseline) * -expm1(-rate * elapsed)
            for asymptote in (response.asymptote_low, response.asymptote_high)
            for rate in (response.annual_rate_low, response.annual_rate_high)
        ]
        rows.append(CoverProjection(year=year, low=max(0, min(values)), high=min(100, max(values))))
    return rows


def evaluate(
    plan: RestorationEcology, surveys: dict[uuid.UUID, SurveyRead]
) -> RestorationEcologyResult:
    results = []
    for target in plan.species_targets:
        baseline = target.baseline_percent
        result = SpeciesTargetResult(
            target=target,
            baseline_percent=baseline,
            baseline_scope="entered-assumption" if baseline is not None else "unknown",
            target_relation="not-modeled",
            limitations=[],
        )
        if target.baseline_survey_id:
            survey = surveys.get(target.baseline_survey_id)
            if survey is None:
                raise InvalidInputError("The species target's baseline survey is unavailable.")
            row = next(
                (
                    item
                    for item in survey.summary.species
                    if item.taxon.casefold() == target.taxon.casefold()
                    and item.stratum == target.stratum
                ),
                None,
            )
            if row is None:
                raise InvalidInputError(
                    f"{target.taxon} ({target.stratum}) has no recorded summary in the selected survey. "
                    "Missing observations cannot be treated as zero cover."
                )
            baseline = row.mean_percent
            result.baseline_percent = baseline
            result.baseline_scope = "surveyed-plots"
            result.survey_sha256 = survey.sha256
            result.observed_on = survey.observed_on
            result.assessed_area_m2 = row.assessed_area_m2
            result.assessed_land_fraction = min(
                1, row.assessed_area_m2 / survey.summary.land_area_m2
            )
            result.identification_status = row.identifications
            result.limitations.append(
                "Baseline cover is the immutable survey's area-weighted sampled-plot mean, "
                "not a whole-land estimate. Targets and any response curve use the same sampled scope. "
                "Identification, season, plot placement and sampling effort require review."
            )
        elif baseline is not None:
            result.limitations.append(
                "Baseline cover is an entered assumption; its stated basis and spatial scope have not been verified."
            )
        else:
            result.limitations.append(
                "Baseline cover is unknown. No response curve or zero baseline is substituted."
            )
        if target.response:
            if baseline is None:
                raise InvalidInputError(
                    "A conditional response curve requires a measured or explicitly assumed baseline."
                )
            result.projection = trajectory(baseline, target.response, target.target_year)
            last = result.projection[-1]
            result.target_relation = (
                "inside-target"
                if last.low >= target.target_low and last.high <= target.target_high
                else "outside-target"
                if last.high < target.target_low or last.low > target.target_high
                else "overlaps-target"
            )
            result.limitations.append(
                "The response envelope propagates the entered parameter bounds in an exponential approach "
                "to an assumed asymptote. It is a conditional what-if calculation, not a fitted ecological "
                "forecast, probability of success or statistical confidence interval. Treatment effects are "
                "not inferred from their names or costs. Disturbance, competition and changing rates are excluded."
            )
        results.append(result)
    return RestorationEcologyResult(
        reference_basis=plan.reference_basis,
        reference_evidence_ids=plan.reference_evidence_ids,
        site_constraints=plan.site_constraints,
        species=results,
        limitations=[
            "Species and vegetation strata overlap; do not sum or normalize their percentages "
            "into exclusive cover classes.",
            "Targets describe a proposed condition. Citations preserve the stated basis but do not "
            "automatically establish local applicability, native status or restoration suitability.",
            "Monitoring must use comparable plots, methods, strata and seasons. A parameter envelope "
            "is not an observed range or a probability interval.",
            "Species plans do not add treatment or monitoring costs: those are counted once "
            "in the scenario's explicit cost schedule.",
        ],
    )
