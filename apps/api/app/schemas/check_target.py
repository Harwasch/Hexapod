"""A site's surveyed check targets (M3b), as the API reads and replaces them."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field, field_validator

from app.models.enums import CheckTargetRole, HeightDatum
from app.schemas.base import CamelModel

#: The most targets one site holds: a survey sets out a handful (LIVING_WORLD.md section 6
#: reserves two or three checkpoints; the synthetic tree has eight). A ceiling only, so a
#: payload cannot be made arbitrarily large.
MAX_TARGETS = 200


class CheckTargetCreate(CamelModel):
    """One surveyed mark: where it is in WGS84, from which height datum, and how well.

    `sigmaHM` / `sigmaVM` are the survey's own one-sigma, in metres, as its instrument
    reported them: they decide the tolerance an epoch's check RMSE is held to (3 sigma of
    the declared uncertainties, `tools/pipeline/check_targets.survey_tolerance`), so a
    wrong one makes a wrong claim.
    """

    label: str = Field(min_length=1, max_length=64)
    role: CheckTargetRole = CheckTargetRole.CHECK
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(ge=-180, le=180, allow_inf_nan=False)
    height: float = Field(allow_inf_nan=False)
    height_datum: HeightDatum
    sigma_h_m: float = Field(gt=0, allow_inf_nan=False)
    sigma_v_m: float = Field(gt=0, allow_inf_nan=False)
    #: How it was surveyed: `rtk-rover`, `total-station`, `synthetic`, ...
    source: str = Field(min_length=1, max_length=64)
    surveyed_at: datetime | None = None

    @field_validator("surveyed_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("surveyedAt needs a time zone (e.g. a trailing Z)")
        return value


class CheckTargetSet(CamelModel):
    """The site's whole active set. A label sent with the same values is kept as it is; a
    label sent with different values retires its row and starts a new one; a label not
    sent is retired. Rows are never deleted, so what an epoch was scored with stays."""

    targets: list[CheckTargetCreate] = Field(max_length=MAX_TARGETS)

    @field_validator("targets")
    @classmethod
    def _unique_labels(cls, targets: list[CheckTargetCreate]) -> list[CheckTargetCreate]:
        labels = [target.label for target in targets]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(f"each label names one mark; repeated: {', '.join(repeated)}")
        return targets


class CheckTargetRead(CamelModel):
    """An active check target. Read model: every field is explicit (no defaults) so the
    OpenAPI contract marks it required."""

    id: uuid.UUID
    site_id: uuid.UUID
    label: str
    role: CheckTargetRole
    lat: float
    lon: float
    height: float
    height_datum: HeightDatum
    sigma_h_m: float
    sigma_v_m: float
    source: str
    surveyed_at: datetime | None
    created_at: datetime
