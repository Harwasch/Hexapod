"""Portable, bounded arithmetic recipes and their reproducible results."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.base import CamelModel

Number = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100)]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]


class CalculationInput(CamelModel):
    name: Name
    label: str = Field(min_length=1, max_length=200)
    unit: str = Field(min_length=1, max_length=100)
    values: list[Number | None] = Field(min_length=1, max_length=500)
    origin: Literal["evidence", "question", "assumption"]
    basis: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[uuid.UUID] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def sourced(self) -> CalculationInput:
        if (self.origin == "evidence") != bool(self.evidence_ids):
            raise ValueError(
                "Evidence inputs need citations; question/assumption inputs must not claim them."
            )
        return self


class CalculationFormula(CamelModel):
    name: Name
    label: str = Field(min_length=1, max_length=200)
    expression: str = Field(min_length=1, max_length=1000)
    unit: str = Field(min_length=1, max_length=100)


class CalculationRequest(CamelModel):
    purpose: str = Field(min_length=1, max_length=1000)
    limitations: str = Field(min_length=1, max_length=2000)
    row_labels: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        min_length=1, max_length=500
    )
    inputs: list[CalculationInput] = Field(min_length=1, max_length=20)
    formulas: list[CalculationFormula] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def dimensions(self) -> CalculationRequest:
        names = [value.name for value in self.inputs] + [value.name for value in self.formulas]
        if len(set(names)) != len(names):
            raise ValueError("Input and result names must be unique.")
        if any(len(item.values) not in {1, len(self.row_labels)} for item in self.inputs):
            raise ValueError("Each input needs one shared value or one value per row.")
        if len({identifier for item in self.inputs for identifier in item.evidence_ids}) > 100:
            raise ValueError("A calculation supports at most 100 distinct citations.")
        return self


class CalculationIssue(CamelModel):
    row: int = Field(ge=0, le=499)
    column: Name
    kind: Literal["missing", "undefined"]
    message: str = Field(min_length=1, max_length=300)


class CalculationOutput(CamelModel):
    kind: Literal["calculation"]
    engine_version: Literal["arithmetic-v1"]
    request: CalculationRequest
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rows: list[dict[str, Number | None]] = Field(min_length=1, max_length=500)
    issues: list[CalculationIssue] = Field(default_factory=list, max_length=10000)
