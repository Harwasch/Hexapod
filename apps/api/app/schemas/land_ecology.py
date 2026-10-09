from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.base import CamelModel


class TaxonQuery(CamelModel):
    scientific_name: str = Field(min_length=1, max_length=200)
    kingdom: str | None = Field(default=None, min_length=1, max_length=80)

    @field_validator("scientific_name", "kingdom")
    @classmethod
    def meaningful_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter a scientific name or omit an unknown kingdom.")
        return value


class EcologyRequest(CamelModel):
    dataset: Literal["ecological-context"] = "ecological-context"
    taxa: list[TaxonQuery] = Field(default_factory=list, max_length=20)
    include_ecoregions: bool = True
    include_ecological_sites: bool = True
    include_occurrences: bool = True

    @model_validator(mode="after")
    def useful_request(self) -> EcologyRequest:
        keys = [(q.scientific_name.casefold(), (q.kingdom or "").casefold()) for q in self.taxa]
        if len(set(keys)) != len(keys):
            raise ValueError("Each name and kingdom combination must be unique.")
        if not self.taxa and not any(
            (self.include_ecoregions, self.include_ecological_sites, self.include_occurrences)
        ):
            raise ValueError("Choose a context source or at least one taxonomic name.")
        return self
