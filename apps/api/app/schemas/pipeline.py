"""The recipe catalogue, and the providers a run can be dispatched to.

A6 made recipes data (`tools/pipeline/recipes/*.yaml`) rather than code. This is that
data, read back out over HTTP, because the console's New-run form has to offer every
parameter of every stage and it must not carry a second copy of the recipe files to do
it. A recipe edited on disk changes the form with no frontend build.
"""

from __future__ import annotations

from typing import Any

from app.schemas.base import CamelModel


class RecipeGpu(CamelModel):
    """A stage's GPU requirement. Its presence is the only routing signal there is."""

    tier: str
    preemptible: bool


class RecipeStageRead(CamelModel):
    id: str
    impl: str
    #: The stage's own defaults, which `jobs.params[stageId]` is merged over.
    params: dict[str, Any]
    gpu: RecipeGpu | None


class RecipeRead(CamelModel):
    name: str
    #: The recipe file's own `version`, as a string — it is stamped onto `jobs.recipe_version`.
    version: str
    description: str
    inputs: list[str]
    stages: list[RecipeStageRead]


class ProviderRead(CamelModel):
    """One GPU host a run can be sent to, with the rate the plan's survey recorded.

    `usdPerHourA100` is a reference point, not a quote: it is what the A0 provider survey
    measured for an A100, recorded here so the console can say which tier is the cheap one
    and which is the reliable one. B1's `ProviderAdapter` carries the real per-tier price
    table and records `jobs.cost_usd` from what a run actually cost.
    """

    name: str
    label: str
    #: Tiers this provider is offered with here. Free strings, as `jobs.tier` is.
    tiers: list[str]
    usd_per_hour_a100: float
    #: The published or surveyed hourly rate for each tier that has one. A tier absent
    #: here is unpriced, and a client must say so rather than guess. What lets a page
    #: estimate the cost of a stage that is still running; a finished stage's real cost
    #: is in its step metrics.
    usd_per_hour: dict[str, float]
    #: True where being killed mid-stage is ordinary operation rather than a failure.
    interruptible: bool
    note: str


class PipelineCatalogue(CamelModel):
    """Everything the New-run form needs: what can be run, and where."""

    recipes: list[RecipeRead]
    providers: list[ProviderRead]
