"""The recipe catalogue, read from `tools/pipeline` rather than restated here.

A6's whole point is that a recipe is data. So the API does not keep a table of stage
names and parameter defaults — it reads the same YAML the executor reads, and every
answer it gives about a recipe (its version, its stages, whether a parameter override
names a stage that exists) is the pipeline's answer, not a second implementation of it.

The import is tolerant rather than plain because `tools/pipeline` is a flat uv project
put on `sys.path` by `app.worker.pipeline_bridge`, not an installed package, so whether it
is there is a fact about the host. Until B1a `infra/api.Dockerfile` copied only
`apps/api`, and the deployed API therefore always took the `None` branch below; it now
ships the pipeline and `PIPELINE_DIR` with it, and the `image` job in CI asserts this
endpoint lists both shipped recipes. The tolerance stays for the hosts that still lack it
— a seeder run from a wheel, a sidecar with a different layout — where a deployment keeps
working (captures upload, jobs queue against the shipped versions below, the console says
the catalogue is unavailable) instead of the API failing at import over a directory the
API itself never needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.schemas.pipeline import (
    PipelineCatalogue,
    ProviderRead,
    RecipeGpu,
    RecipeRead,
    RecipeStageRead,
)

if TYPE_CHECKING:  # pragma: no cover - import is for typing only
    from app.worker.pipeline_bridge import Recipe

#: The versions of the shipped recipes, for a deployment that has no `tools/pipeline`.
#:
#: Kept equal to the `version:` in the recipe files, so `jobs.recipe_version` reads the
#: same whether or not the catalogue is on the path. It is a fallback, not a source of
#: truth: where the recipes are readable they are read.
SHIPPED_RECIPE_VERSIONS: dict[str, str] = {
    "splat-ingest": "2",
    "photo-reconstruct": "2",
}


def _providers() -> list[ProviderRead]:
    """Where a GPU stage can be sent, as the console renders it.

    The table itself moved to `tools/pipeline/providers.py` in B1b, because two things
    now need it and only one of them may import the other: this endpoint renders it, and
    `CloudRunner` multiplies a tier's rate by the seconds a provider billed to fill in
    `jobs.cost_usd`. A second copy here is how the price the console shows and the price
    a run is charged at would eventually disagree.

    `usdPerHourA100` is still what A0 measured, and still a reference point rather than a
    quote. The table now also carries published list prices where a provider publishes
    one, each tagged with its URL and the date it was read, so the L4 the shipped recipe
    actually requests has a price at last. What has not changed is the rule: a tier
    nobody can price still has no number, which is why Vast has one rate and not three.
    A deployment's own rates arrive through `PIPELINE_GPU_RATES` and are merged over
    these; see that module for the rest.

    Only reached from `catalogue()`, which already answers None on a host with no
    `tools/pipeline`, so this import cannot be the thing that makes the endpoint fail.
    """
    from app.worker.pipeline_bridge import rates_from_env, with_rates

    return [
        ProviderRead(
            name=entry.name,
            label=entry.label,
            tiers=list(entry.tiers),
            usd_per_hour_a100=entry.usd_per_hour_a100 or 0.0,
            interruptible=entry.interruptible,
            note=entry.note,
        )
        for entry in with_rates(rates_from_env())
    ]


def _load_recipes() -> list[Recipe] | None:
    """Every shipped recipe, or None where `tools/pipeline` is not on this host.

    The import is inside the function because `app.worker.pipeline_bridge` puts the
    pipeline on `sys.path` at import time and raises when it is not there — which is the
    right behaviour for a worker and the wrong behaviour for an API that only wants to
    list what it can offer.
    """
    try:
        from app.worker.pipeline_bridge import load_recipe, recipe_dir
    except ModuleNotFoundError:
        return None
    return [load_recipe(path) for path in sorted(recipe_dir().glob("*.yaml"))]


def available() -> bool:
    return _load_recipes() is not None


def _to_read(recipe: Recipe) -> RecipeRead:
    return RecipeRead(
        name=recipe.name,
        version=str(recipe.version),
        description=recipe.description,
        inputs=list(recipe.inputs),
        stages=[
            RecipeStageRead(
                id=stage.id,
                impl=stage.impl,
                params=dict(stage.params),
                gpu=(
                    RecipeGpu(tier=stage.gpu.tier, preemptible=stage.gpu.preemptible)
                    if stage.gpu
                    else None
                ),
            )
            for stage in recipe.stages
        ],
    )


def catalogue() -> PipelineCatalogue | None:
    """What can be run and where, or None where the recipes are not readable."""
    recipes = _load_recipes()
    if recipes is None:
        return None
    return PipelineCatalogue(
        recipes=[_to_read(recipe) for recipe in recipes],
        providers=_providers(),
    )


def version_of(name: str) -> str | None:
    """The version a run of `name` should be stamped with, or None for an unknown recipe."""
    recipes = _load_recipes()
    if recipes is None:
        return SHIPPED_RECIPE_VERSIONS.get(name)
    for recipe in recipes:
        if recipe.name == name:
            return str(recipe.version)
    return None


def known_names() -> list[str]:
    recipes = _load_recipes()
    if recipes is None:
        return sorted(SHIPPED_RECIPE_VERSIONS)
    return sorted(recipe.name for recipe in recipes)


def check_overrides(name: str, params: dict[str, Any]) -> None:
    """Refuse a parameter override the recipe cannot honour, in the recipe's own words.

    This is `Recipe.with_params` (A6) and nothing else: the API does not know which stages
    a recipe has, it asks. An override naming a stage the recipe lacks raises `RecipeError`
    there, and the message — which names the stages the recipe *does* have — is what the
    console shows. Duplicating that check here is how the two would eventually disagree.

    Silent where the pipeline is not on the path: the worker performs the same merge when
    the run starts, so the check is a better error message, never the only one.
    """
    if not params:
        return
    recipes = _load_recipes()
    if recipes is None:
        return
    from app.worker.pipeline_bridge import PipelineError

    for recipe in recipes:
        if recipe.name == name:
            try:
                recipe.with_params(_stage_keyed(name, params))
            except PipelineError as error:
                # Re-raised as a ValueError only to reach the 422 handler; the message is
                # the pipeline's, word for word, because it is the one that knows which
                # stages the recipe has.
                raise ValueError(str(error)) from error
            return


def _stage_keyed(recipe: str, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`jobs.params` checked for the shape `Recipe.with_params` takes.

    The same shape `app.worker.params` insists on, checked at launch instead of when the
    run starts: a caller who sent `{"lat": 51.5}` meaning `{"georeference": {"lat": 51.5}}`
    hears about it while the form is still open.
    """
    keyed: dict[str, dict[str, Any]] = {}
    for stage_id, overrides in params.items():
        if not isinstance(overrides, dict):
            raise ValueError(
                f"recipe '{recipe}': `params[{stage_id!r}]` must be an object of that "
                f"stage's parameters, for example "
                f"{{'georeference': {{'lat': 51.5, 'lon': -0.12}}}}"
            )
        keyed[str(stage_id)] = dict(overrides)
    return keyed
