"""The recipe catalogue, read from `tools/pipeline` rather than restated here.

A6's whole point is that a recipe is data. So the API does not keep a table of stage
names and parameter defaults — it reads the same YAML the executor reads, and every
answer it gives about a recipe (its version, its stages, whether a parameter override
names a stage that exists) is the pipeline's answer, not a second implementation of it.

Two facts make this a tolerant import rather than a plain one. `tools/pipeline` is a flat
uv project put on `sys.path` by `app.worker.pipeline_bridge`, not an installed package;
and `infra/api.Dockerfile` copies only `apps/api`, so in that image there is no pipeline
to read. A deployment without it keeps working — captures upload, jobs queue against the
shipped versions below, the console says the catalogue is unavailable — instead of the
API failing at import over a directory the API itself never needs.
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

#: Where a GPU stage can be sent, and the rate A0's survey recorded for an A100.
#:
#: Deliberately only the figures the plan actually measured. A per-tier price table that
#: nobody has checked would be worse than none: it would put invented numbers in front of
#: a person deciding how to spend money. B1's `ProviderAdapter` brings the real table and
#: records what a run cost in `jobs.cost_usd`, which is the number that settles it.
PROVIDERS: tuple[ProviderRead, ...] = (
    ProviderRead(
        name="modal",
        label="Modal",
        tiers=["l4", "a10g", "a100"],
        usd_per_hour_a100=2.50,
        interruptible=False,
        note="Per-second billing, scale to zero. The reliable default.",
    ),
    ProviderRead(
        name="runpod-secure",
        label="RunPod Secure",
        tiers=["l4", "a100"],
        usd_per_hour_a100=1.59,
        interruptible=False,
        note="Ordinary rented pods.",
    ),
    ProviderRead(
        name="runpod-community",
        label="RunPod Community",
        tiers=["l4", "a100"],
        usd_per_hour_a100=1.19,
        interruptible=True,
        note="Community hosts; cheaper, and killed without warning.",
    ),
    ProviderRead(
        name="vast",
        label="Vast.ai",
        tiers=["l4", "a100"],
        usd_per_hour_a100=0.52,
        interruptible=True,
        note=(
            "Cheapest sticker, most variable. A0 measured the unverified tier running "
            "20-40% above its listed price once restarts and downtime are priced in."
        ),
    ),
)


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
        providers=list(PROVIDERS),
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
