from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas.common import Problem
from app.schemas.pipeline import PipelineCatalogue
from app.services import recipes as recipe_service
from app.services.errors import NotFoundError

router = APIRouter(prefix="/recipes", tags=["recipes"])

CATALOGUE_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": Problem, "description": "No recipe catalogue on this host"}
}


@router.get(
    "",
    response_model=PipelineCatalogue,
    responses=CATALOGUE_RESPONSES,
    summary="The recipes a run can use, and the providers it can be sent to",
    description=(
        "Read from `tools/pipeline/recipes/*.yaml`, not restated here: a recipe edited on "
        "disk changes what the console offers with no rebuild. Each stage carries its own "
        "parameter defaults, which is what `params` on a new run is merged over, keyed by "
        "stage id. 404 where the pipeline project is not on this host — the API itself "
        "does not need it, so it is not a fault, and jobs still queue against the shipped "
        "recipe versions."
    ),
)
def get_catalogue() -> PipelineCatalogue:
    catalogue = recipe_service.catalogue()
    if catalogue is None:
        raise NotFoundError(
            "recipe catalogue",
            "tools/pipeline is not on this host; set PIPELINE_DIR to where it lives",
        )
    return catalogue
