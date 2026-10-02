from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response

from app.schemas.common import Problem
from app.schemas.text import TextEmbedding
from app.services.text_encoder import MAX_QUERY_CHARS, LazyTextEncoder, prompt_for, query_label

router = APIRouter(tags=["search"])


def _text_encoder(request: Request) -> LazyTextEncoder:
    encoder: LazyTextEncoder = request.app.state.text_encoder
    return encoder


TextEncoderDep = Annotated[LazyTextEncoder, Depends(_text_encoder)]

UNAVAILABLE: dict[int | str, dict[str, Any]] = {
    503: {"model": Problem, "description": "No text encoder on this deployment, or it failed"},
}


@router.get(
    "/text-embeddings",
    response_model=TextEmbedding,
    responses=UNAVAILABLE,
    summary="Embed a search query with the model scan objects were described with",
    description=(
        "SigLIP 2's text tower (google/siglip2-base-patch16-224, the model segmentation "
        "embedded each object's views with), on the query inside the tags' template "
        '("a photo of a {}."). The viewer ranks a scan\'s objects by the cosine between '
        "this and their rows of instances.emb. A leading article is dropped (the template "
        "has one). The first call on a machine loads the model (about 1.5 s); then about "
        "140 ms. The same text always gives the same vector, so responses are cacheable. "
        "503 when this deployment has no encoder: search falls back to tags."
    ),
)
def text_embedding(
    encoder: TextEncoderDep,
    response: Response,
    text: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARS)],
) -> TextEmbedding:
    if not query_label(text):
        raise ValueError("the query names nothing to search for")
    prompt = prompt_for(text)
    name, vector = encoder.embed(prompt)
    response.headers["Cache-Control"] = "public, max-age=86400"
    return TextEmbedding(
        model=name,
        prompt=prompt,
        dim=int(vector.shape[0]),
        embedding=[round(float(v), 6) for v in vector],
    )
