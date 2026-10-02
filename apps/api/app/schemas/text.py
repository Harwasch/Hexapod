"""A query's text embedding, for search by meaning (app/services/text_encoder.py)."""

from __future__ import annotations

from app.schemas.base import CamelModel


class TextEmbedding(CamelModel):
    """The query embedded with SigLIP 2's text tower, comparable with `instances.emb` rows.

    Use it only when `model` equals the scan's `instances.json` `embedding.model`; a
    vector from another model is not in the same space."""

    #: As `instances.json` names it: `siglip:google/siglip2-base-patch16-224`.
    model: str
    #: What was embedded: the query inside the tag template, e.g. "a photo of a spool.".
    prompt: str
    dim: int
    #: `dim` floats, L2-normalised (rounded to 6 decimals).
    embedding: list[float]
