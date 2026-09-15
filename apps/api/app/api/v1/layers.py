from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.api.deps import DbSession
from app.schemas.layer import LayerCreate, LayerRead, LayerUpdate
from app.services import layers as layer_service

router = APIRouter(prefix="/layers", tags=["layers"])


@router.get("", response_model=list[LayerRead], summary="List catalog layers")
def list_layers(db: DbSession) -> list[LayerRead]:
    return [layer_service.layer_to_read(layer) for layer in layer_service.list_layers(db)]


@router.post(
    "", response_model=LayerRead, status_code=status.HTTP_201_CREATED, summary="Add a layer"
)
def create_layer(payload: LayerCreate, db: DbSession) -> LayerRead:
    return layer_service.layer_to_read(layer_service.create_layer(db, payload))


@router.get("/{layer_id}", response_model=LayerRead, summary="Get a layer")
def get_layer(layer_id: uuid.UUID, db: DbSession) -> LayerRead:
    return layer_service.layer_to_read(layer_service.get_layer(db, layer_id))


@router.patch("/{layer_id}", response_model=LayerRead, summary="Update a layer")
def update_layer(layer_id: uuid.UUID, payload: LayerUpdate, db: DbSession) -> LayerRead:
    return layer_service.layer_to_read(layer_service.update_layer(db, layer_id, payload))


@router.delete("/{layer_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a layer")
def delete_layer(layer_id: uuid.UUID, db: DbSession) -> None:
    layer_service.delete_layer(db, layer_id)
