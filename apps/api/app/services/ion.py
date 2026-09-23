"""Cesium ion integration.

Cesium ion is the *current delivery provider* for reality models. This module
wraps the documented parts of its REST API (https://cesium.com/learn/ion/rest-api/)
that the product needs today:

* reading asset metadata / tiling status (``GET /v1/assets/{id}``) so the UI can
  monitor reconstruction jobs the user started in ion;
* a ``ReconstructionProvider`` seam for creating jobs once the photo-input
  ``sourceType`` for reconstruction is documented in ion's public OpenAPI spec.

As of the 2026-09-08 revision of ion's OpenAPI document, reconstruction *outputs*
(``outputs: [{outputType: 3DTILES | LAS | SPLATS_3DTILES}]``) and the
``reconstructionJob`` / ``calibrationJob`` options are documented for
``POST /v1/assets``, but the ``sourceType`` value that accepts photo inputs is
not. We therefore do not create jobs from this backend (no reverse-engineering);
users run reconstruction in ion and register the resulting asset IDs here.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from app.config import Settings, get_settings
from app.schemas.ion import IonAssetMetadata, IonReconstructionCapabilities, IonStatus

CREATE_JOBS_REASON = (
    "Creating photo-reconstruction jobs is not enabled: Cesium ion's public OpenAPI spec "
    "documents reconstruction outputs but not the photo-input sourceType. Run reconstruction "
    "in Cesium ion and register the resulting asset IDs here; the backend can monitor them."
)


class IonError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ReconstructionProvider(Protocol):
    """Seam for reality-reconstruction services (ion today; others later)."""

    def capabilities(self) -> IonReconstructionCapabilities: ...

    def get_asset(self, asset_id: int) -> IonAssetMetadata: ...


class IonClient:
    def __init__(
        self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._settings.cesium_ion_server_token)

    def status(self) -> IonStatus:
        return IonStatus(
            configured=self.configured,
            api_base=self._settings.cesium_ion_api_base,
            reconstruction=self.capabilities(),
        )

    def capabilities(self) -> IonReconstructionCapabilities:
        return IonReconstructionCapabilities(
            monitor_jobs=self.configured,
            register_assets=True,
            create_jobs=False,
            create_jobs_reason=CREATE_JOBS_REASON,
        )

    def get_asset(self, asset_id: int) -> IonAssetMetadata:
        if not self.configured:
            raise IonError(503, "CESIUM_ION_SERVER_TOKEN is not configured")
        url = f"{self._settings.cesium_ion_api_base.rstrip('/')}/v1/assets/{asset_id}"
        headers = {"Authorization": f"Bearer {self._settings.cesium_ion_server_token}"}
        with httpx.Client(timeout=15.0, transport=self._transport) as client:
            response = client.get(url, headers=headers)
        if response.status_code == 404:
            raise IonError(404, f"ion asset {asset_id} not found or not accessible")
        if response.status_code == 401:
            raise IonError(502, "ion rejected the server token (401)")
        if response.status_code >= 400:
            raise IonError(502, f"ion returned HTTP {response.status_code}")
        return IonAssetMetadata.model_validate(response.json())
