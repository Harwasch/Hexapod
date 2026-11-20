"""Analysis adapter: expiring bundle access stays outside the model's tool arguments."""

import asyncio
import base64
import importlib
import re
from pathlib import Path
from typing import Any, Protocol

import anyio

from app.config import Settings
from app.schemas.ask import AskRequest, Entity
from app.storage import ObjectStorage

TOOLS: list[dict[str, Any]] = [
    {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}, "taxon": {"type": "string"}},
            "required": ["prompt"],
            "additionalProperties": False,
        },
    }
    for name, description in (
        ("segment", "Find text-prompted instances with measured world anchors."),
        ("plant_id", "Identify measured plant instances; may be unavailable."),
        ("invasive_lookup", "Annotate measured instances using a location-scoped data source."),
        ("count_measure", "Count instances and bound observed surfaces using metric depth."),
    )
]


class Analysis(Protocol):
    async def context(
        self, request: AskRequest, bundle_url: str | None
    ) -> list[dict[str, Any]]: ...

    async def run(
        self, name: str, arguments: dict[str, Any], request: AskRequest, bundle_url: str | None
    ) -> list[Entity]: ...


class AnalysisUnavailableError(RuntimeError):
    pass


class ModalAnalysis:
    def __init__(self, settings: Settings, storage: ObjectStorage) -> None:
        self.settings, self.storage = settings, storage

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.ask_analysis_enabled:
            raise AnalysisUnavailableError("remote analysis is disabled")
        try:
            modal = importlib.import_module("modal")
        except ImportError as exc:
            raise AnalysisUnavailableError("remote analysis client is not installed") from exc
        function = modal.Function.from_name(self.settings.ask_analysis_app, "analyze")
        call = await function.spawn.aio(payload)
        try:
            result: dict[str, Any] = await call.get.aio(timeout=600)
            return result
        except (asyncio.CancelledError, TimeoutError):
            # A cancelled HTTP consumer must not leave remote work billing indefinitely.
            with anyio.CancelScope(shield=True):
                await call.cancel.aio(terminate_containers=True)
            raise

    def payload(self, request: AskRequest, bundle_url: str | None, action: str) -> dict[str, Any]:
        if bundle_url is None:
            raise AnalysisUnavailableError("a completed capture bundle is required")
        return {
            "action": action,
            "bundleUrl": bundle_url,
            "keyframes": [
                f.model_dump(by_alias=True, exclude_none=True) for f in request.keyframes
            ],
            "location": request.location.model_dump() if request.location else None,
        }

    async def context(self, request: AskRequest, bundle_url: str | None) -> list[dict[str, Any]]:
        if not request.keyframes:
            return []
        result = await self.call(self.payload(request, bundle_url, "context"))
        content: list[dict[str, Any]] = result["content"]
        return content

    async def run(
        self, name: str, arguments: dict[str, Any], request: AskRequest, bundle_url: str | None
    ) -> list[Entity]:
        if name not in {tool["name"] for tool in TOOLS}:
            raise ValueError("unknown analysis tool")
        result = await self.call(
            {**self.payload(request, bundle_url, name), "arguments": arguments}
        )
        if result.get("error"):
            raise AnalysisUnavailableError(result["error"])
        entities = [Entity.model_validate(value) for value in result["entities"]]
        # Each call has a distinct sidecar namespace, independent of user-supplied ids.
        import uuid

        prefix = f"captures/{request.scan_id}/analysis/{uuid.uuid4()}"
        refs: dict[str, str] = {}
        for artifact in result.get("artifacts", []):
            ref = artifact["ref"]
            if not re.fullmatch(r"analysis/mask-[0-9]+-[0-9]+\.png", ref):
                raise ValueError("invalid mask reference from analysis")
            data = base64.b64decode(artifact["data"], validate=True)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError("mask exceeds artifact limit")
            key = f"{prefix}/{Path(ref).name}"
            self.storage.put_object(key, data, "image/png")
            refs[ref] = key.removeprefix(f"captures/{request.scan_id}/")
        for entity in entities:
            for mask in entity.masks:
                if mask.ref not in refs:
                    raise ValueError("analysis returned a mask without its artifact")
                mask.ref = refs[mask.ref]
        return entities
