"""Replaceable analysis modules. Missing inference backends fail explicitly."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from hex_analysis.lifting import F64, bounds, lift_depth


class AnalysisUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    index: int
    image: bytes
    depth: F64 | None
    intrinsics: F64
    transform: F64
    image_size: tuple[int, int]


@dataclass(frozen=True)
class Detection:
    label: str
    mask: npt.NDArray[np.bool_]
    confidence: float
    model: str
    attrs: dict[str, Any] = field(default_factory=dict)


class Segmenter(Protocol):
    """Return independent instance masks on the depth grid, without inventing depth."""

    def segment(self, frame: Frame, prompt: str) -> list[Detection]: ...


class PlantIdentifier(Protocol):
    def identify(self, frame: Frame, detection: Detection) -> dict[str, Any]: ...


class InvasiveSource(Protocol):
    def lookup(self, lat: float, lon: float, taxon: str) -> dict[str, Any]: ...


class SAM3Segmenter:
    """Deployment supplies the licensed weights and tested inference implementation.

    GPU execution and mask alignment cannot be verified on the development machine.
    No synthetic detections stand in for missing weights or a missing dependency.
    """

    def segment(self, frame: Frame, prompt: str) -> list[Detection]:
        raise AnalysisUnavailableError("segmentation backend and weights are not configured")


class OpenPlantIdentifier:
    """Open-weight backend seam; species accuracy and backend selection remain unverified."""

    model_id = "open-plant-unselected"

    def identify(self, frame: Frame, detection: Detection) -> dict[str, Any]:
        raise AnalysisUnavailableError("plant identification backend is not selected")


class FixtureInvasiveSource:
    """Tiny demonstration dataset, never a claim of comprehensive invasive status."""

    def lookup(self, lat: float, lon: float, taxon: str) -> dict[str, Any]:
        if 32 <= lat <= 42 and -125 <= lon <= -114 and taxon.casefold() == "cortaderia jubata":
            return {"taxon": taxon, "listed": True, "source": "fixture", "region": "California"}
        return {"taxon": taxon, "listed": None, "source": "fixture", "coverage": "unknown"}


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


@dataclass
class Registry:
    segmenter: Segmenter = field(default_factory=SAM3Segmenter)
    plants: PlantIdentifier = field(default_factory=OpenPlantIdentifier)
    invasive: InvasiveSource = field(default_factory=FixtureInvasiveSource)

    def run(
        self,
        name: str,
        arguments: dict[str, Any],
        frames: list[Frame],
        location: dict[str, float] | None,
        write_mask: Callable[[int, npt.NDArray[np.bool_]], str],
    ) -> list[dict[str, Any]]:
        if name not in {tool["name"] for tool in TOOLS}:
            raise ValueError("unknown analysis tool")
        if set(arguments) - {"prompt", "taxon"} or not isinstance(arguments.get("prompt"), str):
            raise ValueError("tool requires a text prompt")
        entities: list[dict[str, Any]] = []
        # Avoid double counting across viewpoints until cross-frame association is verified.
        # The reference keyframe is explicit in attrs and mask provenance.
        if not frames:
            raise AnalysisUnavailableError("no keyframes supplied")
        frame = frames[0]
        if frame.depth is None:
            raise AnalysisUnavailableError("reference keyframe has no metric depth")
        for detection in self.segmenter.segment(frame, arguments["prompt"]):
            points = lift_depth(
                frame.depth, detection.mask, frame.intrinsics, frame.transform, frame.image_size
            )
            measured = bounds(points, int(detection.mask.sum()))
            if measured is None:
                continue
            attrs = {
                **detection.attrs,
                "validDepthPixels": measured.valid_pixels,
                "selectedPixels": measured.selected_pixels,
                "measurement": "observed-surface-bounds",
                "referenceFrame": frame.index,
            }
            if name == "plant_id":
                attrs.update(self.plants.identify(frame, detection))
            if name == "invasive_lookup":
                if location is None:
                    raise AnalysisUnavailableError("location is required for invasive lookup")
                taxon = arguments.get("taxon", detection.label)
                if not isinstance(taxon, str):
                    raise ValueError("taxon must be text")
                attrs["invasive"] = self.invasive.lookup(location["lat"], location["lon"], taxon)
            # A phone location is not a calibrated ARKit-world to WGS84 transform.
            # Retain it as context; never attach it to every object's geo anchor.
            entities.append(
                {
                    "label": detection.label,
                    "anchor": {"world": dict(zip(("x", "y", "z"), measured.center, strict=True))},
                    "extent": dict(zip(("x", "y", "z"), measured.extent, strict=True)),
                    "masks": [
                        {
                            "keyframe": f"frames.jsonl#{frame.index}",
                            "ref": write_mask(frame.index, detection.mask),
                        }
                    ],
                    "confidence": detection.confidence,
                    "attrs": attrs,
                    "source": {"module": name, "model": detection.model},
                }
            )
        if name == "count_measure":
            for entity in entities:
                entity["attrs"]["instanceCountInReferenceFrame"] = len(entities)
        return entities
