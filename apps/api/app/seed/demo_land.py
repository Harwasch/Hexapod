"""Oak Bench Ranch as a saved Land in the pilot workspace: an opt-in seed for the joined loop.

The demo ranch is a catalog site (`demo_ranch.py`) whose simulated fleet works three zones.
Land, research and mission planning need it as a saved Land too. Selected while the ranch's
project is in view, the Land layers onto that project ("layer" mode in
`apps/web/src/missions/landGround.ts`): the simulated fleet stays, and the Land's boundary and
inventory join as private zones a plan can name ("monitor the fence every week"). This writes
that Land from `demo_ranch.json`, the file the site and the web app read, through the services
behind the Land panel's own routes:

* **The Land "Oak Bench Ranch"**: the demo outline as its boundary, recorded as a study area
  (not a property line), with the file's account of what is simulated.
* **Three inventory features**, each a candidate (nobody has checked them on the ground) that
  names its source, clipped to the outline:
  - *Perimeter fence*, a line on the outline's east, north and west edges. Simulated: no fence
    was mapped or surveyed here.
  - *Figueroa Creek*, the USGS NHD flowlines (public domain), merged into one line.
  - *Oak woodland*, the ESA WorldCover 2021 tree cover and shrubland patch (CC BY 4.0).
    WorldCover does not map species; oak is assumed from the region's oak savanna.

Nothing observed is made up: no inspections, findings, evidence, rasters or research results.
Those come from people and from real runs. `--research` queues the Land's real overview (what
the panel's overview button does), which the research worker (`pnpm dev:research`) runs
against public sources.

    uv run python -m app.seed                        # the catalog, with the demo-ranch site
    uv run python -m app.seed.demo_land              # the Land (idempotent)
    uv run python -m app.seed.demo_land --research   # ... and queue its overview

Idempotent by stable ids: the Land's id derives from its workspace, and each feature's from the
Land and the feature's key in the file (which is also its external reference,
`demo_ranch.json` / `fence`). A rerun creates what is missing and brings the outline, the words
and the sources back to the file's as new revisions, so the history keeps any edit made in
between. A name or a status set in the panel, evidence linked to a feature, and anything a user
added to the Land are left as they are.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import TypeAdapter
from shapely.geometry import LineString, MultiLineString, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, transform
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.land_feature import LandFeature
from app.models.workspace import PILOT_WORKSPACE_ID, Workspace
from app.schemas.geojson import Footprint, MapGeometry
from app.schemas.land import BoundarySource, LandCreate, LandRead, LandRevise
from app.schemas.land_features import FeatureExternalRef, LandFeatureCreate, LandFeatureRevise
from app.schemas.research import RunRead
from app.seed.demo_ranch import load
from app.services import land as land_service
from app.services import land_features, research
from app.services.errors import NotFoundError
from app.services.identity import principal_id

logger = logging.getLogger("twin.seed.demo_land")

#: The records are the pilot operator's, as the API names the pilot (`workspace_deps`).
PILOT_PRINCIPAL = principal_id("pilot", "operator")
#: The seeded features' external reference: the file they come from, keyed as in it.
NAMESPACE = "demo_ranch.json"
#: The revision note on every change a rerun makes.
NOTE = "Brought in step with demo_ranch.json (python -m app.seed.demo_land)"
#: Coordinates are stored to 1 cm, as the ranch's derived zones are.
DIGITS = 7

FOOTPRINT: TypeAdapter[Footprint] = TypeAdapter(Footprint)
GEOMETRY: TypeAdapter[MapGeometry] = TypeAdapter(MapGeometry)

Outcome = Literal["created", "updated", "unchanged"]


@dataclass
class Seeded:
    land_id: uuid.UUID
    land: Outcome
    revision: int
    area_m2: float
    features: dict[str, Outcome] = field(default_factory=dict)
    overview: RunRead | None = None
    #: Whether a research worker will pick the overview up now; None when none was queued.
    worker: str | None = None


def land_id(workspace_id: uuid.UUID, slug: str) -> uuid.UUID:
    return uuid.uuid5(workspace_id, f"demo-land/{slug}")


def feature_request_key(identifier: uuid.UUID, key: str) -> uuid.UUID:
    return uuid.uuid5(identifier, f"demo-land/{key}")


def feature_id(identifier: uuid.UUID, key: str) -> uuid.UUID:
    """The id `land_features.create` gives the feature with this seed's request key."""
    return uuid.uuid5(identifier, f"feature/{feature_request_key(identifier, key)}")


def _rounded[G: BaseGeometry](geometry: G) -> G:
    def rounded(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        if isinstance(x, float):
            return round(x, DIGITS), round(y, DIGITS)
        return [round(v, DIGITS) for v in x], [round(v, DIGITS) for v in y]

    result: G = transform(rounded, geometry)
    return result


def _geometry(geometry: BaseGeometry) -> MapGeometry:
    return GEOMETRY.validate_python(mapping(geometry))


def _clipped_line(geometry: dict[str, Any], boundary: BaseGeometry, key: str) -> LineString:
    line = shape(geometry)
    if isinstance(line, MultiLineString):
        line = linemerge(line)
    inside = line.intersection(boundary)
    if isinstance(inside, MultiLineString):
        inside = linemerge(inside)
    if not isinstance(inside, LineString) or inside.is_empty:
        raise ValueError(
            f"features.{key} must be one continuous line inside the outline; "
            f"it is {inside.geom_type} there"
        )
    return _rounded(inside)


def _clipped_polygon(geometry: dict[str, Any], boundary: BaseGeometry, key: str) -> Polygon:
    inside = shape(geometry).intersection(boundary)
    if not isinstance(inside, Polygon) or inside.is_empty:
        raise ValueError(f"features.{key} must stay one polygon inside the outline")
    return _rounded(inside)


def _source(document: dict[str, Any], source_id: str) -> dict[str, Any]:
    source: dict[str, Any] = document["sources"][source_id]
    return source


def _credit(source: dict[str, Any]) -> str:
    licence = source.get("license") or {}
    return f"{source['text']} ({licence['name']})" if licence.get("name") else source["text"]


def land_payload(document: dict[str, Any]) -> LandCreate:
    outline = _source(document, document["boundary"]["source"])
    creek = document["features"]["creek"]
    return LandCreate(
        name=document["name"],
        description=(
            f"{document['description']} {creek['name']} and the oak woodland come from public "
            "maps (USGS NHD, ESA WorldCover 2021); the perimeter fence is simulated. Nothing "
            "here has been inspected on the ground."
        ),
        boundary=FOOTPRINT.validate_python(document["boundary"]["geometry"]),
        source=BoundarySource(
            method="imported",
            label=f"{outline['label']} ({NAMESPACE})",
            record_id=document["slug"],
            attribution=outline["text"],
            meaning="study-area",
        ),
    )


def feature_payloads(
    document: dict[str, Any], identifier: uuid.UUID
) -> dict[str, LandFeatureCreate]:
    """The inventory, keyed as in the file. Every geometry is clipped to the outline."""
    boundary = shape(document["boundary"]["geometry"])
    features = document["features"]
    fence, creek, oak = features["fence"], features["creek"], features["oakWoodland"]
    nhd, worldcover = _source(document, creek["source"]), _source(document, oak["source"])

    def payload(key: str, **fields: Any) -> LandFeatureCreate:
        return LandFeatureCreate(
            request_key=feature_request_key(identifier, key),
            status="candidate",
            external_ref=FeatureExternalRef(namespace=NAMESPACE, record_id=key),
            **fields,
        )

    return {
        "fence": payload(
            "fence",
            name="Perimeter fence",
            # The inventory has no fence category; "other" is what the panel offers for one.
            category="other",
            geometry=_geometry(_clipped_line(fence["geometry"], boundary, "fence")),
            source=BoundarySource(
                method="drawn",
                label="Demo outline fence, simulated",
                meaning="physical-feature",
            ),
            description=(
                f"{fence['note']} Simulated: no fence was mapped or surveyed here. It is here "
                "so fence work can be planned."
            ),
            attributes={"simulated": True},
        ),
        "creek": payload(
            "creek",
            name=creek["name"],
            category="water",
            geometry=_geometry(_clipped_line(creek["geometry"], boundary, "creek")),
            source=BoundarySource(
                method="mapped-feature",
                label=f"{nhd['label']} (NHD), {creek['name']}",
                url=nhd["url"],
                attribution=_credit(nhd),
                meaning="physical-feature",
            ),
            description=f"{creek['note']} Merged into one line. Not checked on the ground.",
        ),
        "oakWoodland": payload(
            "oakWoodland",
            name="Oak woodland",
            category="vegetation",
            geometry=_geometry(_clipped_polygon(oak["geometry"], boundary, "oakWoodland")),
            source=BoundarySource(
                method="imagery",
                label=f"{worldcover['label']} tree cover and shrubland",
                url=worldcover["url"],
                attribution=_credit(worldcover),
                meaning="physical-feature",
            ),
            # Not the file's note: it gives the unclipped patch's acres, and the panel shows
            # the area of the geometry saved here.
            description=(
                "The largest ESA WorldCover 2021 patch of tree cover and shrubland (classes 10 "
                "and 20) in the demo outline. WorldCover does not map species: oak is assumed "
                "from the region's oak savanna. Not checked on the ground."
            ),
            attributes={"species": "oak (assumed)"},
        ),
    }


def _outline(footprint: Footprint) -> list[list[list[tuple[float, float]]]]:
    """Rings rounded to 1 cm, so the stored MultiPolygon and the file's Polygon compare."""
    polygons = [footprint.coordinates] if footprint.type == "Polygon" else footprint.coordinates
    return [
        [[(round(p[0], DIGITS), round(p[1], DIGITS)) for p in ring] for ring in polygon]
        for polygon in polygons
    ]


def _land(
    db: Session, workspace_id: uuid.UUID, identifier: uuid.UUID, wanted: LandCreate
) -> tuple[LandRead, Outcome]:
    try:
        existing = land_service.get_land(db, workspace_id, identifier)
    except NotFoundError:
        return land_service.create_land(db, workspace_id, wanted, identifier=identifier), "created"
    if (
        _outline(existing.boundary) == _outline(wanted.boundary)
        and existing.description == wanted.description
        and existing.source == wanted.source
    ):
        return existing, "unchanged"
    # The name is the user's once the Land exists: the panel lets them rename it.
    revised = LandRevise(
        **wanted.model_dump(exclude={"name"}),
        name=existing.name,
        expected_revision=existing.revision,
        note=NOTE,
    )
    return land_service.revise_land(db, workspace_id, identifier, revised), "updated"


def _feature(
    db: Session,
    workspace_id: uuid.UUID,
    identifier: uuid.UUID,
    key: str,
    wanted: LandFeatureCreate,
) -> Outcome:
    row = db.get(LandFeature, feature_id(identifier, key))
    if row is None:
        land_features.create(db, workspace_id, identifier, wanted)
        return "created"
    current = LandFeatureCreate.model_validate(row.content)
    # What a user decides stays theirs: the name, the status (confirmed, retired), the evidence
    # they linked, and the request key of their last edit.
    kept = wanted.model_copy(
        update={
            "name": current.name,
            "status": current.status,
            "evidence_ids": current.evidence_ids,
            "request_key": current.request_key,
        }
    )
    if kept.model_dump(mode="json") == row.content:
        return "unchanged"
    land_features.revise(
        db,
        workspace_id,
        identifier,
        row.id,
        LandFeatureRevise(**kept.model_dump(), expected_revision=row.revision, note=NOTE),
    )
    return "updated"


def seed_demo_land(
    db: Session,
    *,
    workspace_id: uuid.UUID = PILOT_WORKSPACE_ID,
    queue_research: bool = False,
    document: dict[str, Any] | None = None,
) -> Seeded:
    """Creates Oak Bench Ranch, or brings it in step with the file, in the workspace."""
    document = document or load()
    if workspace_id == PILOT_WORKSPACE_ID:
        # The API makes the pilot workspace on first use; a fresh database may not have it yet.
        db.execute(
            insert(Workspace)
            .values(id=PILOT_WORKSPACE_ID, name="Pilot workspace")
            .on_conflict_do_nothing(index_elements=[Workspace.id])
        )
    identifier = land_id(workspace_id, document["slug"])
    land, outcome = _land(db, workspace_id, identifier, land_payload(document))
    seeded = Seeded(identifier, outcome, land.revision, land.area_m2)
    for key, wanted in feature_payloads(document, identifier).items():
        seeded.features[key] = _feature(db, workspace_id, identifier, key, wanted)
    if queue_research:
        # One overview per boundary revision: a rerun returns the one already queued or done.
        _, seeded.overview = research.ensure_overview(
            db, workspace_id, identifier, PILOT_PRINCIPAL, land.revision
        )
        worker, _ = research.research_status(db, workspace_id)
        seeded.worker = worker.state
    return seeded


def report(seeded: Seeded) -> str:
    acres = seeded.area_m2 / 4046.8564224
    lines = [
        f"Oak Bench Ranch: {seeded.land} (land {seeded.land_id}, revision {seeded.revision}, "
        f"{acres:,.0f} ac) in the pilot workspace",
        "features: " + ", ".join(f"{key} {outcome}" for key, outcome in seeded.features.items()),
    ]
    if seeded.overview is not None:
        run = seeded.overview
        line = f"overview: {run.status} (run {run.id})"
        if run.status == "queued" and seeded.worker != "online":
            line += "; no research worker is running, start one with `pnpm dev:research`"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.seed.demo_land",
        description=(
            "Save the demo ranch, Oak Bench Ranch, as a Land in the pilot workspace, with its "
            "fence, creek and oak woodland as inventory features. Idempotent."
        ),
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="also queue the Land's overview for the research worker (pnpm dev:research)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from app.db import get_session_factory

    with get_session_factory()() as session:
        seeded = seed_demo_land(session, queue_research=args.research)
    sys.stdout.write(report(seeded))


if __name__ == "__main__":
    main()
