from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models.research import ResearchRun
from app.research import queue
from app.research.providers.archives import commons, historical_maps, open_license
from app.research.providers.base import SourceContext
from app.research.worker import ResearchWorker
from app.schemas.land_archives import ArchiveMedia
from app.schemas.research import ArtifactContent, GalleryOutput
from app.services.errors import InvalidInputError
from tests.test_research import evidence, start
from tests.test_terrain import BOUNDARY


def photo(identifier: int = 1, **metadata: dict[str, str]) -> dict[str, Any]:
    return {
        "pageid": identifier,
        "title": "File:Historical test photograph.jpg",
        "coordinates": [{"lon": -122.132, "lat": 47.648, "globe": "earth"}],
        "imageinfo": [
            {
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Historical_test.jpg",
                "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Test.jpg/640px-Test.jpg",
                "mime": "image/jpeg",
                "sha1": "test-source-version",
                "extmetadata": {
                    "ObjectName": {"value": "Historical test photograph"},
                    "Artist": {"value": "<a href='https://example.test'>A. Photographer</a>"},
                    "ImageDescription": {
                        "value": "<p>A documented event, location still needs verification.</p>"
                    },
                    "DateTimeOriginal": {"value": "circa 1890-1895"},
                    "DateTime": {"value": "2020-01-01"},
                    "LicenseShortName": {"value": "CC BY-SA 4.0"},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                    **metadata,
                },
            }
        ],
    }


def map_record() -> dict[str, Any]:
    return {
        "title": "USGS historical test sheet 1890",
        "sourceId": "map-fixture",
        "publicationDate": "1890-01-01",
        "metaUrl": "https://www.sciencebase.gov/catalog/item/fixture",
        "previewGraphicURL": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/PDF/fixture_tn.jpg",
        "downloadURL": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/PDF/fixture.pdf",
        "boundingBox": {"minX": -123, "maxX": -122, "minY": 47, "maxY": 48},
    }


def transport(request: httpx.Request) -> httpx.Response:
    if request.url.host == "commons.wikimedia.org":
        nearby = photo(2)
        nearby["coordinates"][0]["lon"] = -122.12
        restricted = photo(
            3, LicenseUrl={"value": "https://creativecommons.org/licenses/by-nc/4.0/"}
        )
        return httpx.Response(
            200, json={"query": {"pages": {"1": photo(), "2": nearby, "3": restricted}}}
        )
    return httpx.Response(200, json={"items": [map_record()], "total": 1, "errors": []})


def test_archive_metadata_retains_uncertain_dates_locations_and_licensed_attribution() -> None:
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        result = commons(SourceContext(BOUNDARY), http)
        maps = historical_maps(SourceContext(BOUNDARY), http)
    assert len(result.evidence) == 2 and result.data["excludedLicense"] == 1
    first, nearby = result.evidence[0][1], result.evidence[1][1]
    assert first.observed_at is None and first.media is not None
    assert first.media.source_date == "circa 1890-1895"  # Never substitute the upload date.
    assert first.media.creator == "A. Photographer"
    assert "<" not in first.media.description
    assert first.spatial_relevance == "within" and nearby.spatial_relevance == "nearby"
    assert "does not prove" in first.relevance_note
    sheet = maps.evidence[0][1]
    assert sheet.spatial_relevance == "intersects" and sheet.media is not None
    assert sheet.media.source_date == "1890" and "georeference" in sheet.relevance_note
    assert sheet.media.location_meaning == "catalog-footprint"
    assert open_license({"LicenseShortName": {"value": "Unknown"}}) is None
    with pytest.raises(ValueError, match="host"):
        ArchiveMedia.model_validate(
            {**first.media.model_dump(), "preview_url": "https://unregistered.example/image.jpg"}
        )


def test_archive_run_without_model_publishes_saved_media_and_validated_galleries(
    client: TestClient, db: Session, sessions: sessionmaker[Session]
) -> None:
    _, inv, run, _ = start(client)
    row = db.get(ResearchRun, uuid.UUID(run["id"]))
    assert row is not None
    row.kind = "archive"
    db.commit()
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        assert ResearchWorker(sessions, Settings(_env_file=None), client=http).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{inv['id']}").json()
    assert detail["runs"][0]["status"] == "succeeded", detail
    assert len(detail["evidence"]) == 3 and len(detail["artifacts"]) == 2
    assert all(item["output"]["kind"] == "gallery" for item in detail["artifacts"])
    assert detail["evidence"][0]["media"]["creator"] == "A. Photographer"
    media_id = uuid.UUID(detail["evidence"][0]["id"])
    start(client)
    claimed = queue.claim(db)
    assert claimed is not None
    local = queue.save_evidence(db, *claimed, "ordinary", evidence())
    with pytest.raises(InvalidInputError):
        queue.save_artifact(
            db,
            *claimed,
            "foreign",
            ArtifactContent(
                title="Gallery",
                method="Fixture",
                evidence_ids=[local],
                output=GalleryOutput(kind="gallery", evidence_ids=[media_id]),
            ),
        )
    db.rollback()
    with pytest.raises(InvalidInputError, match="archive media"):
        queue.save_artifact(
            db,
            *claimed,
            "ordinary",
            ArtifactContent(
                title="Gallery",
                method="Fixture",
                evidence_ids=[local],
                output=GalleryOutput(kind="gallery", evidence_ids=[local]),
            ),
        )
