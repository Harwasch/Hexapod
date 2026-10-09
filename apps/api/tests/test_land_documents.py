from __future__ import annotations

import hashlib
import io
import uuid

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.models.workspace import Workspace
from app.schemas.land_documents import LandDocumentCreate
from app.services import land_documents
from app.services.errors import InvalidInputError, NotFoundError
from tests.test_land import BODY

TEXT = b"Deed recorded in 1890. Mineral rights are reserved.\fA later amendment may change access rights."


def initiate(client, land_id, data=TEXT, media_type="text/plain", **overrides):
    payload = {
        "title": "Historic deed",
        "filename": "deed.txt" if media_type == "text/plain" else "deed.pdf",
        "mediaType": media_type,
        "sizeBytes": len(data),
        "kind": "deed",
        "sourceNote": "User supplied county record",
        "documentDate": "1890-01-01",
        "recordingNumber": "BOOK-4-PAGE-9",
        "requestKey": str(uuid.uuid4()),
        **overrides,
    }
    path = f"/api/v1/land/{land_id}/documents"
    result = client.post(path, json=payload)
    assert result.status_code == 201, result.text
    return path, payload, result.json()


def upload(client, land_id, data=TEXT, media_type="text/plain", **overrides):
    path, payload, document = initiate(client, land_id, data, media_type, **overrides)
    response = client.put(
        f"{path}/{document['id']}/content",
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    return path, payload, response.json()


def pdf(with_text=True):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if with_text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (Mineral rights reserved in the 1890 deed.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_private_immutable_originals_pages_idempotency_and_search(client, db):
    land = client.post("/api/v1/land", json=BODY).json()
    path, payload, document = upload(client, land["id"])
    assert document["status"] == "ready" and document["pageCount"] == 2
    assert document["sha256"] == hashlib.sha256(TEXT).hexdigest()
    identifier = f"{path}/{document['id']}"
    assert client.post(path, json=payload).json()["id"] == document["id"]
    assert client.post(path, json={**payload, "title": "Different metadata"}).status_code == 409
    assert client.put(identifier + "/content", content=TEXT).json()["id"] == document["id"]
    assert client.put(identifier + "/content", content=b"x" * len(TEXT)).status_code == 409
    response = client.get(identifier + "/content")
    assert response.content == TEXT
    assert response.headers["cache-control"] == "private, no-store"
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"
    page = client.get(identifier + "/pages/2").json()
    assert "amendment" in page["text"] and page["page"] == 2
    assert client.get(identifier + "/pages/3").status_code == 404
    hits = client.get(path + "/search", params={"q": "mineral"}).json()
    assert hits[0]["page"] == 1 and "Mineral rights" in hits[0]["excerpt"]
    assert client.get(path + "/search", params={"q": "%%%"}).json() == []
    private = Workspace(name="Other workspace")
    db.add(private)
    db.commit()
    with pytest.raises(NotFoundError):
        land_documents.scoped(db, private.id, uuid.UUID(land["id"]), uuid.UUID(document["id"]))
    other = client.post("/api/v1/land", json={**BODY, "name": "Other land"}).json()
    assert (
        client.get(f"/api/v1/land/{other['id']}/documents/{document['id']}/content").status_code
        == 404
    )


def test_pdf_extraction_scans_corruption_truncation_and_upload_bounds(client):
    land = client.post("/api/v1/land", json=BODY).json()
    path, _, document = upload(client, land["id"], pdf(), "application/pdf")
    assert document["status"] == "ready", document
    assert "1890 deed" in client.get(f"{path}/{document['id']}/pages/1").json()["text"]
    _, _, scan = upload(client, land["id"], pdf(False), "application/pdf")
    assert scan["status"] == "needs-ocr" and scan["extractedCharacters"] == 0
    _, _, corrupt = upload(client, land["id"], b"not a PDF", "application/pdf")
    assert corrupt["status"] == "unreadable" and corrupt["warnings"]
    _, _, long_page = upload(client, land["id"], b"A" * 25000)
    page = client.get(f"{path}/{long_page['id']}/pages/1").json()
    assert page["truncated"] and len(page["text"]) == 20000
    _, _, pending = initiate(client, land["id"], b"abc")
    assert client.put(f"{path}/{pending['id']}/content", content=b"abcd").status_code == 413
    assert client.put(f"{path}/{pending['id']}/content", content=b"a").status_code == 422
    assert client.get(f"{path}/{pending['id']}/content").status_code == 404


def test_document_relationships_pin_existing_pages_and_stay_in_land(client):
    land = client.post("/api/v1/land", json=BODY).json()
    path, _, source = upload(client, land["id"])
    _, _, target = upload(
        client,
        land["id"],
        b"This amendment supersedes the earlier access terms.",
        title="Amendment",
    )
    body = {
        "requestKey": str(uuid.uuid4()),
        "fromDocumentId": target["id"],
        "fromPage": 1,
        "toDocumentId": source["id"],
        "toPage": 2,
        "relation": "amends",
        "basis": "Both pages describe an amendment to access.",
    }
    response = client.post(path + "/links", json=body)
    assert response.status_code == 201, response.text
    assert client.post(path + "/links", json=body).json()["id"] == response.json()["id"]
    assert client.post(path + "/links", json={**body, "fromPage": 10}).status_code == 404
    assert (
        client.post(path + "/links", json={**body, "toDocumentId": str(uuid.uuid4())}).status_code
        == 404
    )
    assert len(client.get(path + "/links").json()) == 1


def test_pending_uploads_reserve_workspace_quota(client, db):
    land = client.post("/api/v1/land", json=BODY).json()
    _path, payload, document = initiate(client, land["id"])
    from app.models.workspace import PILOT_WORKSPACE_ID

    with pytest.raises(InvalidInputError, match="allowance"):
        land_documents.create(
            db,
            PILOT_WORKSPACE_ID,
            uuid.UUID(land["id"]),
            "operator",
            LandDocumentCreate.model_validate({**payload, "requestKey": str(uuid.uuid4())}),
            len(TEXT),
        )
    assert document["status"] == "awaiting-upload"


def test_agent_cites_exact_private_document_page_without_inventing_a_public_url(
    client, db, sessions
):
    import json

    from sqlalchemy import select

    from app.config import Settings
    from app.models.research import ResearchRun
    from app.research.model import (
        CompleteAction,
        DecisionResult,
        DocumentReadAction,
        FindingAction,
        ResearchDecision,
    )
    from app.research.worker import ResearchWorker
    from app.schemas.research import FindingContent
    from tests.test_research import start

    land, investigation, run, _ = start(client)
    _path, _payload, document = upload(client, land["id"])
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    row.kind = "investigation"
    db.commit()

    class Model:
        def decide(self, context, max_tokens):
            state = json.loads(context)
            if not state["previousActions"]:
                assert state["landDocuments"][0]["id"] == document["id"]
                action = DocumentReadAction(kind="read_document_pages", document_id=document["id"])
            else:
                source = next(iter(state["retrieved"].values()))
                identifier = source["evidenceIds"][0]
                if len(state["previousActions"]) == 1:
                    action = FindingAction(
                        kind="publish_finding",
                        finding=FindingContent(
                            title="A mineral reservation appears in the deed",
                            summary="Page 1 records a reservation of mineral rights in an 1890 deed.",
                            category="rights",
                            evidence_ids=[identifier],
                            confidence="uncertain",
                            uncertainty=(
                                "Current legal effect, parcel lineage and later releases "
                                "have not been established."
                            ),
                        ),
                    )
                else:
                    action = CompleteAction(
                        kind="complete",
                        summary="The deed records a mineral reservation; current applicability is unresolved.",
                        evidence_ids=[identifier],
                    )
            return DecisionResult(
                ResearchDecision(progress="Checking the document page", action=action), 100
            )

    assert ResearchWorker(sessions, Settings(_env_file=None), model=Model()).run_once()
    db.expire_all()
    detail = client.get(f"/api/v1/research/investigations/{investigation['id']}").json()
    assert len(detail["findings"]) == 1
    evidence = detail["evidence"][0]
    assert evidence["url"] is None
    assert (
        evidence["document"]["page"] == 1 and evidence["document"]["sha256"] == document["sha256"]
    )
    assert evidence["spatialRelevance"] == "unresolved"
    assert "Mineral rights" in evidence["excerpt"]
