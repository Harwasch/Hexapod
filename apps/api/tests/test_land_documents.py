from __future__ import annotations

import hashlib
import io
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy.orm import Session, sessionmaker

from app.models.workspace import Workspace
from app.schemas.land_documents import LandDocumentCreate
from app.services import land_documents
from app.services.errors import InvalidInputError, NotFoundError
from tests.test_land import BODY

TEXT = b"Deed recorded in 1890. Mineral rights are reserved.\fA later amendment may change access rights."


def initiate(
    client: TestClient,
    land_id: str,
    data: bytes = TEXT,
    media_type: str = "text/plain",
    **overrides: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
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


def upload(
    client: TestClient,
    land_id: str,
    data: bytes = TEXT,
    media_type: str = "text/plain",
    **overrides: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    path, payload, document = initiate(client, land_id, data, media_type, **overrides)
    response = client.put(
        f"{path}/{document['id']}/content",
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    return path, payload, response.json()


def pdf(with_text: bool = True) -> bytes:
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


def test_private_immutable_originals_pages_idempotency_and_search(
    client: TestClient, db: Session
) -> None:
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


def test_pdf_extraction_scans_corruption_truncation_and_upload_bounds(client: TestClient) -> None:
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


def test_document_relationships_pin_existing_pages_and_stay_in_land(client: TestClient) -> None:
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


def test_pending_uploads_reserve_workspace_quota(client: TestClient, db: Session) -> None:
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


@pytest.mark.parametrize("use_ocr", [False, True])
def test_agent_cites_exact_private_document_page_without_inventing_a_public_url(
    client: TestClient, db: Session, sessions: sessionmaker[Session], use_ocr: bool
) -> None:
    import json

    from sqlalchemy import select

    from app.config import Settings
    from app.models.research import ResearchRun
    from app.research.model import (
        CompleteAction,
        DecisionResult,
        DocumentOcrAction,
        DocumentReadAction,
        FindingAction,
        ResearchDecision,
    )
    from app.research.worker import ResearchWorker
    from app.schemas.research import FindingContent
    from tests.test_research import start

    land, investigation, run, _ = start(client)
    if use_ocr:
        from app.services import document_ocr

        if not document_ocr.capabilities().available:
            pytest.skip("OCR engine unavailable")
    _path, _payload, document = upload(
        client,
        land["id"],
        pdf() if use_ocr else TEXT,
        "application/pdf" if use_ocr else "text/plain",
    )
    row = db.scalar(select(ResearchRun).where(ResearchRun.id == run["id"]))
    assert row is not None
    row.kind = "investigation"
    db.commit()

    class Model:
        def decide(self, context: str, max_tokens: int) -> DecisionResult:
            action: CompleteAction | DocumentOcrAction | DocumentReadAction | FindingAction
            state = json.loads(context)
            if not state["previousActions"]:
                assert state["landDocuments"][0]["id"] == document["id"]
                action = (
                    DocumentOcrAction(kind="ocr_document_page", document_id=document["id"], page=1)
                    if use_ocr
                    else DocumentReadAction(kind="read_document_pages", document_id=document["id"])
                )
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
    if use_ocr:
        reading = client.get(
            f"/api/v1/land/{land['id']}/documents/{document['id']}/ocr/{evidence['document']['ocrId']}"
        ).json()
        assert evidence["excerpt"] == reading["text"]
        assert "1890" in evidence["excerpt"]
    else:
        assert "Mineral rights" in evidence["excerpt"]
    assert bool(evidence["document"]["ocrId"]) == use_ocr


def test_scanned_pdf_ocr_is_private_immutable_searchable_and_citable(
    client: TestClient, db: Session
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    from app.models.workspace import PILOT_WORKSPACE_ID
    from app.research.documents import retrieve
    from app.services import document_ocr

    if not document_ocr.capabilities().available:
        pytest.skip("Install Poppler and Tesseract with English language data for OCR integration")
    image = Image.new("RGB", (1400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (60, 80), "MINERAL RIGHTS RESERVED", fill="black", font=ImageFont.load_default(size=55)
    )
    draw.text((60, 180), "Recorded in 1890", fill="black", font=ImageFont.load_default(size=45))
    buffer = io.BytesIO()
    image.save(buffer, format="PDF")
    original = buffer.getvalue()
    land = client.post("/api/v1/land", json=BODY).json()
    path, _, document = upload(client, land["id"], original, "application/pdf")
    assert document["status"] == "needs-ocr"
    url = f"{path}/{document['id']}/pages/1/ocr"
    response = client.post(url, json={"language": "eng"})
    assert response.status_code == 200, response.text
    ocr = response.json()
    assert "MINERAL RIGHTS RESERVED" in ocr["text"]
    assert ocr["sha256"] == hashlib.sha256(original).hexdigest()
    assert ocr["textSha256"] == hashlib.sha256(ocr["text"].encode()).hexdigest()
    assert ocr["engine"] == "Tesseract" and ocr["warnings"]
    assert client.post(url, json={"language": "eng"}).json() == ocr
    assert len(client.get(url).json()) == 1
    assert client.get(f"{path}/{document['id']}/pages/1").json()["text"] == ""
    assert client.get(f"{path}/{document['id']}/content").content == original
    preview = client.get(f"{path}/{document['id']}/pages/1/image")
    assert preview.status_code == 200 and preview.content.startswith(b"\x89PNG")
    assert preview.headers["cache-control"] == "private, no-store"
    hits = client.get(path + "/search", params={"q": "MINERAL"}).json()
    assert hits[0]["ocrId"] == ocr["id"]
    found = retrieve(db, PILOT_WORKSPACE_ID, uuid.UUID(land["id"]), query="MINERAL")
    citation = found.evidence[0][1]
    assert citation.document is not None
    assert str(citation.document.ocr_id) == ocr["id"]
    assert "Machine OCR" in citation.relevance_note
    assert found.data["pages"][0]["extraction"] == "machine-ocr"
    other = client.post("/api/v1/land", json={**BODY, "name": "Other land"}).json()
    assert (
        client.get(
            f"/api/v1/land/{other['id']}/documents/{document['id']}/ocr/{ocr['id']}"
        ).status_code
        == 404
    )
    assert (
        client.post(url.replace("/pages/1/", "/pages/2/"), json={"language": "eng"}).status_code
        == 404
    )
    assert client.post(url, json={"language": "eng;cat"}).status_code == 422


def test_ocr_unavailable_does_not_change_the_original(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.schemas.land_documents import DocumentOcrCapabilities
    from app.services import document_ocr

    monkeypatch.setattr(
        document_ocr,
        "capabilities",
        lambda: DocumentOcrCapabilities(available=False, languages=[], reason="Missing OCR engine"),
    )
    land = client.post("/api/v1/land", json=BODY).json()
    original = pdf(False)
    path, _, document = upload(client, land["id"], original, "application/pdf")
    response = client.post(f"{path}/{document['id']}/pages/1/ocr", json={"language": "eng"})
    assert response.status_code == 422
    assert client.get(f"{path}/{document['id']}/content").content == original
