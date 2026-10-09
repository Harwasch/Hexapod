from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.land_document import LandDocumentBlob, LandDocumentOcr
from app.schemas.land_documents import DocumentOcrCapabilities, DocumentOcrRead
from app.services import land_documents
from app.services.errors import InvalidInputError, NotFoundError


@lru_cache(maxsize=1)
def capabilities() -> DocumentOcrCapabilities:
    engine = shutil.which("tesseract")
    if not shutil.which("pdftoppm") or not engine:
        return DocumentOcrCapabilities(
            available=False,
            languages=[],
            reason="This server needs Poppler and Tesseract to read scanned pages.",
        )
    try:
        result = subprocess.run(  # noqa: S603 -- installed engine with fixed arguments
            [engine, "--list-langs"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        installed = set(result.stdout.decode().splitlines())
        languages = sorted(installed & {"eng", "spa", "fra", "deu"})
        return DocumentOcrCapabilities(
            available=bool(languages),
            languages=languages,
            reason="One page is processed at a time; machine readings require verification against the original.",
        )
    except (OSError, subprocess.SubprocessError):
        return DocumentOcrCapabilities(
            available=False, languages=[], reason="The OCR engine could not be started."
        )


def read(row: LandDocumentOcr) -> DocumentOcrRead:
    return DocumentOcrRead(
        **row.content,
        id=row.id,
        document_id=row.document_id,
        page=row.page,
        language=row.language,
        text=row.text,
        created_at=row.created_at,
    )


def scoped(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    document_id: uuid.UUID,
    ocr_id: uuid.UUID,
) -> LandDocumentOcr:
    land_documents.scoped(db, workspace_id, land_id, document_id)
    row = db.get(LandDocumentOcr, ocr_id)
    if row is None or row.document_id != document_id:
        raise NotFoundError("document OCR", ocr_id)
    return row


def extract(
    db: Session,
    workspace_id: uuid.UUID,
    land_id: uuid.UUID,
    document_id: uuid.UUID,
    page: int,
    language: str,
    *,
    commit: bool = True,
) -> DocumentOcrRead:
    document = land_documents.scoped(db, workspace_id, land_id, document_id, lock=True)
    land_documents.page(db, document, page)
    if document.metadata_json["media_type"] != "application/pdf":
        raise InvalidInputError(
            "OCR applies to PDF pages. Plain-text records are already readable."
        )
    identifier = uuid.uuid5(document_id, f"ocr/v1/{page}/{language}")
    existing = db.get(LandDocumentOcr, identifier)
    if existing is not None:
        return read(existing)
    availability = capabilities()
    if not availability.available or language not in availability.languages:
        raise InvalidInputError(f"OCR language {language} is unavailable. {availability.reason}")
    original = db.get(LandDocumentBlob, document_id)
    if original is None:
        raise NotFoundError("document original", document_id)
    with tempfile.TemporaryDirectory(prefix="land-ocr-") as temporary:
        path = Path(temporary) / "original.pdf"
        path.write_bytes(original.data)
        try:
            result = subprocess.run(  # noqa: S603 -- trusted module and generated path; validated page/language
                [sys.executable, "-m", "app.analysis.document_ocr", str(path), str(page), language],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=48,
                check=True,
            )
            content = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise InvalidInputError(
                "OCR exceeded its processing budget or could not start. The original is unchanged."
            ) from error
    if "error" in content:
        raise InvalidInputError(content["error"])
    text = content.pop("text")
    # Validate the complete persisted result before accepting a processor response.
    content["sha256"] = document.sha256
    if content["text_sha256"] != hashlib.sha256(text.encode()).hexdigest() or len(text) > 20000:
        raise InvalidInputError("OCR returned an invalid text result.")
    row = LandDocumentOcr(
        id=identifier,
        document_id=document_id,
        page=page,
        language=language,
        text=text,
        content=content,
    )
    db.add(row)
    db.flush()
    response = read(row)
    if commit:
        db.commit()
    return response


def preview(
    db: Session, workspace_id: uuid.UUID, land_id: uuid.UUID, document_id: uuid.UUID, page: int
) -> bytes:
    document = land_documents.scoped(db, workspace_id, land_id, document_id)
    land_documents.page(db, document, page)
    if document.metadata_json["media_type"] != "application/pdf":
        raise InvalidInputError("Page images are available for PDF originals.")
    original = db.get(LandDocumentBlob, document_id)
    if original is None:
        raise NotFoundError("document original", document_id)
    with tempfile.TemporaryDirectory(prefix="land-page-") as temporary:
        path = Path(temporary) / "original.pdf"
        path.write_bytes(original.data)
        try:
            result = subprocess.run(  # noqa: S603 -- trusted processor and generated file; validated page
                [sys.executable, "-m", "app.analysis.document_ocr", str(path), str(page), "image"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise InvalidInputError(
                "This page could not be rendered within the processing limits. Download the original for review."
            ) from error
    if not result.stdout.startswith(b"\x89PNG\r\n\x1a\n") or len(result.stdout) > 16 * 1024 * 1024:
        raise InvalidInputError("The renderer returned an invalid page image.")
    return result.stdout
