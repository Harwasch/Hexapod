"""Bounded PDF/text extraction in a disposable subprocess; no links or scripts execute."""

from __future__ import annotations

import io
import json
import re
import resource
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 500
MAX_PAGE_CHARACTERS = 20000
MAX_TOTAL_CHARACTERS = 2_000_000


def extract(data: bytes, media_type: str) -> dict[str, Any]:
    if len(data) > MAX_BYTES:
        raise ValueError("The document exceeds the 20 MiB limit.")
    warnings: list[str] = []
    page_text: Iterable[tuple[int, str]]
    if media_type == "application/pdf":
        from pypdf import PdfReader

        if not data.startswith(b"%PDF-"):
            raise ValueError("This file does not have a PDF signature.")
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ValueError("Upload an unencrypted copy of the document.")
        total_pages = len(reader.pages)
        if total_pages > MAX_PAGES:
            raise ValueError("Split documents longer than 500 pages before uploading.")
        page_text = (
            (index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)
        )
    elif media_type == "text/plain":
        text = data.decode("utf-8-sig")
        # A form-feed explicitly marks a page; never manufacture page numbers from line counts.
        pieces = text.split("\f")
        total_pages = len(pieces)
        if total_pages > MAX_PAGES:
            raise ValueError("Split documents longer than 500 pages before uploading.")
        page_text = enumerate(pieces, 1)
        warnings.append("Text page numbers follow form-feed separators supplied in the file.")
    else:
        raise ValueError("Choose a PDF or UTF-8 text document.")
    pages = []
    characters = 0
    for number, raw in page_text:
        clean = re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", raw).strip()
        text = clean[: min(MAX_PAGE_CHARACTERS, max(0, MAX_TOTAL_CHARACTERS - characters))]
        truncated = len(text) < len(clean)
        pages.append({"page": number, "text": text, "truncated": truncated})
        characters += len(text)
    if any(page["truncated"] for page in pages):
        warnings.append(
            "Extracted text is limited to 20,000 characters per page and 2,000,000 per document. "
            "The original file is preserved."
        )
    if any(not page["text"] for page in pages):
        warnings.append(
            "Some pages have no extractable text. Scanned pages, handwriting and figures need OCR or visual review."
        )
    return {
        "status": "ready" if characters else "needs-ocr",
        "pages": pages,
        "page_count": total_pages,
        "characters": characters,
        "warnings": warnings,
    }


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        result = extract(Path(sys.argv[1]).read_bytes(), sys.argv[2])
    except Exception as error:
        result = {
            "status": "unreadable",
            "pages": [],
            "page_count": 0,
            "characters": 0,
            "warnings": [
                "Text extraction failed. The original is preserved for review.",
                str(error)[:500],
            ],
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
