"""Fixed, resource-bounded page rendering/OCR; never executes document or model code."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

LANGUAGES = {"eng", "spa", "fra", "deu"}


def render_page(path: Path, page: int) -> Path:
    if not 1 <= page <= 500:
        raise ValueError("Unsupported page.")
    output = path.parent / "page"
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise ValueError("The page renderer is unavailable.")
    # All arguments except the trusted generated file path are validated scalars.
    subprocess.run(  # noqa: S603 -- fixed renderer command, no shell
        [
            renderer,
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            "-scale-to",
            "2400",
            "-png",
            str(path),
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
    )
    image = output.with_suffix(".png")
    if not image.is_file() or image.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("The rendered page exceeds the OCR image limit.")
    return image


def extract(path: Path, page: int, language: str) -> dict[str, Any]:
    if language not in LANGUAGES:
        raise ValueError("Unsupported OCR language.")
    engine = shutil.which("tesseract")
    if not engine:
        raise ValueError("The OCR engine is unavailable.")
    image = render_page(path, page)
    result = subprocess.run(  # noqa: S603 -- fixed OCR command, validated language
        [engine, str(image), "stdout", "-l", language, "--psm", "3", "tsv"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=25,
    )
    version = (
        subprocess.run(  # noqa: S603 -- installed engine with fixed arguments
            [engine, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        .stdout.decode(errors="replace")
        .splitlines()[0][:200]
    )
    lines: dict[tuple[str, str, str], list[str]] = {}
    confidences = []
    for row in csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    ):
        word = row.get("text", "").strip()
        if row.get("level") != "5" or not word:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        lines.setdefault(key, []).append(word)
        confidence = float(row["conf"])
        if 0 <= confidence <= 100:
            confidences.append(confidence)
    raw = "\n".join(" ".join(words) for words in lines.values())
    text = raw[:20000]
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "engine": "Tesseract",
        "engine_version": version,
        "render_max_pixels": 2400,
        "mean_word_confidence": sum(confidences) / len(confidences) if confidences else None,
        "truncated": len(raw) > len(text),
        "warnings": [
            "Machine-extracted text may misread names, numbers, boundaries and handwriting. "
            "Verify important passages against the original.",
            "OCR word confidence is an engine score, not a calibrated probability of correctness.",
        ]
        + (["No readable text was recognized on this page."] if not text else []),
    }


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    os.environ["OMP_THREAD_LIMIT"] = "1"
    if sys.argv[3] == "image":
        try:
            sys.stdout.buffer.write(render_page(Path(sys.argv[1]), int(sys.argv[2])).read_bytes())
        except Exception:
            raise SystemExit(1) from None
        return
    try:
        result = extract(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    except Exception:
        result = {
            "error": "OCR could not read this page within the rendering, time or memory limits. "
            "The original is unchanged."
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
