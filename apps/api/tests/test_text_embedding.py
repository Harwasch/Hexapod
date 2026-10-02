"""`GET /api/v1/text-embeddings`: a search query in the space scan objects were embedded in.

No database. The route and the lazy holder run against a fake encoder; the real one
(onnxruntime + the exported tower) runs only where `TEXT_ENCODER_DIR` points at an export
(`tools/captures/export_text_encoder.py`), since the export is ~565 MB and needs torch to
make. Its parity with transformers is checked by the export itself, at image build time.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import REPO_ROOT, Settings
from app.main import create_app
from app.services import text_encoder as te

DIM = 8


class FakeEncoder:
    name = te.MODEL_NAME
    dim = DIM

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, prompt: str) -> np.ndarray:
        self.calls.append(prompt)
        v = np.arange(1, DIM + 1, dtype=np.float32) * (1 + len(prompt) % 3)
        return v / np.linalg.norm(v)


class Loader:
    """Stands in for `OnnxTextEncoder`: counts loads, can fail."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.loads = 0
        self.fail = fail
        self.encoder = FakeEncoder()

    def __call__(self, directory: Path, threads: int) -> FakeEncoder:
        self.loads += 1
        if self.fail is not None:
            raise self.fail
        return self.encoder


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "hexapod.text-encoder"}))
    return tmp_path


def _client(holder: te.LazyTextEncoder) -> Iterator[TestClient]:
    app = create_app(Settings())
    app.state.text_encoder = holder
    with TestClient(app) as client:
        yield client


@pytest.fixture
def loader() -> Loader:
    return Loader()


@pytest.fixture
def client(export_dir: Path, loader: Loader) -> Iterator[TestClient]:
    yield from _client(te.LazyTextEncoder(export_dir, loader=loader))


# ---- the prompt is segmentation's ------------------------------------------------------


def _captures_constant(name: str) -> str:
    source = (REPO_ROOT / "tools" / "captures" / "segment_models.py").read_text()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = node.value
            assert isinstance(value, ast.Constant) and isinstance(value.value, str)
            return value.value
    raise AssertionError(f"{name} is not in segment_models.py")


def test_same_model_and_template_as_segmentation() -> None:
    assert _captures_constant("SIGLIP_MODEL") == te.MODEL
    assert _captures_constant("TAG_TEMPLATE") == te.TAG_TEMPLATE
    assert f"siglip:{te.MODEL}" == te.MODEL_NAME  # SiglipEmbedder.name


@pytest.mark.parametrize(
    ("typed", "prompt"),
    [
        ("spool", "a photo of a spool."),
        ("  Cable   SPOOL ", "a photo of a cable spool."),
        ("a cable spool", "a photo of a cable spool."),
        ("an apple.", "a photo of a apple."),  # the template says "a", as the tags did
        ("The flagpole", "a photo of a flagpole."),
        ("a photo of a log cabin.", "a photo of a log cabin."),
        ("antenna", "a photo of a antenna."),  # "an" only as a word
    ],
)
def test_prompt_is_the_tag_template(typed: str, prompt: str) -> None:
    assert te.prompt_for(typed) == prompt


def test_token_ids_match_the_processor_layout() -> None:
    class Pieces:
        def encode(self, text: str) -> list[int]:
            return [10 + ord(c) for c in text]

    ids = te.token_ids(Pieces(), "Ab", max_length=6, eos=1, pad=0)
    assert ids == [10 + ord("a"), 10 + ord("b"), 1, 0, 0, 0]
    assert te.token_ids(Pieces(), "x" * 20, max_length=6, eos=1, pad=0)[-1] == 1


# ---- the route ---------------------------------------------------------------------------


def test_embeds_the_prompt(client: TestClient, loader: Loader) -> None:
    response = client.get("/api/v1/text-embeddings", params={"text": "A Cable spool"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "siglip:google/siglip2-base-patch16-224"
    assert body["prompt"] == "a photo of a cable spool."
    assert body["dim"] == DIM and len(body["embedding"]) == DIM
    assert np.linalg.norm(body["embedding"]) == pytest.approx(1, abs=1e-5)
    assert "max-age" in response.headers["cache-control"]
    assert loader.encoder.calls == ["a photo of a cable spool."]


def test_loads_once_and_caches_prompts(client: TestClient, loader: Loader) -> None:
    for text in ("spool", "Spool", "a spool", "pumpkin"):
        assert client.get("/api/v1/text-embeddings", params={"text": text}).status_code == 200
    assert loader.loads == 1
    assert loader.encoder.calls == ["a photo of a spool.", "a photo of a pumpkin."]


@pytest.mark.parametrize("text", ["", "x" * (te.MAX_QUERY_CHARS + 1)])
def test_rejects_empty_and_long_queries(client: TestClient, text: str) -> None:
    assert client.get("/api/v1/text-embeddings", params={"text": text}).status_code == 422


def test_rejects_a_query_that_is_only_an_article(client: TestClient, loader: Loader) -> None:
    response = client.get("/api/v1/text-embeddings", params={"text": "the"})
    assert response.status_code == 422
    assert loader.loads == 0


def test_unconfigured_is_503() -> None:
    for client in _client(te.LazyTextEncoder(None)):
        response = client.get("/api/v1/text-embeddings", params={"text": "spool"})
        assert response.status_code == 503
        assert response.json()["status"] == 503
        assert "TEXT_ENCODER_DIR" in response.json()["detail"]


def test_missing_export_is_503(tmp_path: Path, loader: Loader) -> None:
    for client in _client(te.LazyTextEncoder(tmp_path / "absent", loader=loader)):
        assert client.get("/api/v1/text-embeddings", params={"text": "x"}).status_code == 503
    assert loader.loads == 0


def test_a_failed_load_is_503_and_not_retried(export_dir: Path) -> None:
    loader = Loader(fail=ImportError("No module named 'onnxruntime'"))
    for client in _client(te.LazyTextEncoder(export_dir, loader=loader)):
        for _ in range(3):
            response = client.get("/api/v1/text-embeddings", params={"text": "spool"})
            assert response.status_code == 503
            assert "onnxruntime" in response.json()["detail"]
    assert loader.loads == 1


def test_the_settings_reach_the_holder(export_dir: Path) -> None:
    app = create_app(Settings(text_encoder_dir=export_dir, text_encoder_threads=3))
    holder = app.state.text_encoder
    assert isinstance(holder, te.LazyTextEncoder)
    assert holder._directory == export_dir and holder._threads == 3


# ---- the real encoder, where an export is present ------------------------------------------

REAL = os.environ.get("TEXT_ENCODER_DIR")


@pytest.mark.skipif(not REAL, reason="TEXT_ENCODER_DIR is not set (no exported text tower)")
def test_real_encoder() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("sentencepiece")
    encoder = te.OnnxTextEncoder(Path(str(REAL)))
    assert encoder.name == te.MODEL_NAME and encoder.dim == 768
    spool, cable, pumpkin = (
        encoder.embed(te.prompt_for(q)) for q in ("spool", "cable spool", "pumpkin")
    )
    assert np.linalg.norm(spool) == pytest.approx(1, abs=1e-5)
    assert float(spool @ cable) > float(spool @ pumpkin)
