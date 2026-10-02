"""Search by meaning: a query's text embedding, in the space the scan's instances were embedded in.

Segmentation describes each instance with SigLIP 2's *image* tower
(`tools/captures/segment_models.SiglipEmbedder`, `instances.emb`); a query is only
comparable with those rows if it goes through the *same* model's text tower with the same
prompt (`TAG_TEMPLATE`) and the same tokenisation. This runs that tower on the API's CPU,
with onnxruntime and sentencepiece, from what `tools/captures/export_text_encoder.py`
writes (the image's `text-encoder` build stage; `TEXT_ENCODER_DIR`).

Why here, measured 2026-10-02 (docs/SCENE_OBJECTS.md §3 step 4 has the table):

* not in the browser -- the smallest published text tower is the 283 MB int8 ONNX (plus a
  34 MB tokenizer), and it is not the same model for ranking (cosine 0.92-0.98 to float32;
  top-10 overlap down to 3/10 on a real scan);
* not on Modal -- a container cold start (image pull, torch import, weights) is seconds
  to tens of seconds per idle period, on top of the Fly machine's own start;
* here: the first request 1.3 s (load included), then ~140 ms a query on one core; about
  +500 MB resident (onnxruntime's float32 weights and their prepacked copies; the 393 MB
  token table is memory-mapped and a query touches <= 64 rows of it) -- ~640 MB for the
  whole app process, inside the app machine's 1 GB. Cosine to the transformers reference
  1.000000 (min 0.9999998), the same top-10 on every query and scan tried.

The model loads on the first request, not at start-up, so a machine that never serves a
search never pays for it. Embeddings of the same prompt are cached.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

import numpy as np

#: `segment_models.SIGLIP_MODEL` / `TAG_TEMPLATE` (tests/test_text_embedding.py holds them
#: equal). The template is the one the tags were scored with, so a query reads like a tag.
MODEL = "google/siglip2-base-patch16-224"
#: How `instances.json`'s `embedding.model` names it (`SiglipEmbedder.name`).
MODEL_NAME = f"siglip:{MODEL}"
TAG_TEMPLATE = "a photo of a {}."
MAX_QUERY_CHARS = 200

_ARTICLE = re.compile(r"^(?:a photo of )?(?:(?:a|an|the)(?:\s+|$))?", re.IGNORECASE)


class TextEncoderUnavailableError(RuntimeError):
    """No encoder here (not installed, not configured, or failed to load): answer 503."""


def query_label(text: str) -> str:
    """What a typed query names: lower-cased, whitespace collapsed, a trailing full stop and
    a leading article (or "a photo of a") dropped, since the template supplies them."""
    label = " ".join(text.lower().split()).rstrip(" .")
    label = _ARTICLE.sub("", label, count=1).strip()
    return label[:MAX_QUERY_CHARS]


def prompt_for(text: str) -> str:
    """The prompt a query is embedded as: `TAG_TEMPLATE` around `query_label`."""
    return TAG_TEMPLATE.format(query_label(text))


class TextEncoder(Protocol):
    name: str
    dim: int

    def embed(self, prompt: str) -> np.ndarray:
        """(dim,) float32, L2-normalised."""
        ...


def token_ids(pieces: Any, text: str, max_length: int, eos: int, pad: int) -> list[int]:
    """Lower-cased sentencepiece pieces, then `<eos>`, truncated and padded to `max_length`:
    `AutoProcessor(text=..., padding="max_length", max_length=64, truncation=True)`."""
    ids = [*(int(i) for i in pieces.encode(text.lower()))][: max_length - 1]
    ids.append(eos)
    return ids + [pad] * (max_length - len(ids))


class OnnxTextEncoder:
    """The exported tower: sentencepiece ids -> fp16 table rows -> ONNX transformer."""

    def __init__(self, directory: Path, threads: int = 1) -> None:
        import onnxruntime as ort
        import sentencepiece as spm

        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format") != "hexapod.text-encoder" or manifest.get("model") != MODEL:
            raise TextEncoderUnavailableError(f"{directory} is not a {MODEL} text encoder")
        if manifest.get("template") != TAG_TEMPLATE:
            raise TextEncoderUnavailableError("the encoder was exported for another template")
        files = manifest["files"]
        self.name: str = manifest["name"]
        self.dim: int = int(manifest["dim"])
        self._max_length = int(manifest["maxLength"])
        self._eos = int(manifest["eosId"])
        self._pad = int(manifest["padId"])
        self._pieces = spm.SentencePieceProcessor(model_file=str(directory / files["tokenizer"]))
        self._table = np.load(directory / files["table"], mmap_mode="r")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(directory / files["tower"]), options, providers=["CPUExecutionProvider"]
        )

    def embed(self, prompt: str) -> np.ndarray:
        ids = token_ids(self._pieces, prompt, self._max_length, self._eos, self._pad)
        rows = np.asarray(self._table[np.array([ids], np.int64)], np.float32)
        out = np.asarray(self._session.run(None, {"token_embeds": rows})[0][0], np.float32)
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 0 else out


class LazyTextEncoder:
    """Loads the encoder on first use (once, under a lock) and caches recent prompts.

    A load that fails is remembered, so a machine without the files answers 503 at once
    rather than retrying a 2 s load on every keystroke."""

    def __init__(
        self,
        directory: Path | None,
        threads: int = 1,
        loader: Any = OnnxTextEncoder,
        cache_size: int = 512,
    ) -> None:
        self._directory = directory
        self._threads = threads
        self._loader = loader
        self._lock = threading.Lock()
        self._encoder: TextEncoder | None = None
        self._failure: str | None = None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def get(self) -> TextEncoder:
        if self._encoder is not None:
            return self._encoder
        with self._lock:
            if self._encoder is None:
                if self._failure is None:
                    self._failure = self._load()
                if self._failure is not None:
                    raise TextEncoderUnavailableError(self._failure)
            assert self._encoder is not None
            return self._encoder

    def _load(self) -> str | None:
        if self._directory is None:
            return "search by meaning is not configured here (TEXT_ENCODER_DIR is unset)"
        if not (self._directory / "manifest.json").is_file():
            return f"no text encoder at {self._directory}"
        try:
            self._encoder = self._loader(self._directory, self._threads)
        except TextEncoderUnavailableError as error:
            return str(error)
        except Exception as error:  # an import or load failure is a 503, not a 500
            return f"the text encoder did not load: {type(error).__name__}: {error}"
        return None

    def embed(self, prompt: str) -> tuple[str, np.ndarray]:
        """(model name, embedding) for `prompt`."""
        encoder = self.get()
        cached = self._cache.get(prompt)
        if cached is None:
            cached = encoder.embed(prompt)
            with self._lock:
                self._cache[prompt] = cached
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return encoder.name, cached
