"""Artifact integrity and offline readiness without model weights or GPU packages."""

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ltx25_bootstrap", Path(__file__).parents[1] / "ltx25" / "bootstrap.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def fixture(tmp_path):
    source = tmp_path / "source"
    weights = tmp_path / "weights"
    source.mkdir()
    weights.mkdir()
    module = source / "packages/ltx-core/src/ltx_core/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("PINNED = True\n")
    header = json.dumps(
        {
            "__metadata__": {"gemma_config": "{}"},
            "tokenizer_json": {},
            "hf_asset__tokenizer_config.json": {},
            "hf_asset__processor_config.json": {},
        }
    ).encode()
    packed = weights / "te.safetensors"
    packed.write_bytes(struct.pack("<Q", len(header)) + header)
    manifest = {
        "commit": "source-pin",
        "checkpointRevision": "checkpoint-pin",
        "sourceFiles": {
            module.relative_to(source).as_posix(): bootstrap.digest(module)
        },
        "components": {"text_encoder": "te.safetensors"},
        "files": [
            {
                "path": packed.name,
                "size": packed.stat().st_size,
                "sha256": bootstrap.digest(packed),
            }
        ],
    }
    return source, weights, manifest


def test_ready_requires_explicit_full_hash_verification_and_detects_replacement(
    tmp_path,
):
    source, weights, manifest = fixture(tmp_path)
    assert not bootstrap.check_artifacts(source, weights, manifest)["ready"]
    bootstrap.verify_assets(weights, manifest)
    assert bootstrap.check_artifacts(source, weights, manifest)["ready"]
    path = weights / "te.safetensors"
    path.write_bytes(path.read_bytes() + b"changed")
    assert not bootstrap.check_artifacts(source, weights, manifest)["ready"]
    with pytest.raises(ValueError, match="mismatch"):
        bootstrap.verify_assets(weights, manifest)


def test_verification_rejects_wrong_hash_even_with_matching_size(tmp_path):
    _, weights, manifest = fixture(tmp_path)
    manifest["files"][0]["sha256"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="mismatch"):
        bootstrap.verify_assets(weights, manifest)
    assert not (weights / bootstrap.RECEIPT).exists()


def test_readiness_rejects_modified_or_shadowing_source(tmp_path):
    source, weights, manifest = fixture(tmp_path)
    bootstrap.verify_assets(weights, manifest)
    path = source / next(iter(manifest["sourceFiles"]))
    path.with_name("extra.py").write_text("UNREVIEWED = True\n")
    assert not bootstrap.check_artifacts(source, weights, manifest)["ready"]
    path.with_name("extra.py").unlink()
    path.write_text("PINNED = False\n")
    assert not bootstrap.check_artifacts(source, weights, manifest)["ready"]


def test_packed_encoder_requires_offline_tokenizer_and_processor(tmp_path):
    path = tmp_path / "encoder.safetensors"
    header = json.dumps(
        {"__metadata__": {"gemma_config": "{}"}, "tokenizer_json": {}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="tokenizer_config"):
        bootstrap.verify_packed_text_encoder(path)
    path.write_bytes(struct.pack("<Q", 100_000_000))
    with pytest.raises(ValueError, match="bound"):
        bootstrap.verify_packed_text_encoder(path)


def test_bad_receipt_is_unavailable_not_startup_exception(tmp_path):
    source, weights, manifest = fixture(tmp_path)
    (weights / bootstrap.RECEIPT).write_text("[]")
    assert not bootstrap.check_artifacts(source, weights, manifest)["ready"]
