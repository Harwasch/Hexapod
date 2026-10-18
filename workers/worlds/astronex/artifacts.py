"""Small, GPU-free validation of the pinned release's on-disk layout.

These checks establish completeness, not checkpoint content hashes or successful
GPU inference. Hugging Face's pinned download supplies content provenance.
"""
from __future__ import annotations
import json
from pathlib import Path

SOURCE_FILES = (
    "inference/causal_consumer.yaml", "inference/pipelines/causal_diffusion_inference.py",
    "models/wan22_components.py", "models/wan_wrapper.py", "utils/default_config.yaml",
    "utils/checkpoint_io.py", "utils/camera_trajectory.py",
)
# Verified against the exact checkpoint's Hugging Face file listing.
CHECKPOINT_FILES = (
    "config.json", "transformer/config.json", "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors", "text_encoder/config.json",
    "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json",
    "tokenizer/spiece.model", "tokenizer/special_tokens_map.json",
)
MAX_INDEX_BYTES = 8 * 1024 * 1024


def nonempty_file(path: Path):
    return path.is_file() and path.stat().st_size > 0


def matches_revision(path: Path, revision: str):
    return path.is_file() and path.stat().st_size <= 256 and path.read_text().strip() == revision


def complete_shards(root: Path, relative_index: str):
    index = root / relative_index
    if not nonempty_file(index) or index.stat().st_size > MAX_INDEX_BYTES:
        return False
    data = json.loads(index.read_text())
    if not isinstance(data, dict):
        return False
    mapping = data.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        return False
    filenames = mapping.values()
    if any(not isinstance(raw, str) or not raw for raw in filenames):
        return False
    # A shard can hold hundreds of tensor keys; stat each file only once.
    for raw in set(mapping.values()):
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
            return False
        if not nonempty_file(index.parent / path):
            return False
    return True


def check_artifacts(source: Path, weights: Path, manifest):
    try:
        if not matches_revision(source / ".hexapod-reviewed-source", manifest["commit"]):
            return False, "Install the pinned source using astronex/bootstrap.py"
        if any(not nonempty_file(source / name) for name in SOURCE_FILES):
            return False, "The pinned inference source is incomplete"
        if not matches_revision(weights / ".hexapod-checkpoint-revision", manifest["checkpointRevision"]):
            return False, "Prepare the pinned checkpoint using astronex/bootstrap.py --weights"
        if not complete_shards(weights, "model.safetensors.index.json"):
            return False, "The pinned denoiser checkpoint shards are incomplete"
        if not complete_shards(weights, "text_encoder/model.safetensors.index.json"):
            return False, "The pinned text encoder checkpoint shards are incomplete"
        if any(not nonempty_file(weights / name) for name in CHECKPOINT_FILES):
            return False, "The checkpoint is missing a required model or tokenizer file"
    except (OSError, ValueError, UnicodeError, TypeError, KeyError):
        return False, "The pinned inference files could not be verified; check downloads and file permissions"
    return True, None
