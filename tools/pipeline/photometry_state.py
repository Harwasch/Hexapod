"""Retain the small PPISP checkpoint state independently of disposable splat checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save(runner: Any, checkpoint: dict[str, Any]) -> None:
    """Called at the checkpoint seam, before the same step's PLY export.

    JSON preserves tensor dtype/shape and values, but contains no splats or optimizer
    moments. It is sufficient to reconstruct post_processing with load_state_dict.
    Frame indices are Dataset indices (not positions in the frozen test split).
    """
    if runner.cfg.post_processing != "ppisp" or "post_processing" not in checkpoint:
        return
    tensors = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in checkpoint["post_processing"].items()
    }
    frames = [runner.parser.image_names[int(i)] for i in runner.trainset.indices]
    document = {
        "schema": "post-processing-state/1",
        "kind": "ppisp",
        "checkpointStep": int(checkpoint["step"]),
        "stepConvention": "zero-based, same state as checkpoint and pre-update PLY",
        "displayReferred": bool(getattr(runner, "hex_settings", {}).get("ppisp_display_referred")),
        "useController": bool(runner.cfg.ppisp_use_controller),
        "trainFrames": frames,
        "stateDict": tensors,
    }
    target = Path(runner.cfg.result_dir) / "post_processing.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, allow_nan=False) + "\n")
    temporary.replace(target)
