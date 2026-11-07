"""Extraction changes invalidate the Preview's frame of reference; no database needed."""

from app.api.v1.phone import _checked_options, _refine_from


def test_preview_limits_are_allowed_and_extraction_change_restarts_normalize() -> None:
    preview = {
        "max_side": 1600,
        "fps": 4,
        "keep": 100,
        "keep_video": 180,
        "max_candidates": 1200,
        "select": "viewpoint",
    }
    assert _checked_options("photo-reconstruct", {"normalize": preview}) == {"normalize": preview}
    assert _checked_options("photo-reconstruct", {"normalize": {}}) == {"normalize": {}}
    previous = {"normalize": preview}
    assert _refine_from(previous, {"normalize": {}}) == "normalize"
    assert _refine_from(previous, {"normalize": {"max_side": 2400}}) == "normalize"
    assert _refine_from(previous, {"normalize": preview}) == "train"
    assert _refine_from(previous, {}) == "train"
