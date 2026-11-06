"""No database: preserve geometry by default, while allowing deliberate restriction."""

from app.api.v1.phone import _checked_options
from app.schemas.capture import QUALITY_BARS


def test_phone_accepts_mark_and_explicit_preview_restriction() -> None:
    assert "mark" in QUALITY_BARS
    requested = {"quality": {"bar": "mark"}, "train": {"restrict_preview_mask": True}}
    assert _checked_options("photo-reconstruct", requested) == requested
