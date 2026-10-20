"""Which kind of capture this is, from what its frames say about themselves.

One vocabulary with the API's `EpochPlatform` (decision D4: the survey tiers), checked
equal by `apps/api/tests/test_worker_params.py`:

* `rtk-drone` -- a drone whose RTK solution was fixed for most frames: the survey tier;
* `drone` -- a drone with consumer GNSS (or an RTK module that was not fixed);
* `phone-video` / `phone-photo` -- the coarse tier;
* `unknown` -- detection ran and found nothing it recognises. That is the honest
  fallback for a camera this module has never heard of, not an error.

The rules, first match wins (`detect`):

1. the recipe or run asked (`platform` param): it must be one of the five;
2. photos, at least half with DJI `RtkFlag` 50 (fixed): `rtk-drone`;
3. photos, at least half with a DJI flight record (`RelativeAltitude` or
   `FlightYawDegree`): `drone`, saying how many were RTK-fixed when a module was there;
4. a video whose make or encoder says DJI: `drone` -- UNVERIFIED, there is no DJI drone
   video in the repository to check what one writes;
5. a video from an Apple make with QuickTime keys: `phone-video`; any `com.android.` key:
   `phone-video`;
6. photos, at least half from a phone make (`PHONE_MAKES`): `phone-photo`;
7. the upload came in on a phone key (`platform_hint: phone`): `phone-video` for a video,
   `phone-photo` for stills;
8. otherwise `unknown`.

"At least half" is a classification rule -- which description fits most of the frames --
not a tolerance on a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from frame_meta import RTK_FIXED, FrameRecord
from video import VideoMeta

__all__ = ["PHONE_MAKES", "PLATFORMS", "Platform", "detect"]

PLATFORMS: tuple[str, ...] = ("rtk-drone", "drone", "phone-video", "phone-photo", "unknown")

#: Camera makes that are phones, casefolded. Not exhaustive: a phone maker missing here
#: lands its photos at `unknown`, which is honest, rather than at a tier it was not shown
#: to be.
PHONE_MAKES: frozenset[str] = frozenset(
    {
        "apple",
        "google",
        "samsung",
        "xiaomi",
        "oneplus",
        "huawei",
        "oppo",
        "vivo",
        "motorola",
        "nothing",
    }
)

#: What a phone-key upload says it is (`apps/api` `worker/params.py` sends it).
PHONE_HINT = "phone"


@dataclass(frozen=True)
class Platform:
    value: str
    #: Which rule decided: `param`, `dji-rtk-fixed`, `drone-xmp`, `drone-make`,
    #: `phone-quicktime`, `phone-android`, `phone-make`, `phone-key`, `none`.
    source: str
    evidence: str

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "source": self.source, "evidence": self.evidence}


def _at_least_half(count: int, total: int) -> bool:
    return total > 0 and 2 * count >= total


def detect(
    *,
    fmt: str,
    records: Sequence[FrameRecord],
    video: VideoMeta | None = None,
    requested: str | None = None,
    hint: str | None = None,
) -> Platform:
    """The capture's platform. `fmt` is `video` or `images`; see the module for the rules."""
    if requested is not None:
        if requested not in PLATFORMS:
            raise ValueError(
                f"platform={requested!r} is not one of {', '.join(PLATFORMS)} (the survey "
                f"tiers of decision D4)"
            )
        return Platform(requested, "param", "asked for by the recipe or the run")
    total = len(records)
    if fmt != "video" and total:
        djis = [r.dji for r in records if r.dji is not None]
        fixed = sum(1 for d in djis if d.rtk_flag == RTK_FIXED)
        flagged = sum(1 for d in djis if d.rtk_flag is not None)
        if _at_least_half(fixed, total):
            return Platform(
                "rtk-drone", "dji-rtk-fixed", f"{fixed}/{total} frames RtkFlag 50 (fixed)"
            )
        flying = sum(1 for d in djis if d.is_flight)
        if _at_least_half(flying, total):
            evidence = f"{flying}/{total} frames carry DJI flight XMP"
            if flagged:
                evidence += f"; RTK module present, {fixed}/{total} fixed: consumer tier"
            return Platform("drone", "drone-xmp", evidence)
    if fmt == "video" and video is not None:
        make = (video.make or "").casefold()
        encoder = (video.tags.get("encoder") or "").casefold()
        if "dji" in make or "dji" in encoder:
            return Platform(
                "drone",
                "drone-make",
                f"video make/encoder {video.make or video.tags.get('encoder')!r} (UNVERIFIED rule)",
            )
        apple_keys = sorted(k for k in video.tags if k.startswith("com.apple.quicktime."))
        if make == "apple" and apple_keys:
            return Platform(
                "phone-video", "phone-quicktime", f"make Apple, {len(apple_keys)} QuickTime keys"
            )
        android_keys = sorted(k for k in video.tags if k.startswith("com.android."))
        if android_keys:
            return Platform(
                "phone-video", "phone-android", f"Android keys: {', '.join(android_keys)}"
            )
    if fmt != "video" and total:
        phones = [r.make for r in records if r.make and r.make.casefold() in PHONE_MAKES]
        if _at_least_half(len(phones), total):
            makes = ", ".join(sorted(set(phones)))
            return Platform("phone-photo", "phone-make", f"{len(phones)}/{total} frames by {makes}")
    if hint == PHONE_HINT:
        value = "phone-video" if fmt == "video" else "phone-photo"
        return Platform(value, "phone-key", "uploaded with a phone key")
    return Platform("unknown", "none", "no rule matched what the frames say")
