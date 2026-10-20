"""Which kind of capture this is, from what its frames say about themselves.

One vocabulary with the API's `EpochPlatform` (decision D4: the survey tiers), checked
equal by `apps/api/tests/test_worker_params.py`:

* `rtk-drone` -- a drone whose RTK solution was fixed, with a stated accuracy, on every
  frame (`rtk_evidence`). The label alone never makes a capture surveyed: registration
  decides that from its own alignment, and only for an `rtk-drone` whose
  `platformEvidence` is `metadata`;
* `drone` -- a drone with consumer GNSS (or an RTK module that was not fixed);
* `phone-video` / `phone-photo` -- the coarse tier;
* `unknown` -- detection ran and found nothing it recognises. That is the honest
  fallback for a camera this module has never heard of, not an error.

The rules, first match wins (`detect`):

1. the recipe or run asked (`platform` param): it must be one of the five. What the
   frames show on their own is used when it is the same platform; otherwise the request
   is recorded as `claimed`;
2. photos, every one with DJI `RtkFlag` 50 (fixed) and a stated accuracy inside a
   carrier wavelength (`rtk_evidence`): `rtk-drone`;
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
not a tolerance on a measurement. "Every frame" for the survey tier is not one either: a
capture is survey evidence only where each frame's position is, and a frame that lost its
fix is one that is not; such a capture is a `drone`, and its fixed frames still carry
their own RTK accuracy in `frames_meta.json` for registration to weigh.

Every platform says what it rests on, `platformEvidence` (`PLATFORM_EVIDENCE`):

* `metadata` -- the frames' own metadata showed it (DJI XMP, EXIF make, QuickTime or
  Android keys);
* `claimed` -- only a request (`platform` param, from a capture's `metadata.platform`,
  which any write-token client may set) or the upload route (`platform_hint`) said so;
* `none` -- nothing matched (`unknown`).

It travels with the value: `source_meta.platform`, `frames_meta.platform`, the manifest's
`capture.platformEvidence` and `registration.json`'s `platformEvidence`. A claimed
`rtk-drone` is never the survey tier: the worker records it on the epoch as a `drone`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from frame_meta import RTK_FIXED, FrameRecord
from video import VideoMeta

__all__ = [
    "PHONE_MAKES",
    "PLATFORMS",
    "PLATFORM_EVIDENCE",
    "RTK_FIXED_STD_MAX_M",
    "Platform",
    "detect",
    "rtk_evidence",
]

PLATFORMS: tuple[str, ...] = ("rtk-drone", "drone", "phone-video", "phone-photo", "unknown")
#: What a platform rests on (the module's `platformEvidence`).
PLATFORM_EVIDENCE: tuple[str, ...] = ("metadata", "claimed", "none")

#: The largest stated RTK standard deviation a fixed solution can have, in metres: one
#: wavelength of the GPS L1 carrier, c / 1575.42 MHz. A fixed solution has resolved the
#: whole carrier cycles, so its error is a fraction of one; a frame that states a whole
#: cycle or more is not fixed whatever its flag says. Physics, not a tuned threshold.
RTK_FIXED_STD_MAX_M = 299_792_458.0 / 1_575_420_000.0

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
    #: What it rests on: `metadata`, `claimed` or `none` (`PLATFORM_EVIDENCE`).
    platform_evidence: str = "metadata"

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "source": self.source,
            "evidence": self.evidence,
            "platformEvidence": self.platform_evidence,
        }


def _at_least_half(count: int, total: int) -> bool:
    return total > 0 and 2 * count >= total


def _stated(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0 < value < RTK_FIXED_STD_MAX_M


def rtk_evidence(record: FrameRecord) -> bool:
    """True when the frame's position is survey evidence: DJI `RtkFlag` 50 (fixed) and a
    stated standard deviation in longitude, latitude and height, each inside one carrier
    wavelength (`RTK_FIXED_STD_MAX_M`)."""
    dji = record.dji
    return (
        dji is not None
        and dji.rtk_flag == RTK_FIXED
        and _stated(dji.rtk_std_lon_m)
        and _stated(dji.rtk_std_lat_m)
        and _stated(dji.rtk_std_hgt_m)
    )


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
        shown = _detected(fmt=fmt, records=records, video=video, hint=None)
        if shown.value == requested and shown.platform_evidence == "metadata":
            return shown
        return Platform(requested, "param", "asked for by the recipe or the run", "claimed")
    return _detected(fmt=fmt, records=records, video=video, hint=hint)


def _detected(
    *, fmt: str, records: Sequence[FrameRecord], video: VideoMeta | None, hint: str | None
) -> Platform:
    total = len(records)
    if fmt != "video" and total:
        djis = [r.dji for r in records if r.dji is not None]
        fixed = sum(1 for d in djis if d.rtk_flag == RTK_FIXED)
        flagged = sum(1 for d in djis if d.rtk_flag is not None)
        evidenced = sum(1 for r in records if rtk_evidence(r))
        if evidenced == total:
            return Platform(
                "rtk-drone",
                "dji-rtk-fixed",
                f"{total}/{total} frames RtkFlag 50 (fixed) with a stated accuracy",
            )
        flying = sum(1 for d in djis if d.is_flight)
        if _at_least_half(flying, total):
            evidence = f"{flying}/{total} frames carry DJI flight XMP"
            if flagged:
                evidence += (
                    f"; RTK module present, {fixed}/{total} fixed, {evidenced}/{total} fixed "
                    "with a stated accuracy: consumer tier"
                )
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
        return Platform(value, "phone-key", "uploaded with a phone key", "claimed")
    return Platform("unknown", "none", "no rule matched what the frames say", "none")
