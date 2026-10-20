"""An epoch's alignment is recorded, and its provenance follows it (M3b.2, decision D4).

`registration.register` the way the worker calls it, with `registration.json` documents
carrying the `alignment` block `catalog` copies out of `epoch_alignment.json`. What is
checked is what lands in the database and what `GET /sites/{id}/epochs` then says.
"""

from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Capture, Epoch
from app.models.enums import CaptureKind, GeorefMethod, ScaleSource
from app.services import epochs
from app.storage import S3Storage
from tests.test_registration_epochs import _assets, _capture, _echoed, _epochs, run


def aligned(
    *,
    method: str,
    verdict: str,
    platform: str,
    uncertainty: float,
    label: str,
    reference: uuid.UUID | None = None,
) -> dict[str, Any]:
    """The part of `epoch_alignment.json` (design section 4.4) the API reads or keeps."""
    return {
        "version": 1,
        "method": method,
        "referenceEpochId": None if reference is None else str(reference),
        "referenceJobId": None,
        "platform": platform,
        "platformSource": "frames_meta",
        "tier": "survey" if platform == "rtk-drone" else "coarse",
        # What align_epoch (#51) reads survey from: an RTK fix in the frames' metadata.
        "rtkEvidence": "30/30 frames carry an RTK fixed solution (RtkFlag 50)"
        if platform == "rtk-drone"
        else None,
        "verdict": verdict,
        "confidence": 0.9 if verdict != "unaligned" else 0.0,
        "uncertaintyM": uncertainty,
        "label": label,
        "reasons": [],
        "residualM": {"rms": 0.01, "median": 0.01, "p90": 0.02, "max": 0.03},
        "frameSource": "epoch-alignment" if method == "visual" else "georef",
    }


def second_epoch(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    alignment_for: Any,
    *,
    first_scale: str = "exif-gps",
    first_pct: float | None = None,
) -> tuple[Capture, Capture]:
    """Epoch 1 (EXIF-aligned unless told otherwise), then a second capture of its site
    registered with `alignment_for(epoch 1's id)`."""
    first = _capture(db, "first")
    document = _echoed_free(scale=first_scale, pct=first_pct)
    run(db, storage, tmp_path, first, document)
    assert first.site_id is not None
    one = epochs.epoch_of_capture(db, first.id)
    assert one is not None
    second = _capture(db, "second", metadata_={"siteId": str(first.site_id)})
    document = _echoed(first.site_id, scaleSource="exif-gps", uncertaintyM=5.0)
    document["alignment"] = alignment_for(one.id)
    run(db, storage, tmp_path, second, document)
    db.expire_all()
    return first, second


def _echoed_free(*, scale: str, pct: float | None) -> dict[str, Any]:
    from tests.test_asset_scale import _document

    document = _document()
    document["georef"]["scaleSource"] = scale
    if pct is not None:
        document["georef"]["frame"]["scaleEstimate"] = {"uncertaintyPct": pct}
    return document


def _epoch_of(db: Session, capture: Capture) -> Epoch:
    epoch = epochs.epoch_of_capture(db, capture.id)
    assert epoch is not None
    return epoch


def _shown(client: TestClient, site_id: uuid.UUID) -> list[dict[str, Any]]:
    response = client.get(f"/api/v1/sites/{site_id}/epochs")
    assert response.status_code == 200, response.text
    return [
        {k: (e["alignment"] or {}).get(k) for k in ("verdict", "label", "uncertaintyM")}
        for e in response.json()
    ]


def test_a_phone_epoch_aligned_visually_is_coarse_and_placed_by_its_reference(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    label = "Phone video, aligned visually, ±0.065 m"
    first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: aligned(method="visual", verdict="coarse", platform="phone-video",
                            uncertainty=0.065, label=label, reference=ref),
    )  # fmt: skip

    epoch = _epoch_of(db, second)
    assert epoch.alignment is not None and epoch.alignment["label"] == label
    assert epoch.platform == "phone-video"
    assert (second.georef_method, second.scale_source, second.uncertainty_m) == (
        GeorefMethod.VISUAL_EPOCH,
        ScaleSource.VISUAL_EPOCH,
        0.065,
    )
    assert first.site_id is not None
    asset = next(a for a in _assets(db, first.site_id) if a.epoch_id == epoch.id)
    provenance = asset.render_config["provenance"]
    assert {k: provenance[k] for k in ("georefMethod", "scaleSource", "uncertaintyM")} == {
        "georefMethod": "visual-epoch",
        "scaleSource": "visual-epoch",
        "uncertaintyM": 0.065,
    }
    assert _shown(client, first.site_id)[1] == {
        "verdict": "coarse",
        "label": label,
        "uncertaintyM": 0.065,
    }


def test_an_rtk_epoch_surveyed_against_its_targets_says_so(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    label = "RTK drone, surveyed, check targets ±0.024 m"
    first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: aligned(method="visual", verdict="survey", platform="rtk-drone",
                            uncertainty=0.024, label=label, reference=ref),
    )  # fmt: skip
    epoch = _epoch_of(db, second)
    # The platform the verdict was judged for, though the run's manifest said nothing.
    assert epoch.platform == "rtk-drone"
    assert second.georef_method is GeorefMethod.VISUAL_EPOCH
    assert first.site_id is not None
    assert _shown(client, first.site_id)[1]["label"] == label


def test_an_unaligned_epoch_keeps_its_gnss_provenance(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    """D4: an epoch that fails alignment is kept with its GNSS placement and labelled
    so -- never shown as aligned, and never `visual-epoch`."""
    label = "Drone, not aligned: placed by its GPS, ±5 m"
    first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: aligned(method="gnss-similarity", verdict="unaligned", platform="drone",
                            uncertainty=5.0, label=label, reference=ref),
    )  # fmt: skip
    assert (second.georef_method, second.scale_source, second.uncertainty_m) == (
        GeorefMethod.EXIF_GPS,
        ScaleSource.EXIF_GPS,
        5.0,
    )
    assert first.site_id is not None
    assert _shown(client, first.site_id)[1] == {
        "verdict": "unaligned",
        "label": label,
        "uncertaintyM": 5.0,
    }


def test_an_rtk_epoch_that_kept_its_own_frame_keeps_its_gnss_provenance(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The reference was coarser than the RTK epoch, so `align_epoch` kept its own
    `gnss-prior-ba` frame (the visual fit is only a check): not `visual-epoch`."""
    _first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: {**aligned(method="gnss-prior-ba", verdict="coarse", platform="rtk-drone",
                               uncertainty=0.035, label="RTK drone, aligned to its GNSS, ±0.035 m",
                               reference=ref), "visualCheck": {"method": "visual"}},
    )  # fmt: skip
    assert second.georef_method is GeorefMethod.EXIF_GPS
    assert second.uncertainty_m == 0.035


def test_an_epoch_aligned_to_an_estimated_scale_inherits_the_estimate(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """Aligned to an epoch whose scale was estimated from the camera's height, the new
    epoch's scale is that estimate, carried over -- not a metric scale."""
    _first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: aligned(method="visual", verdict="coarse", platform="phone-video",
                            uncertainty=0.5, label="Phone video, aligned visually, ±0.5 m",
                            reference=ref),
        first_scale="camera-height-estimate", first_pct=22.4,
    )  # fmt: skip
    assert second.georef_method is GeorefMethod.VISUAL_EPOCH
    assert second.scale_source is ScaleSource.CAMERA_HEIGHT_ESTIMATE
    epoch = _epoch_of(db, second)
    asset = next(a for a in _assets(db, epoch.site_id) if a.epoch_id == epoch.id)
    assert asset.render_config["provenance"]["scaleUncertaintyPct"] == 22.4


def test_a_splat_upload_is_recorded_unaligned_with_why(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = _capture(db, "splat")
    capture.kind = CaptureKind.GAUSSIAN_SPLAT
    db.commit()
    from tests.test_asset_scale import _document

    document = _document()
    document["georef"].update(georefMethod="manual", uncertaintyM=10.0)
    run(db, storage, tmp_path, capture, document)
    assert capture.site_id is not None
    [epoch] = _epochs(db, capture.site_id)
    assert epoch.alignment is not None
    assert {k: epoch.alignment[k] for k in ("method", "verdict", "uncertaintyM", "label")} == {
        "method": "none",
        "verdict": "unaligned",
        "uncertaintyM": 10.0,
        "label": "Splat upload, not aligned: placed by hand, ±10 m",
    }
    assert epoch.alignment["reasons"] == ["a splat upload has no frames to match"]


def test_a_re_run_records_its_own_alignment(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: aligned(method="gnss-similarity", verdict="unaligned", platform="drone",
                            uncertainty=5.0, label="Drone, not aligned: placed by its GPS, ±5 m",
                            reference=ref),
    )  # fmt: skip
    assert first.site_id is not None
    one = _epoch_of(db, first)
    document = _echoed(first.site_id, scaleSource="exif-gps")
    document["alignment"] = aligned(
        method="visual", verdict="coarse", platform="drone", uncertainty=0.08,
        label="Drone, aligned visually, ±0.08 m", reference=one.id,
    )  # fmt: skip
    before = copy.deepcopy(_epoch_of(db, second).alignment)
    run(db, storage, tmp_path, second, copy.deepcopy(document))
    db.expire_all()
    epoch = _epoch_of(db, second)
    assert epoch.alignment is not None and epoch.alignment["verdict"] == "coarse"
    assert second.georef_method is GeorefMethod.VISUAL_EPOCH
    # R2 review of #52: what it replaced -- the earlier run's verdict, reference and
    # reasons -- is kept in the epoch's history, never lost to the re-run.
    assert before is not None and before["verdict"] == "unaligned"
    assert epoch.history[-1]["alignment"] == before


def test_the_alignment_block_is_read_defensively(tmp_path: Path) -> None:
    from tests.test_asset_scale import _document, _registration

    document = _document()
    document["alignment"] = "not an object"
    assert _registration(tmp_path, document).alignment is None
    document["alignment"] = {"verdict": "coarse"}
    assert _registration(tmp_path, document).alignment == {"verdict": "coarse"}


def test_survey_is_never_recorded_for_anything_but_an_rtk_drone(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    """D6 amendment, held at the API too: an alignment that claimed survey for a phone
    (align_epoch never writes one) is recorded coarse, in align_epoch's words, with why."""
    first, second = second_epoch(
        db, storage, tmp_path,
        lambda ref: aligned(method="visual", verdict="survey", platform="phone-video",
                            uncertainty=0.065, label="Phone video, surveyed, check targets ±0.065 m",
                            reference=ref),
    )  # fmt: skip
    assert first.site_id is not None
    assert _shown(client, first.site_id)[1] == {
        "verdict": "coarse",
        "label": "Phone video, aligned visually, ±0.065 m",
        "uncertaintyM": 0.065,
    }
    epoch = _epoch_of(db, second)
    assert epoch.alignment is not None and "only an RTK drone" in epoch.alignment["reasons"][-1]


def test_survey_is_never_recorded_without_rtk_evidence_in_the_data(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    """Coordinator (R2 review of #51/#52): the platform string never makes survey on its
    own. An RTK-drone alignment that claims survey with no `rtkEvidence` (align_epoch
    never writes one) is recorded coarse, with why, at the API too."""

    def claimed(ref: uuid.UUID) -> dict[str, Any]:
        block = aligned(method="visual", verdict="survey", platform="rtk-drone",
                        uncertainty=0.04, label="RTK drone, surveyed, check targets ±0.04 m",
                        reference=ref)  # fmt: skip
        block["rtkEvidence"] = None
        return block

    first, second = second_epoch(db, storage, tmp_path, claimed)
    assert first.site_id is not None
    assert _shown(client, first.site_id)[1] == {
        "verdict": "coarse",
        "label": "RTK drone, aligned visually, ±0.04 m",
        "uncertaintyM": 0.04,
    }
    epoch = _epoch_of(db, second)
    assert epoch.alignment is not None and "no RTK evidence" in epoch.alignment["reasons"][-1]
    assert epochs.claimed_verdict(epoch.alignment, "rtk-drone") == "coarse"
    assert epochs.claimed_verdict({"verdict": "survey"}, "rtk-drone") == "coarse"
