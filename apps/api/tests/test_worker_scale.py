"""A scale the pipeline estimated, on its way into the catalog: kept as an estimate.

`exif_gps` sizes a phone video with no GPS by how high a handheld phone is, and writes
`scaleSource: camera-height-estimate` with its ±% in `frame.scaleEstimate`. Two things
can go wrong on the way in, and these hold both: the source folded into `unresolved`
(the capture is drawn the size it now is while the record says it has no size), or the
estimate reaching the inspector without the figure that says how rough it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import Capture, Job, Site
from app.models.enums import CaptureKind, Representation, ScaleSource
from app.schemas.common import Provenance
from app.storage import S3Storage
from app.worker import registration


def _georef(scale_source: str, **frame: Any) -> dict[str, Any]:
    return {
        "lat": 37.8,
        "lon": -122.4,
        "height": 12.5,
        "georefMethod": "exif-gps",
        "scaleSource": scale_source,
        "uncertaintyM": 10.0,
        "frame": {"source": "camera-up", "scale": 0.83, **frame},
    }


ESTIMATE = {"method": "camera-height", "priorM": 1.5, "cameras": 179, "uncertaintyPct": 22.4}


def _read(tmp_path: Path, georef: dict[str, Any]) -> registration.Registration:
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({"slug": "spool", "georef": georef}))
    return registration.Registration.read(path)


def test_an_estimated_scale_is_read_as_one_with_its_uncertainty(tmp_path: Path) -> None:
    read = _read(tmp_path, _georef("camera-height-estimate", scaleEstimate=ESTIMATE))

    assert read.scale_source is ScaleSource.CAMERA_HEIGHT_ESTIMATE
    assert read.scale_uncertainty_pct == 22.4


def test_only_an_estimate_carries_a_scale_uncertainty(tmp_path: Path) -> None:
    """A measured scale has no ±% of this kind, an unknown source is `unresolved` with no
    scale to describe, and a figure that is not a number is no figure."""
    measured = _read(tmp_path, _georef("exif-gps", scaleEstimate=ESTIMATE))
    unknown = _read(tmp_path, _georef("source", scaleEstimate=ESTIMATE))
    garbled = _read(
        tmp_path, _georef("camera-height-estimate", scaleEstimate={"uncertaintyPct": "lots"})
    )
    bare = _read(tmp_path, _georef("camera-height-estimate"))

    assert (measured.scale_source, measured.scale_uncertainty_pct) == (ScaleSource.EXIF_GPS, None)
    assert (unknown.scale_source, unknown.scale_uncertainty_pct) == (ScaleSource.UNRESOLVED, None)
    assert garbled.scale_source is ScaleSource.CAMERA_HEIGHT_ESTIMATE
    assert garbled.scale_uncertainty_pct is None
    assert bare.scale_uncertainty_pct is None


def test_registering_records_the_estimate_on_the_capture_and_in_its_provenance(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    capture = Capture(slug="spool", name="Spool", kind=CaptureKind.VIDEO, metadata_={})
    db.add(capture)
    db.flush()
    job = Job(capture_id=capture.id, recipe="photo-reconstruct", recipe_version="10", params={})
    db.add(job)
    db.commit()
    storage.put_object(f"runs/{job.id}/package/splat/tileset.json", b"{}", "application/json")

    registration.register(
        db,
        storage,
        capture=capture,
        job_id=job.id,
        registration=_read(tmp_path, _georef("camera-height-estimate", scaleEstimate=ESTIMATE)),
        tiles_stage_id="package",
    )

    db.refresh(capture)
    assert capture.scale_source is ScaleSource.CAMERA_HEIGHT_ESTIMATE
    assert capture.site_id is not None
    site = db.get(Site, capture.site_id)
    assert site is not None
    splat = next(a for a in site.assets if a.representation == Representation.GAUSSIAN_SPLAT)
    provenance = splat.render_config["provenance"]
    assert provenance["scaleSource"] == "camera-height-estimate"
    assert provenance["scaleUncertaintyPct"] == 22.4
    # ...and the API reads back what the worker wrote, as the inspector will receive it.
    read_back = Provenance.model_validate(provenance)
    assert read_back.scale_source is ScaleSource.CAMERA_HEIGHT_ESTIMATE
    assert read_back.scale_uncertainty_pct == 22.4
