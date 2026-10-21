"""The video's own location, recorded on the capture before its run is placed.

`services/epochs.site_for_run` matches a capture to an existing site before its run
starts, because the pipeline places the capture in that site's frame. The point it
matches by was the one sent with the upload -- the browser's fix -- which is where the
video was *uploaded* from: a clip recorded on the ranch and sent from home joined
nothing, or the wrong site. The video says where it was made, in its container's
location atoms (ADR 0013, "Joining a site by the video's GPS"), and this is where the
worker reads them: the capture's video, in the bucket, by a few ranged GETs
(`services/video_location.py`), before the frame is chosen.

The answer is kept on the capture as `metadata.videoLocation` -- `{lat, lon, accuracyM,
key}`, or null when the video carries none -- so every later reader of the join agrees
with the first: the site context seed, a rejected run's epoch, the registration that
writes the epoch (which records the source), and a resumed or retried run on another
worker. It is read once: the uploaded bytes do not change. Only the worker writes it
(`schemas/capture.CaptureCreate` refuses it), so it is never a claim the request made.

A storage failure is recorded as no location, and the run goes on matched by the
upload's fix, as before: a capture is never failed for want of a better guess at its
site. Recorded rather than left for a later read, because the frame is chosen again when
a stage is retried (`runner._override`), and a read that succeeded there would move the
run's remaining stages into another site's frame halfway.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import type_coerce, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from app.models import Capture, CaptureFile
from app.models.enums import UploadStatus
from app.services import epochs
from app.services.video_location import (
    QUICKTIME_SUFFIXES,
    VIDEO_SUFFIXES,
    location_in_storage,
)
from app.storage import ObjectStorage

log = logging.getLogger("app.worker")


def video_of(capture: Capture) -> CaptureFile | None:
    """The uploaded video the pipeline will take (`tools/pipeline/video.pick_source`: the
    first video file by name), or None for a capture without one."""
    videos = sorted(
        (
            source
            for source in capture.files
            if source.status is UploadStatus.COMPLETE
            and Path(source.filename).suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda source: Path(source.filename).name,
    )
    return videos[0] if videos else None


def record_video_location(db: Session, storage: ObjectStorage, capture: Capture) -> None:
    """Read the capture's video's location into `metadata.videoLocation`, once.

    Nothing for a capture that has a record already, or has no video. The transaction is
    ended before the bucket is read (as `runner._seed` does), and the record is merged
    into the metadata in one statement, so nothing else in it is overwritten."""
    if epochs.VIDEO_LOCATION in (capture.metadata_ or {}):
        return
    video = video_of(capture)
    if video is None:
        return
    key, name = video.storage_key, Path(video.filename).name
    if db.in_transaction():
        db.commit()
    failed = False
    if Path(name).suffix.lower() not in QUICKTIME_SUFFIXES:
        found = None  # Matroska, WebM, AVI: no QuickTime location atoms to read
    else:
        try:
            found = location_in_storage(storage, key)
        except Exception as error:  # any failure means "matched by the upload"
            log.warning(
                "capture %s: could not read the location in %s (%s); recorded as none, "
                "so it is matched to a site by the location sent with its upload",
                capture.id,
                name,
                error,
            )
            found, failed = None, True
    record = {epochs.VIDEO_LOCATION: found.to_metadata() if found is not None else None}
    db.execute(
        update(Capture)
        .where(Capture.id == capture.id)
        .values(metadata_=Capture.metadata_.op("||")(type_coerce(record, JSONB)))
    )
    db.commit()
    db.refresh(capture)
    if failed:
        return
    if found is None:
        log.info(
            "capture %s: %s carries no location; it is matched to a site by the location "
            "sent with its upload",
            capture.id,
            name,
        )
    else:
        log.info(
            "capture %s: %s says it was made at (%.6f, %.6f), +-%s m (%s)",
            capture.id,
            name,
            found.lat,
            found.lon,
            "?" if found.accuracy_m is None else f"{found.accuracy_m:g}",
            found.key,
        )
