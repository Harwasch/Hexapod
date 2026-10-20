"""register records each revision's confidence map; an epoch says what was measured (A4).

Through `registration.register` the way the worker calls it, with the committed fixture
grids (data/tiles/synthetic-yard-epochs) as what `catalog` copied up from the tileset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset, Capture, EpochMap
from app.models.enums import CaptureKind
from app.services import epoch_maps
from app.storage import S3Storage
from app.worker import registration
from tests.confidence_fixtures import descriptor, measured, register_run, tiles_url


def _capture(db: Session, slug: str = "yard") -> Capture:
    capture = Capture(slug=slug, name=slug.title(), kind=CaptureKind.VIDEO, metadata_={})
    db.add(capture)
    db.commit()
    return capture


def _maps(db: Session) -> list[EpochMap]:
    db.expire_all()
    return list(db.scalars(select(EpochMap).order_by(EpochMap.created_at)))


def _assets(db: Session, site_id: object) -> list[Asset]:
    db.expire_all()
    return list(
        db.scalars(select(Asset).where(Asset.site_id == site_id).order_by(Asset.created_at))
    )


def test_a_measured_run_records_its_map_and_the_epoch_reads_it(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    capture = _capture(db)
    job = register_run(db, storage, tmp_path, capture, measured("e1"))

    [row] = _maps(db)
    [asset] = _assets(db, capture.site_id)
    assert (row.asset_id, row.job_id, row.kind) == (asset.id, job, "confidence")
    assert row.url == tiles_url(job).replace("tileset.json", "confidence.bin")
    assert (row.cell_m, row.level, row.format, row.version) == (1.0, 2, "hexapod.cellgrid", 1)
    assert row.descriptor == descriptor("e1")
    assert row.summary == descriptor("e1")["summary"]

    [epoch] = client.get(f"/api/v1/sites/{capture.site_id}/epochs").json()
    assert epoch["confidence"] == {
        "status": "measured",
        "assetId": str(asset.id),
        "url": row.url,
        "cellM": 1.0,
        "level": 2,
        "policy": "weak-v1",
        "weakShare": 0.5,
        "weakCells": 656,
        "scoredCells": 1304,
        "unseenCells": 8,
        "reasons": {
            "views": 324,
            "spread": 324,
            "gsd": 130,
            "accuracy": 0,
            "unseen": 8,
            "alignment": 0,
        },
        "alignmentCap": {
            "uncertaintyM": 0.0,
            "factor": 1.0,
            "priorEpochs": 0,
            "source": "site_context.json",
        },
    }
    site = client.get(f"/api/v1/sites/{capture.site_id}").json()
    assert site["epochs"][0]["confidence"] == epoch["confidence"]


def test_a_re_run_s_map_belongs_to_its_new_revision(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    """A Refine is a new asset with its own map; the superseded revision keeps the one it
    was measured with, and the epoch reads the current one's."""
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"))
    register_run(db, storage, tmp_path, capture, measured("e2"))

    old, new = _assets(db, capture.site_id)
    assert old.valid_to is not None and new.valid_to is None
    maps = {row.asset_id: row for row in _maps(db)}
    assert maps[old.id].level == 2 and maps[new.id].level == 1
    [epoch] = client.get(f"/api/v1/sites/{capture.site_id}/epochs").json()
    assert epoch["currentAssetId"] == str(new.id)
    assert epoch["confidence"]["assetId"] == str(new.id)
    assert epoch["confidence"]["cellM"] == 0.5


def test_a_run_that_measured_nothing_reads_as_not_measured(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    """A splat upload has no capture evidence: no map, and `confidence` is null. A re-run
    that measured nothing after one that did is a revision with no map of its own."""
    upload = _capture(db, "upload")
    register_run(db, storage, tmp_path, upload, measured(None))
    assert _maps(db) == []
    [epoch] = client.get(f"/api/v1/sites/{upload.site_id}/epochs").json()
    assert epoch["confidence"] is None

    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"))
    register_run(db, storage, tmp_path, capture, measured(None))
    [epoch] = client.get(f"/api/v1/sites/{capture.site_id}/epochs").json()
    assert epoch["confidence"] is None
    assert len(_maps(db)) == 1  # the superseded revision's, kept


def test_the_same_tiles_keep_their_map_and_follow_what_the_run_says(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """A republish of the very tiles the epoch shows -- a content-hashed generation, so
    the URL vouches for the bytes (#52) -- is no new revision: the map is the run's,
    unchanged when it says the same, replaced when it does not."""
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"))
    [asset] = _assets(db, capture.site_id)
    url = asset.source["url"]
    [first] = _maps(db)

    register_run(db, storage, tmp_path, capture, measured("e1"), url=url, content_addressed=True)
    [again] = _maps(db)
    assert again.id == first.id and again.descriptor == first.descriptor

    register_run(db, storage, tmp_path, capture, measured("e2"), url=url, content_addressed=True)
    [replaced] = _maps(db)
    assert replaced.id == first.id and replaced.level == 1
    assert len(_assets(db, capture.site_id)) == 1


def test_a_republish_that_carried_the_grid_carries_the_map(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The same tiles, republished by a run that measured nothing, with the grid carried
    into the new generation (`carry.py`): the new revision's map is the old one, beside
    its new URL."""
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, measured("e1"))
    register_run(db, storage, tmp_path, capture, measured(None), carried=("confidence",))

    old, new = _assets(db, capture.site_id)
    maps = {row.asset_id: row for row in _maps(db)}
    assert set(maps) == {old.id, new.id}
    assert maps[new.id].descriptor == maps[old.id].descriptor
    assert maps[new.id].url == new.source["url"].replace("tileset.json", "confidence.bin")


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"format": "other"}, "not a hexapod.cellgrid"),
        ({"uri": "../../other/confidence.bin"}, "not a file beside"),
        ({"uri": "https://evil.example.com/x.bin"}, "not a file beside"),
        ({"kind": "surface"}, "not a confidence grid"),
        ({"cellM": 2.0}, "is not level 2's cell"),
        ({"summary": {"weak": 1}}, "summary"),
        ({"alignmentCap": None}, "alignmentCap"),
        ({"padding": "x" * 70_000}, "over"),
    ],
)
def test_a_descriptor_the_api_cannot_vouch_for_is_not_recorded(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    words: str,
) -> None:
    """Logged, and the tiles registered all the same: the epoch reads as not measured."""
    monkeypatch.setattr(logging.getLogger("app.services.epoch_maps"), "disabled", False)
    capture = _capture(db)
    document = measured("e1")
    document["confidence"].update(change)
    with caplog.at_level(logging.WARNING, logger="app.services.epoch_maps"):
        register_run(db, storage, tmp_path, capture, document)
    assert _maps(db) == []
    assert len(_assets(db, capture.site_id)) == 1
    assert words in caplog.text


def test_the_descriptor_check_takes_what_the_pipeline_writes() -> None:
    for epoch in ("e1", "e2"):
        assert epoch_maps.confidence_descriptor(descriptor(epoch)) == descriptor(epoch)
    with pytest.raises(epoch_maps.DescriptorError):
        epoch_maps.confidence_descriptor(["not", "an", "object"])


def test_registration_reads_the_grid_catalog_copied_up(tmp_path: Path) -> None:
    path = tmp_path / "registration.json"
    import json

    path.write_text(json.dumps(measured("e2")))
    assert registration.Registration.read(path).confidence == descriptor("e2")
    path.write_text(json.dumps(measured(None)))
    assert registration.Registration.read(path).confidence is None
