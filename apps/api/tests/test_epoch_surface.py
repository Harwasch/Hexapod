"""register records each revision's height surface; an epoch says whether its heights were
measured (A5).

Through `registration.register` the way the worker calls it, with the committed fixture
surfaces (data/tiles/synthetic-yard-surfaces) as what `catalog` copied up from the tileset,
beside the confidence grids they sit on (synthetic-yard-epochs).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Asset, Capture, EpochMap
from app.models.enums import CaptureKind
from app.services import epoch_maps
from app.storage import S3Storage
from app.worker import registration
from tests.confidence_fixtures import measured, register_run, tiles_url
from tests.surface_fixtures import DOCUMENT, surface_descriptor
from tests.test_asset_scale import HEIGHT


def _capture(db: Session, slug: str = "yard") -> Capture:
    capture = Capture(slug=slug, name=slug.title(), kind=CaptureKind.VIDEO, metadata_={})
    db.add(capture)
    db.commit()
    return capture


def _surfaces(db: Session) -> list[EpochMap]:
    db.expire_all()
    return list(
        db.scalars(select(EpochMap).where(EpochMap.kind == "surface").order_by(EpochMap.created_at))
    )


def _assets(db: Session, site_id: object) -> list[Asset]:
    db.expire_all()
    return list(
        db.scalars(select(Asset).where(Asset.site_id == site_id).order_by(Asset.created_at))
    )


def surveyed(epoch: str | None, surface: str | None = None) -> dict[str, Any]:
    """registration.json with `epoch`'s confidence grid and `surface`'s height surface (by
    default the same epoch's; None for a run that measured no surface)."""
    document = measured(epoch)
    if surface is not None:
        document["surface"] = surface_descriptor(surface, HEIGHT)
    return document


def test_a_measured_run_records_its_surface_and_the_epoch_reads_it(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    capture = _capture(db)
    job = register_run(db, storage, tmp_path, capture, surveyed("e1", "e1"))

    [row] = _surfaces(db)
    [asset] = _assets(db, capture.site_id)
    assert (row.asset_id, row.job_id) == (asset.id, job)
    assert row.url == tiles_url(job).replace("tileset.json", "surface.bin")
    assert (row.cell_m, row.level) == (1.0, 2)
    assert row.descriptor == surface_descriptor("e1", HEIGHT)
    kinds = {m.kind for m in db.scalars(select(EpochMap).where(EpochMap.asset_id == asset.id))}
    assert kinds == {"confidence", "surface"}

    [epoch] = client.get(f"/api/v1/sites/{capture.site_id}/epochs").json()
    summary = DOCUMENT["epochs"]["e1"]["descriptor"]["summary"]
    assert epoch["surface"] == {
        "status": "measured",
        "assetId": str(asset.id),
        "url": row.url,
        "cellM": 1.0,
        "level": 2,
        "policy": "surface-v1",
        "calibration": "uncalibrated",
        "cells": summary["cells"],
        "surfacedCells": summary["surfaced"],
        "medianSigmaM": 0.03,
        "originHeightM": HEIGHT,
        "groundWindowM": None,  # the fixture's descriptor names no estimators
    }
    assert epoch["confidence"]["level"] == 2
    site = client.get(f"/api/v1/sites/{capture.site_id}").json()
    assert site["epochs"][0]["surface"] == epoch["surface"]


def test_a_re_run_s_surface_belongs_to_its_new_revision(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, surveyed("e1", "e1"))
    register_run(db, storage, tmp_path, capture, surveyed("e2", "e2"))

    old, new = _assets(db, capture.site_id)
    rows = {row.asset_id: row for row in _surfaces(db)}
    assert rows[old.id].level == 2 and rows[new.id].level == 1
    [epoch] = client.get(f"/api/v1/sites/{capture.site_id}/epochs").json()
    assert epoch["surface"]["assetId"] == str(new.id) and epoch["surface"]["cellM"] == 0.5


def test_a_run_that_measured_no_surface_reads_as_not_measured(
    db: Session, storage: S3Storage, tmp_path: Path, client: TestClient
) -> None:
    """A run before the surface stage (or a splat upload): `surface` null, its confidence
    as measured; a re-run without one after a run with one is a revision with none."""
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, surveyed("e1", None))
    [epoch] = client.get(f"/api/v1/sites/{capture.site_id}/epochs").json()
    assert epoch["surface"] is None and epoch["confidence"] is not None

    other = _capture(db, "other")
    register_run(db, storage, tmp_path, other, surveyed("e1", "e1"))
    register_run(db, storage, tmp_path, other, surveyed("e1", None))
    [epoch] = client.get(f"/api/v1/sites/{other.site_id}/epochs").json()
    assert epoch["surface"] is None
    assert len(_surfaces(db)) == 1  # the superseded revision's, kept


def test_a_republish_that_carried_the_surface_carries_its_map(
    db: Session, storage: S3Storage, tmp_path: Path
) -> None:
    """The same tiles republished by a run that measured nothing, with both grids carried
    into the new generation: both maps follow the new revision; one carried alone, only
    that one."""
    capture = _capture(db)
    register_run(db, storage, tmp_path, capture, surveyed("e1", "e1"))
    register_run(db, storage, tmp_path, capture, surveyed(None), carried=("confidence", "surface"))
    old, new = _assets(db, capture.site_id)
    rows = {row.asset_id: row for row in _surfaces(db)}
    assert set(rows) == {old.id, new.id}
    assert rows[new.id].descriptor == rows[old.id].descriptor
    assert rows[new.id].url == new.source["url"].replace("tileset.json", "surface.bin")

    register_run(db, storage, tmp_path, capture, surveyed(None), carried=("confidence",))
    *_, newest = _assets(db, capture.site_id)
    kinds = {m.kind for m in db.scalars(select(EpochMap).where(EpochMap.asset_id == newest.id))}
    assert kinds == {"confidence"}


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"kind": "confidence"}, "not a surface grid"),
        ({"uri": "../surface.bin"}, "not a file beside"),
        ({"channels": [{"name": "dsm", "type": "i2", "unitM": 0.01}]}, "'ground'"),
        (
            {
                "channels": [
                    {"name": "dsm", "type": "i4", "unitM": 0.01},
                    {"name": "ground", "type": "i2", "unitM": 0.01},
                    {"name": "sigma", "type": "u1", "unitM": 0.01},
                    {"name": "support", "type": "u1"},
                ]
            },
            "'dsm' of type i2",
        ),
        (
            {
                "channels": [
                    {"name": "dsm", "type": "i2", "unitM": 0.001},
                    {"name": "ground", "type": "i2", "unitM": 0.01},
                    {"name": "sigma", "type": "u1", "unitM": 0.01},
                    {"name": "support", "type": "u1"},
                ]
            },
            "in 0.01 m",
        ),
        (
            {
                "channels": [
                    {"name": "dsm", "type": "i2", "unitM": 0.01},
                    {"name": "ground", "type": "i2", "unitM": 0.01},
                    {"name": "sigma", "type": "u1", "unitM": 0.01},
                    {"name": "support", "type": "u1"},
                ]
            },
            "'groundSigma'",
        ),
        ({"noData": 0}, "noData"),
        ({"vertical": None}, "no vertical"),
        ({"vertical": {"originHeightM": "high"}}, "originHeightM"),
        ({"summary": {"cells": 3}}, "summary.surfaced"),
        ({"status": None}, "status"),
    ],
)
def test_a_surface_the_api_cannot_vouch_for_is_not_recorded(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    words: str,
) -> None:
    """Logged; the tiles and the confidence grid registered all the same."""
    monkeypatch.setattr(logging.getLogger("app.services.epoch_maps"), "disabled", False)
    capture = _capture(db)
    document = surveyed("e1", "e1")
    document["surface"].update(change)
    with caplog.at_level(logging.WARNING, logger="app.services.epoch_maps"):
        register_run(db, storage, tmp_path, capture, document)
    assert _surfaces(db) == []
    assert len(_assets(db, capture.site_id)) == 1
    assert db.scalars(select(EpochMap).where(EpochMap.kind == "confidence")).first() is not None
    assert words in caplog.text


def test_an_epoch_reads_its_ground_window_from_the_estimators() -> None:
    import uuid

    found = surface_descriptor("e1", HEIGHT)
    found["estimators"] = {"groundWindowM": 5.0}
    row = EpochMap(
        asset_id=uuid.uuid4(),
        url="u",
        cell_m=1.0,
        level=2,
        descriptor=found,
        summary=found["summary"],
    )
    assert epoch_maps.surface_to_read(row).ground_window_m == 5.0


def test_the_surface_check_takes_what_the_pipeline_writes() -> None:
    for epoch in ("e1", "e2"):
        found = surface_descriptor(epoch)
        assert epoch_maps.surface_descriptor(found) == found


def test_registration_reads_the_surface_catalog_copied_up(tmp_path: Path) -> None:
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(surveyed("e2", "e2")))
    assert registration.Registration.read(path).surface == surface_descriptor("e2", HEIGHT)
    path.write_text(json.dumps(surveyed("e2")))
    assert registration.Registration.read(path).surface is None
