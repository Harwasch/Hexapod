"""`PUT /assets/{id}/scale`: an existing scan's real-world size, set at runtime.

A phone video registered before the pipeline estimated scales is drawn at one COLMAP unit
to the metre, two to eight times too big, and its files cannot be resized in place. The
asset carries a factor the viewer applies instead -- absolute, relative to the model as
registered -- and everything the catalog places on the globe moves with it about the
tiles' origin. These hold what that has to mean: the write token gates it, the bounds and
the evidence are checked, two scales compose to the second, a reset and a re-run both go
back to the registration, and only a splat a pipeline run placed can be resized at all.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import _db
from app.config import Settings
from app.main import create_app
from app.models import Asset, Capture, Job, Site
from app.models.enums import CaptureKind, Representation, RunStatus
from app.seed import seed
from app.services import placement
from app.storage import S3Storage
from app.worker import registration
from app.worker.carry import CarryPlan
from tests.conftest import SQUARE, site_payload

TOKEN = "scale-token"
LAT, LON, HEIGHT = 44.7965, -111.1346, 1520.0
#: The registered splat's own bounding box, east/north/up metres about the origin.
BOX = {"min": [-4.0, -2.0, -1.0], "max": [6.0, 8.0, 3.0]}
#: Two of the capture's ground cells, metres east and north of the origin and up from it.
CELLS = [(2.0, 4.0, -1.0), (-3.0, 1.0, -0.5)]


def _deg(east: float, north: float) -> tuple[float, float]:
    """Degrees of longitude and latitude `east`/`north` metres from the origin."""
    return (
        east / (placement.METRES_PER_DEGREE * math.cos(math.radians(LAT))),
        north / placement.METRES_PER_DEGREE,
    )


def _document(slug: str = "spool") -> dict[str, Any]:
    """`registration.json` for a phone video registered at an unresolved scale."""
    samples = []
    for east, north, up in CELLS:
        dlon, dlat = _deg(east, north)
        samples.append({"lon": LON + dlon, "lat": LAT + dlat, "z": up})
    return {
        "slug": slug,
        "title": "Spool",
        "georef": {
            "lat": LAT,
            "lon": LON,
            "height": HEIGHT,
            "georefMethod": "exif-gps",
            "scaleSource": "unresolved",
            "uncertaintyM": 10.0,
            "frame": {"source": "camera-up", "scale": 1.0},
        },
        "bboxLocalM": BOX,
        "ground": {"origin": {"lat": LAT, "lon": LON, "height": HEIGHT}, "samples": samples},
    }


def _registration(tmp_path: Path, document: dict[str, Any]) -> registration.Registration:
    path = tmp_path / f"registration-{uuid.uuid4()}.json"
    path.write_text(json.dumps(document))
    return registration.Registration.read(path)


def _job(db: Session, capture: Capture) -> Job:
    """A finished run of the capture: one active run per capture, so a re-run's run is the
    second finished one rather than a second in flight."""
    job = Job(
        capture_id=capture.id,
        recipe="photo-reconstruct",
        recipe_version="10",
        params={},
        status=RunStatus.COMPLETE,
    )
    db.add(job)
    db.commit()
    return job


def _register(
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    capture: Capture,
    document: dict[str, Any] | None = None,
) -> Job:
    """One run of the capture registered, as the worker registers it."""
    job = _job(db, capture)
    url = f"https://cdn.example.com/twin-worker-test/runs/{job.id}/package/splat/tileset.json"
    live = registration.live_tileset_url(db, capture)
    registration.register(
        db,
        storage,
        capture=capture,
        job_id=job.id,
        registration=_registration(tmp_path, document or _document()),
        tiles_stage_id="package",
        published=registration.Published(
            tileset=url, thumbnail=None, coverage=None, carry=CarryPlan(based_on=live)
        ),
    )
    return job


@pytest.fixture
def scan(db: Session, storage: S3Storage, tmp_path: Path) -> tuple[Capture, Asset]:
    """A phone video's site and splat, registered by a pipeline run at an unresolved scale."""
    capture = Capture(slug="spool", name="Spool", kind=CaptureKind.VIDEO, metadata_={})
    db.add(capture)
    db.commit()
    _register(db, storage, tmp_path, capture)
    db.refresh(capture)
    assert capture.site_id is not None
    splat = registration.splat_asset(db, capture.site_id)
    assert splat is not None
    return capture, splat


def _put(client: TestClient, asset: Asset, body: dict[str, Any], **kwargs: Any) -> Any:
    return client.put(f"/api/v1/assets/{asset.id}/scale", json=body, **kwargs)


def _site(client: TestClient, asset: Asset) -> dict[str, Any]:
    response = client.get(f"/api/v1/sites/{asset.site_id}")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _ring(site: dict[str, Any]) -> list[tuple[float, float]]:
    return [(p[0], p[1]) for p in site["boundary"]["coordinates"][0][0]]


def _offsets_m(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Each vertex's metres east and north of the origin."""
    one_east, one_north = _deg(1.0, 1.0)
    return [((lon - LON) / one_east, (lat - LAT) / one_north) for lon, lat in ring]


def _assert_scaled(ring: list[tuple[float, float]], factor: float) -> None:
    """The ring is the registered bounding box, resized about the origin by `factor`."""
    (min_e, min_n, _), (max_e, max_n, _) = BOX["min"], BOX["max"]
    expected = [
        (min_e, min_n),
        (max_e, min_n),
        (max_e, max_n),
        (min_e, max_n),
        (min_e, min_n),
    ]
    got = _offsets_m(ring)
    assert len(got) == len(expected)
    for (east, north), (want_e, want_n) in zip(got, expected, strict=True):
        assert east == pytest.approx(want_e * factor, abs=1e-6)
        assert north == pytest.approx(want_n * factor, abs=1e-6)


# --- the gate and the body ---------------------------------------------------------------


@pytest.fixture
def gated(db: Session) -> Iterator[TestClient]:
    """An app with the write token configured, the database overridden and nothing else."""
    app = create_app(Settings(api_write_token=TOKEN))

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


def test_setting_a_scale_needs_the_write_token(
    gated: TestClient, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan

    refused = _put(gated, splat, {"scale": 0.5})
    wrong = _put(gated, splat, {"scale": 0.5}, headers={"Authorization": "Bearer nope"})
    allowed = _put(gated, splat, {"scale": 0.5}, headers={"Authorization": f"Bearer {TOKEN}"})

    assert refused.status_code == 401
    assert wrong.status_code == 401
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["renderConfig"]["scale"] == 0.5


@pytest.mark.parametrize(
    "body",
    [
        {"scale": 0},
        {"scale": -1},
        {"scale": 0.005},
        {"scale": 101},
        {},
        {"evidence": {"method": "direct"}},
        {"reset": True, "scale": 0.5},
        {"reset": True, "evidence": {"method": "direct"}},
        {"scale": 0.5, "evidence": {"method": "measured-length", "measuredLengthM": 2.0}},
        # The length says 0.4 (0.8 x 1 / 2), not 0.5.
        {
            "scale": 0.5,
            "evidence": {
                "method": "measured-length",
                "measuredLengthM": 2.0,
                "trueLengthM": 1.0,
                "measuredAtScale": 0.8,
            },
        },
        {"scale": 0.5, "evidence": {"method": "camera-height-estimate", "cameras": 40}},
        {"scale": 0.5, "evidence": {"method": "guess"}},
        {"scale": 0.5, "evidence": {"method": "direct", "setAt": "2026-10-05T00:00:00Z"}},
    ],
)
def test_a_scale_out_of_bounds_or_badly_evidenced_is_refused(
    client: TestClient, scan: tuple[Capture, Asset], body: dict[str, Any]
) -> None:
    _, splat = scan
    before = _site(client, splat)

    response = _put(client, splat, body)

    assert response.status_code == 422, response.text
    assert _site(client, splat)["boundary"] == before["boundary"]
    assert client.get(f"/api/v1/assets/{splat.id}").json()["renderConfig"]["scale"] == 1.0


def test_an_unknown_asset_is_404(client: TestClient, scan: tuple[Capture, Asset]) -> None:
    response = client.put(f"/api/v1/assets/{uuid.uuid4()}/scale", json={"scale": 0.5})
    assert response.status_code == 404


# --- what a scale does --------------------------------------------------------------------


def test_a_registered_scan_starts_at_one_with_nothing_stored(
    client: TestClient, db: Session, scan: tuple[Capture, Asset]
) -> None:
    """Unscaled is no keys at all, so the code before them can still read the row."""
    _, splat = scan
    db.refresh(splat)

    assert "scale" not in splat.render_config
    assert "scaleEvidence" not in splat.render_config
    read = client.get(f"/api/v1/assets/{splat.id}").json()
    assert read["renderConfig"]["scale"] == 1.0
    assert read["renderConfig"]["scaleEvidence"] is None
    _assert_scaled(_ring(_site(client, splat)), 1.0)


def test_scales_compose_to_the_last_one_not_their_product(
    client: TestClient, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan

    first = _put(client, splat, {"scale": 0.8})
    second = _put(client, splat, {"scale": 0.5})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["renderConfig"]["scale"] == 0.5
    _assert_scaled(_ring(_site(client, splat)), 0.5)


def test_a_scale_moves_the_boundary_centroid_footprint_and_ground_about_the_origin(
    client: TestClient, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan
    patched = client.patch(f"/api/v1/assets/{splat.id}", json={"footprint": SQUARE})
    assert patched.status_code == 200, patched.text
    before_site = _site(client, splat)
    before = client.get(f"/api/v1/assets/{splat.id}").json()

    response = _put(client, splat, {"scale": 0.4})

    assert response.status_code == 200, response.text
    after = response.json()
    _assert_scaled(_ring(_site(client, splat)), 0.4)
    # The centroid moved toward the origin by the same factor.
    old_c, new_c = before_site["centroid"], _site(client, splat)["centroid"]
    assert new_c["longitude"] - LON == pytest.approx(0.4 * (old_c["longitude"] - LON), abs=1e-12)
    assert new_c["latitude"] - LAT == pytest.approx(0.4 * (old_c["latitude"] - LAT), abs=1e-12)
    # So did the asset's own footprint.
    for (lon0, lat0), (lon1, lat1) in zip(
        before["footprint"]["coordinates"][0][0],
        after["footprint"]["coordinates"][0][0],
        strict=True,
    ):
        assert lon1 - LON == pytest.approx(0.4 * (lon0 - LON), abs=1e-9)
        assert lat1 - LAT == pytest.approx(0.4 * (lat0 - LAT), abs=1e-9)
    # And each ground cell: across and up, about the origin's height.
    cells = after["renderConfig"]["groundSamples"]
    assert len(cells) == len(CELLS)
    for cell, (east, north, up) in zip(cells, CELLS, strict=True):
        dlon, dlat = _deg(0.4 * east, 0.4 * north)
        assert cell["lon"] == pytest.approx(LON + dlon, abs=1e-9)
        assert cell["lat"] == pytest.approx(LAT + dlat, abs=1e-9)
        assert cell["height"] == pytest.approx(HEIGHT + 0.4 * up, abs=1e-9)
    # The site's area is the scaled box's: 10 m x 10 m at 0.4 is 16 m2.
    assert _site(client, splat)["areaM2"] == pytest.approx(16.0, rel=0.01)


def test_a_measured_length_is_manual_and_says_what_was_measured(
    client: TestClient, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan
    evidence = {
        "method": "measured-length",
        "measuredLengthM": 2.4,
        "trueLengthM": 0.9,
        "measuredAtScale": 1.0,
        "note": "the spool's diameter",
    }

    response = _put(client, splat, {"scale": 0.375, "evidence": evidence})

    assert response.status_code == 200, response.text
    read = response.json()
    stored = read["renderConfig"]["scaleEvidence"]
    assert {k: stored[k] for k in evidence} == evidence
    assert stored["setAt"] is not None
    assert stored["registeredScaleSource"] == "unresolved"
    assert read["provenance"]["scaleSource"] == "manual"
    assert read["provenance"]["scaleUncertaintyPct"] is None


def test_no_evidence_is_a_direct_manual_scale(
    client: TestClient, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan

    read = _put(client, splat, {"scale": 0.5}).json()

    assert read["renderConfig"]["scaleEvidence"]["method"] == "direct"
    assert read["provenance"]["scaleSource"] == "manual"


def test_an_estimate_is_recorded_as_one_and_a_reset_restores_the_registration(
    client: TestClient, db: Session, scan: tuple[Capture, Asset]
) -> None:
    """Estimated, then measured over it, then reset: the reset goes back to what the scan
    was registered as -- `unresolved` -- not to the estimate before the measurement."""
    capture, splat = scan
    registered = client.get(f"/api/v1/assets/{splat.id}").json()
    estimate = {
        "method": "camera-height-estimate",
        "cameraHeightM": 1.5,
        "cameraHeightUnits": 4.6875,
        "metresPerUnit": 0.32,
        "cameras": 56,
        "uncertaintyPct": 23.1,
    }

    estimated = _put(client, splat, {"scale": 0.32, "evidence": estimate}).json()
    assert estimated["provenance"]["scaleSource"] == "camera-height-estimate"
    assert estimated["provenance"]["scaleUncertaintyPct"] == 23.1
    assert estimated["renderConfig"]["scaleEvidence"]["cameras"] == 56

    measured = _put(
        client,
        splat,
        {
            "scale": 0.3,
            "evidence": {"method": "measured-length", "measuredLengthM": 3.0, "trueLengthM": 0.9},
        },
    ).json()
    assert measured["provenance"]["scaleSource"] == "manual"
    assert measured["provenance"]["scaleUncertaintyPct"] is None
    assert measured["renderConfig"]["scaleEvidence"]["registeredScaleSource"] == "unresolved"

    reset = _put(client, splat, {"reset": True})

    assert reset.status_code == 200, reset.text
    read = reset.json()
    assert read["renderConfig"]["scale"] == 1.0
    assert read["renderConfig"]["scaleEvidence"] is None
    assert read["provenance"] == registered["provenance"]
    for cell, was in zip(
        read["renderConfig"]["groundSamples"],
        registered["renderConfig"]["groundSamples"],
        strict=True,
    ):
        assert cell == pytest.approx(was, abs=1e-9)
    _assert_scaled(_ring(_site(client, splat)), 1.0)
    db.refresh(splat)
    assert "scale" not in splat.render_config
    # The capture's own record describes its files, which a runtime scale did not touch.
    db.refresh(capture)
    assert capture.scale_source is not None and capture.scale_source.value == "unresolved"


def test_a_patch_of_the_render_config_keeps_the_scale(
    client: TestClient, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan
    _put(client, splat, {"scale": 0.5})

    patched = client.patch(
        f"/api/v1/assets/{splat.id}",
        json={"renderConfig": {"heightOffsetM": 0.3, "scale": 2.0, "clampToGround": True}},
    )

    assert patched.status_code == 200, patched.text
    render = patched.json()["renderConfig"]
    assert render["heightOffsetM"] == 0.3
    assert render["scale"] == 0.5
    assert render["scaleEvidence"]["method"] == "direct"
    assert patched.json()["provenance"]["scaleSource"] == "manual"
    _assert_scaled(_ring(_site(client, splat)), 0.5)


# --- what cannot be resized ---------------------------------------------------------------


def test_an_asset_no_pipeline_run_placed_is_refused(client: TestClient, db: Session) -> None:
    """A seeded or hand-made site records no origin for its tiles: 409, and nothing moves."""
    site = client.post("/api/v1/sites", json=site_payload()).json()
    ion_splat = site["assets"][0]
    tiles_splat = client.post(
        "/api/v1/assets",
        json={
            "siteId": site["id"],
            "name": "Hand-made splat",
            "representation": "gaussian-splat",
            "source": {"type": "3d-tiles-url", "url": "https://example.com/t/tileset.json"},
        },
    ).json()
    mesh = client.post(
        "/api/v1/assets",
        json={
            "name": "Loose mesh",
            "representation": "mesh",
            "source": {"type": "3d-tiles-url", "url": "https://example.com/m/tileset.json"},
        },
    ).json()

    for asset in (ion_splat, tiles_splat, mesh):
        response = client.put(f"/api/v1/assets/{asset['id']}/scale", json={"scale": 0.5})
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "not_scalable"
        assert "cannot be rescaled" in response.json()["detail"]
    assert (
        client.get(f"/api/v1/sites/{site['id']}").json()["boundary"]["coordinates"][0]
        == (SQUARE["coordinates"])
    )


def test_only_the_registered_splat_of_a_pipeline_site_is_resized(
    client: TestClient, db: Session, scan: tuple[Capture, Asset]
) -> None:
    _, splat = scan
    other = client.post(
        "/api/v1/assets",
        json={
            "siteId": str(splat.site_id),
            "name": "A second splat",
            "representation": "gaussian-splat",
            "source": {"type": "3d-tiles-url", "url": "https://example.com/s/tileset.json"},
        },
    ).json()

    response = client.put(f"/api/v1/assets/{other['id']}/scale", json={"scale": 0.5})

    assert response.status_code == 409
    assert response.json()["code"] == "not_scalable"


# --- a re-run -------------------------------------------------------------------------------


def test_a_re_run_resets_the_scale_and_takes_the_boundary_back(
    client: TestClient,
    db: Session,
    storage: S3Storage,
    tmp_path: Path,
    scan: tuple[Capture, Asset],
) -> None:
    """The new tiles carry the run's own scale, so the correction to the old ones goes:
    scale 1, no evidence, the new run's provenance, and the boundary resized back."""
    capture, splat = scan
    assert _put(client, splat, {"scale": 0.4}).status_code == 200
    estimated = _document()
    estimated["georef"]["scaleSource"] = "camera-height-estimate"
    estimated["georef"]["frame"] = {
        "source": "camera-up",
        "scale": 0.4,
        "scaleEstimate": {"method": "camera-height", "uncertaintyPct": 21.5},
    }

    second = _register(db, storage, tmp_path, capture, estimated)

    db.expire_all()
    asset = db.get(Asset, splat.id)
    assert asset is not None
    assert str(second.id) in asset.source["url"]
    assert "scale" not in asset.render_config
    assert "scaleEvidence" not in asset.render_config
    assert asset.render_config["provenance"]["scaleSource"] == "camera-height-estimate"
    assert asset.render_config["provenance"]["scaleUncertaintyPct"] == 21.5
    _assert_scaled(_ring(_site(client, splat)), 1.0)
    site = db.get(Site, splat.site_id)
    assert site is not None
    assert site.metadata_["jobId"] == str(second.id)
    assert site.metadata_["registration"]["georef"]["scaleSource"] == "camera-height-estimate"
    # Still the one splat, and it can be resized again about the new registration's origin.
    splats = db.scalars(
        select(Asset).where(
            Asset.site_id == site.id, Asset.representation == Representation.GAUSSIAN_SPLAT
        )
    ).all()
    assert [a.id for a in splats] == [splat.id]
    assert _put(client, splat, {"scale": 0.9}).status_code == 200
    _assert_scaled(_ring(_site(client, splat)), 0.9)


def test_seeding_twice_stores_no_scale_keys_on_unscaled_assets(db: Session) -> None:
    """The seed refreshes a seeded asset's render config when it differs from the one it
    wants; compared as stored, an unscaled one never differs by the keys left out."""
    seed(db)
    seed(db)

    db.expire_all()
    assets = db.scalars(select(Asset)).all()
    assert assets
    for asset in assets:
        assert "scale" not in asset.render_config
        assert "scaleEvidence" not in asset.render_config
