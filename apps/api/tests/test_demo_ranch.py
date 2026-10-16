from __future__ import annotations

from itertools import combinations
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Site
from app.seed import seed
from app.seed.data import DEMO_SITE, overview_bookmark
from app.seed.demo_ranch import build, demo_ranch_site, dumps, load


@pytest.fixture(scope="module")
def ranch() -> dict[str, Any]:
    return load()


def test_derived_is_what_the_build_writes(ranch: dict[str, Any]) -> None:
    # `derived` is generated: edit the source half and run the build, never edit it by hand.
    assert build(ranch) == ranch["derived"], "run `python -m app.seed.demo_ranch --write`"


def test_the_file_is_written_as_the_build_writes_it(ranch: dict[str, Any]) -> None:
    from app.seed.demo_ranch import PATH

    assert PATH.read_text(encoding="utf-8") == dumps(ranch)


def test_zones_are_single_polygons_inside_the_outline_and_apart(ranch: dict[str, Any]) -> None:
    to_m = Transformer.from_crs(4326, 32610, always_xy=True).transform
    boundary = transform(to_m, shape(ranch["boundary"]["geometry"]))
    zones = {
        zone_id: transform(to_m, shape(zone["footprint"]))
        for zone_id, zone in ranch["derived"]["zones"].items()
    }
    assert sorted(zones) == ["Z-08", "Z-14", "Z-21"]
    for zone_id, zone in zones.items():
        assert zone.geom_type == "Polygon", zone_id
        assert zone.is_valid, zone_id
        # Rounding to 7 decimal places moves a vertex by a centimetre at most.
        assert zone.difference(boundary).area < 1.0, zone_id
        anchor = transform(
            to_m,
            shape({"type": "Point", "coordinates": ranch["derived"]["zones"][zone_id]["anchor"]}),
        )
        assert zone.contains(anchor), zone_id
    for (a, first), (b, second) in combinations(zones.items(), 2):
        assert first.intersection(second).area < 1.0, (a, b)


def test_every_zone_names_a_source_the_file_describes(ranch: dict[str, Any]) -> None:
    for zone in ranch["derived"]["zones"].values():
        assert zone["source"] in ranch["sources"]
        assert zone["basis"]


def test_the_description_says_what_is_simulated(ranch: dict[str, Any]) -> None:
    text = ranch["description"]
    assert "not a property line" in text
    assert "simulated" in text
    assert "Not affiliated with the landowner" in text


def test_seed_lists_the_ranch_with_its_measured_area(client: TestClient, db: Session) -> None:
    seed(db)
    ranch = load()
    summary = next(s for s in client.get("/api/v1/sites").json() if s["slug"] == "demo-ranch")
    assert summary["representations"] == []
    assert summary["name"] == ranch["name"]
    # PostGIS ST_Area(geography) and pyproj's geodesic area agree on the same ring.
    assert summary["areaM2"] == pytest.approx(ranch["derived"]["siteAreaM2"], abs=1.0)

    detail = client.get(f"/api/v1/sites/{summary['id']}").json()
    camera = ranch["camera"]
    wanted = overview_bookmark(
        "Overview",
        camera["target"][0],
        camera["target"][1],
        camera["targetHeightM"],
        camera["headingDeg"],
        camera["pitchDeg"],
        camera["rangeM"],
    )
    [bookmark] = detail["cameraBookmarks"]
    assert bookmark["name"] == "Overview" and bookmark["isDefault"] is True
    assert bookmark["longitude"] == pytest.approx(wanted.longitude, abs=1e-9)
    assert bookmark["latitude"] == pytest.approx(wanted.latitude, abs=1e-9)
    assert bookmark["height"] == pytest.approx(wanted.height, abs=1e-6)
    assert (bookmark["heading"], bookmark["pitch"]) == (camera["headingDeg"], camera["pitchDeg"])

    # Every layer the zones were made from is credited; WorldCover's CC BY needs it.
    used = {ranch["boundary"]["source"]} | {f["source"] for f in ranch["features"].values()}
    credited = {a["text"] for a in detail["attribution"]}
    assert credited == {ranch["sources"][source]["text"] for source in used}
    assert detail["license"]["spdxId"] == "CC-BY-4.0"
    assert detail["metadata"]["origin"] == "demo-ranch"
    assert set(detail["metadata"]["simulated"]) == {"fleet", "zones", "plans", "work log"}


def test_seed_brings_older_words_up_to_date_but_keeps_a_rename(
    client: TestClient, db: Session
) -> None:
    seed(db)
    redmond = db.scalar(select(Site).where(Site.slug == DEMO_SITE.slug))
    ranch = db.scalar(select(Site).where(Site.slug == "demo-ranch"))
    assert redmond is not None and ranch is not None
    # What an older database says: Redmond was where the simulated fleet ran.
    redmond.description = "Where the simulated fleet demo runs."
    ranch.attribution = []
    ranch.license = None
    ranch.name = "Our ranch"
    db.commit()

    assert seed(db) == {"sites": 0, "layers": 0}
    redmond_read = client.get(f"/api/v1/sites/by-slug/{DEMO_SITE.slug}").json()
    assert redmond_read["description"] == DEMO_SITE.description
    assert "fleet" not in redmond_read["description"]
    ranch_read = client.get("/api/v1/sites/by-slug/demo-ranch").json()
    assert len(ranch_read["attribution"]) == len(demo_ranch_site().attribution)
    assert ranch_read["license"]["name"] == "CC BY 4.0"
    # A name is the user's: the switcher renames sites, and the seed never undoes it.
    assert ranch_read["name"] == "Our ranch"
