"""Open-license archive discovery; catalog locations and dates retain their uncertainty."""

from __future__ import annotations

import math
import re
from html.parser import HTMLParser
from typing import Any

import httpx
from pydantic import HttpUrl
from pyproj import Geod
from shapely.geometry import Point, box

from app.research.providers.base import (
    SourceContext,
    SourceError,
    SourceResult,
    SourceSpec,
    evidence,
    fetch_json,
)
from app.schemas.geojson import Point as GeoPoint
from app.schemas.geojson import Polygon
from app.schemas.land_archives import ArchiveMedia
from app.schemas.research import EvidenceContent

ARCHIVE_SOURCES = {
    "commons-place-images": SourceSpec(
        id="commons-place-images",
        name="Open photographs near this land",
        domain="history",
        coverage="Global, uneven geotagged Wikimedia Commons coverage",
        resolution="Catalog coordinates may describe a camera or a depicted location, not an event footprint",
        license="Record-specific CC0, CC BY, CC BY-SA or explicit public domain; other licenses excluded",
        attribution="Wikimedia Commons and each credited creator",
        documentation_url="https://www.mediawiki.org/wiki/API:Geosearch",
        endpoint="https://commons.wikimedia.org/w/api.php",
    ),
    "usgs-historical-maps": SourceSpec(
        id="usgs-historical-maps",
        name="Historical USGS maps covering the land",
        domain="history",
        coverage="United States and covered territories; historical quadrangle collection",
        resolution="Catalog sheet footprint; not a legal parcel or verified image alignment",
        license="U.S. public domain",
        attribution="U.S. Geological Survey, Historical Topographic Map Collection",
        documentation_url="https://www.usgs.gov/the-national-map-data-delivery/topographic-map-access-points",
        endpoint="https://tnmaccess.nationalmap.gov/api/v1/products",
    ),
}


class PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain(value: object, limit: int = 2000) -> str:
    parser = PlainText()
    parser.feed(str(value or "")[:20_000])
    return " ".join(" ".join(parser.parts).split())[:limit]


def open_license(metadata: dict[str, Any]) -> tuple[str, str] | None:
    name = plain(metadata.get("LicenseShortName", {}).get("value"), 500)
    url = str(metadata.get("LicenseUrl", {}).get("value", "")).strip()
    normalized = url.replace("http://", "https://").rstrip("/")
    if re.fullmatch(
        r"https://creativecommons\.org/licenses/by(?:-sa)?/(?:1\.0|2\.0|2\.5|3\.0|4\.0)", normalized
    ):
        return name or "Creative Commons attribution license", normalized
    if normalized in {
        "https://creativecommons.org/publicdomain/zero/1.0",
        "https://creativecommons.org/publicdomain/mark/1.0",
    }:
        return name or "Public-domain dedication/mark", normalized
    # Commons records sometimes express a public-domain determination without LicenseUrl.
    if (
        name == "Public domain"
        and str(metadata.get("Copyrighted", {}).get("value", "")).lower() == "false"
    ):
        return name, "https://commons.wikimedia.org/wiki/Help:Public_domain"
    return None


def commons(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = ARCHIVE_SOURCES["commons-place-images"]
    lon, lat = context.point
    west, south, east, north = context.geometry.bounds
    if east - west > 2 or north - south > 2:
        return SourceResult(
            spec.id, "uncovered", "Choose a smaller area for nearby archive-photo discovery."
        )
    geod = Geod(ellps="WGS84")
    distance = max(
        geod.inv(lon, lat, x, y)[2]
        for x, y in [(west, south), (west, north), (east, south), (east, north)]
    )
    radius = min(10_000, max(500, math.ceil(distance + 250)))
    # Small batches respect Commons' guidance for expensive imageinfo/extmetadata requests.
    items: list[tuple[str, EvidenceContent]] = []
    excluded = 0
    continuation: dict[str, str | int | float | bool] = {}
    queried = 0
    for _ in range(2):
        raw, _url = fetch_json(
            client,
            spec,
            {
                "action": "query",
                "format": "json",
                "generator": "geosearch",
                "ggscoord": f"{lat}|{lon}",
                "ggsradius": radius,
                "ggslimit": 10,
                "ggsnamespace": 6,
                "prop": "imageinfo|coordinates",
                "iiprop": "url|extmetadata|mime|thumbmime|sha1|timestamp",
                "iiurlwidth": 640,
                "coprimary": "primary",
                "iiextmetadatalanguage": "en",
                **continuation,
            },
        )
        if raw.get("error"):
            raise SourceError("The Commons catalog returned an error.")
        pages = raw.get("query", {}).get("pages", {})
        if not isinstance(pages, dict):
            raise SourceError("Unexpected Commons catalog response.")
        for page in pages.values():
            queried += 1
            if not page.get("imageinfo") or not page.get("coordinates"):
                continue
            info, coordinate = page["imageinfo"][0], page["coordinates"][0]
            if not info.get("thumburl"):
                continue
            meta = info.get("extmetadata", {})
            license_info = open_license(meta)
            if not license_info:
                excluded += 1
                continue
            if info.get("thumbmime", info.get("mime")) not in {
                "image/jpeg",
                "image/png",
                "image/webp",
            }:
                continue
            x, y = coordinate.get("lon"), coordinate.get("lat")
            if (
                not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not (-180 <= x <= 180 and -90 <= y <= 90)
            ):
                continue
            if coordinate.get("globe", "earth") != "earth":
                continue
            within = context.geometry.covers(Point(x, y))
            source_url = info.get("descriptionurl", "")
            if not source_url.startswith("https://commons.wikimedia.org/wiki/File:"):
                continue
            title = plain(meta.get("ObjectName", {}).get("value") or page.get("title"), 500)
            creator = plain(
                meta.get("Artist", {}).get("value")
                or meta.get("Credit", {}).get("value")
                or "Creator not specified in catalog",
                1200,
            )
            relevance = (
                (
                    "The catalog coordinate falls inside the selected boundary. "
                    if within
                    else "The catalog coordinate is nearby, outside the selected boundary. "
                )
                + "It may locate the camera or a depicted subject. "
                "It does not prove that an event happened on this land."
            )
            media = ArchiveMedia(
                kind="photograph",
                title=title,
                description=plain(meta.get("ImageDescription", {}).get("value"), 3000),
                source_url=HttpUrl(source_url),
                preview_url=HttpUrl(info["thumburl"]),
                creator=creator,
                license=license_info[0],
                license_url=HttpUrl(license_info[1]),
                source_date=plain(meta.get("DateTimeOriginal", {}).get("value"), 500) or None,
                date_meaning="Original-date text supplied by the catalog; "
                "may describe creation, depiction or reproduction. "
                "Upload dates are not event dates.",
                location=GeoPoint(coordinates=[x, y]),
                location_meaning="catalog-coordinate",
                relevance=relevance,
                source_version=f"Commons file SHA-1 {info['sha1']}" if info.get("sha1") else None,
            )
            item = evidence(spec, source_url, title, media.model_dump(mode="json"), relevance)
            item.media, item.license, item.attribution = (
                media,
                media.license,
                f"{creator}; Wikimedia Commons; {media.license}",
            )
            item.record_id = str(page["pageid"])
            item.spatial_relevance = "within" if within else "nearby"
            items.append((item.record_id, item))
        next_page = raw.get("continue", {})
        if not next_page:
            continuation = {}
            break
        continuation = {
            key: value
            for key, value in next_page.items()
            if key in {"continue", "ggscontinue"} and isinstance(value, (str, int))
        }
        if not continuation:
            break
    # Preserve discovery order and remove duplicates across provider pages.
    items = list(dict(items).items())
    return SourceResult(
        spec.id,
        "available" if items else "empty",
        f"Found {len(items)} openly licensed photographs in a bounded nearby catalog sample. "
        "Dates and land relevance need verification."
        if items
        else "No reusable geotagged photographs were found in this sample. "
        "Untagged or undigitized archives may still exist.",
        items,
        {
            "records": records_for_agent(items),
            "sampleSize": queried,
            "radiusM": radius,
            "truncated": bool(continuation),
            "limitReached": queried >= 10,
            "excludedLicense": excluded,
            "coverageNote": "Search is capped at 10 km from a representative point and 20 returned records; "
            "it is not an exhaustive archive search.",
        },
    )


def historical_maps(context: SourceContext, client: httpx.Client) -> SourceResult:
    spec = ARCHIVE_SOURCES["usgs-historical-maps"]
    bounds = context.geometry.bounds
    if bounds[2] - bounds[0] > 5 or bounds[3] - bounds[1] > 5:
        return SourceResult(
            spec.id, "uncovered", "Choose a smaller area to discover historical map sheets."
        )
    raw, _url = fetch_json(
        client,
        spec,
        {
            "datasets": "Historical Topographic Maps",
            "bbox": ",".join(str(v) for v in bounds),
            "max": 50,
            "outputFormat": "JSON",
        },
    )
    if raw.get("errors"):
        raise SourceError("The National Map catalog returned an error.")
    candidates = raw.get("items", [])
    items: list[tuple[str, EvidenceContent]] = []
    for record in sorted(candidates, key=lambda value: str(value.get("publicationDate", "9999"))):
        extent = record.get("boundingBox", {})
        try:
            west, south, east, north = [
                float(extent[key]) for key in ("minX", "minY", "maxX", "maxY")
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            continue
        if context.geometry.intersection(box(west, south, east, north)).area <= 0:
            continue
        source_url = str(record.get("metaUrl", ""))
        preview = str(record.get("previewGraphicURL", ""))
        download = str(record.get("downloadURL", ""))
        if not source_url.startswith(
            "https://www.sciencebase.gov/catalog/item/"
        ) or not preview.startswith(
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/"
        ):
            continue
        year = str(record.get("publicationDate", ""))[:4]
        relevance = (
            "The catalog map-sheet bounding box overlaps this land. Sheet coverage is approximate; "
            "it does not establish legal boundaries or georeference the preview image."
        )
        media = ArchiveMedia(
            kind="historical-map",
            title=plain(record.get("title"), 500),
            description=plain(record.get("moreInfo"), 3000),
            source_url=HttpUrl(source_url),
            preview_url=HttpUrl(preview),
            download_url=HttpUrl(download)
            if download.startswith(
                "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/"
            )
            else None,
            creator=spec.attribution,
            license=spec.license,
            license_url=HttpUrl(
                "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits"
            ),
            source_date=year if re.fullmatch(r"\d{4}", year) else None,
            date_meaning="Catalog publication year; surveying, revision and scanning may have different dates.",
            location=Polygon(
                coordinates=[
                    [[west, south], [east, south], [east, north], [west, north], [west, south]]
                ]
            ),
            location_meaning="catalog-footprint",
            relevance=relevance,
            source_version=str(record.get("lastUpdated")) if record.get("lastUpdated") else None,
        )
        item = evidence(spec, source_url, media.title, media.model_dump(mode="json"), relevance)
        item.media, item.record_id, item.spatial_relevance = (
            media,
            str(record.get("sourceId")),
            "intersects",
        )
        items.append((item.record_id, item))
        if len(items) >= 12:
            break
    return SourceResult(
        spec.id,
        "available" if items else "empty",
        f"Found {len(items)} historical map sheets whose catalog footprints overlap this land. "
        "The oldest retrieved sheets are shown first."
        if items
        else "No overlapping historical USGS sheets were found in this bounded catalog search.",
        items,
        {
            "records": records_for_agent(items),
            "catalogTotal": raw.get("total"),
            "retrieved": len(candidates),
            "shown": len(items),
            "truncated": len(items) < len(candidates) or (raw.get("total") or 0) > len(candidates),
            "coverageNote": "Up to 50 catalog records are checked; up to 12 overlapping sheets are shown. "
            "This is not a complete title or land-use history.",
        },
    )


def records_for_agent(items: list[tuple[str, EvidenceContent]]) -> list[dict[str, Any]]:
    return [
        {
            "recordId": key,
            "title": item.title,
            "sourceDate": item.media.source_date,
            "dateMeaning": item.media.date_meaning,
            "description": item.media.description[:600],
            "sourceUrl": str(item.url),
            "license": item.license,
            "relevance": item.relevance_note,
        }
        for key, item in items
        if item.media is not None
    ]
