"""Taxonomic name resolution; never a field identification or native-status determination."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

import httpx

from app.research.providers.base import SourceError, SourceResult, SourceSpec, evidence, fetch_json
from app.schemas.land_ecology import TaxonQuery

CHECKLIST = "7ddf754f-d193-4cc9-b351-99906754a03b"
SPEC = SourceSpec(
    id="col-taxonomy",
    name="Catalogue of Life name matching",
    domain="ecology",
    coverage="Global taxonomy; matching depends on supplied name and classification",
    resolution="Name usage and taxonomic classification, not organism identification or habitat suitability",
    license="https://creativecommons.org/licenses/by/4.0/",
    attribution="Catalogue of Life and contributing taxonomic sources; name matching via GBIF",
    documentation_url="https://techdocs.gbif.org/en/data-processing/taxonomy-interpretation",
    endpoint="https://api.gbif.org/v2/species/match",
)
METADATA = replace(SPEC, endpoint="https://api.gbif.org/v2/species/match/metadata")
LIMITATION = (
    "A taxonomic name match does not verify a field identification, local presence, native or invasive status, "
    "species cover or suitability for restoration. Fuzzy, ambiguous and higher-rank matches need review. "
    "Conservation-status fields from separate licensed datasets are excluded."
)


def text(value: Any, length: int = 300) -> str | None:
    return value[:length] if isinstance(value, str) else None


def usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("key"), str):
        return None
    result = {
        k: text(value.get(k))
        for k in ("key", "name", "canonicalName", "authorship", "rank", "status", "code", "type")
    }
    key = result["key"]
    result["url"] = (
        f"https://www.gbif.org/taxon/{key}"
        if key and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key)
        else None
    )
    return result


def match(query: TaxonQuery, client: httpx.Client) -> SourceResult:
    params: dict[str, str | int | float | bool] = {
        "scientificName": query.scientific_name,
        "checklistKey": CHECKLIST,
        "verbose": "true",
    }
    if query.kingdom:
        params["kingdom"] = query.kingdom
    try:
        index, metadata_url = fetch_json(client, METADATA, {"checklistKey": CHECKLIST})
        main = index.get("mainIndex", {})
        if not isinstance(main, dict) or main.get("datasetKey") != CHECKLIST:
            raise SourceError("The matching service returned a different taxonomy index.")
        raw, url = fetch_json(client, SPEC, params)
        diagnostics = raw.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise SourceError("The matching service omitted its diagnostics.")
        matched, accepted = usage(raw.get("usage")), usage(raw.get("acceptedUsage"))
        match_type = text(diagnostics.get("matchType"), 80) or "UNKNOWN"
        confidence = diagnostics.get("confidence")
        confidence = (
            confidence
            if isinstance(confidence, (float, int))
            and not isinstance(confidence, bool)
            and 0 <= confidence <= 100
            else None
        )
        alternatives = diagnostics.get("alternatives", raw.get("alternatives", []))
        if not isinstance(alternatives, list):
            alternatives = []
        cleaned_alternatives = []
        for item in alternatives[:5]:
            if isinstance(item, dict):
                candidate = usage(item.get("usage", item))
                if candidate:
                    cleaned_alternatives.append(candidate)
        classification = raw.get("classification", [])
        if not isinstance(classification, list):
            classification = []
        data: dict[str, Any] = {
            "query": query.model_dump(mode="json"),
            "checklistKey": CHECKLIST,
            "indexObserved": {
                "created": text(index.get("created")),
                **{
                    k: text(main.get(k))
                    for k in ("clbDatasetKey", "datasetKey", "datasetAlias", "datasetTitle")
                },
                "metadataUrl": metadata_url,
                "interpretation": "Matching-service metadata observed immediately before the lookup; "
                "not the registry's latest published release.",
            },
            "usage": matched,
            "acceptedUsage": accepted,
            "synonym": raw.get("synonym") is True,
            "matchType": match_type,
            "confidence": confidence,
            "diagnosticNote": text(diagnostics.get("note"), 1000),
            "issues": [text(item, 100) for item in raw.get("issues", [])[:30]]
            if isinstance(raw.get("issues"), list)
            else [],
            "classification": [
                {k: text(item.get(k)) for k in ("key", "name", "rank")}
                for item in classification[:30]
                if isinstance(item, dict)
            ],
            "alternatives": cleaned_alternatives,
            "alternativesTruncated": len(alternatives) > 5,
            "nativeStatus": "not-assessed",
            "fieldIdentification": "not-verified",
            "limitation": LIMITATION,
        }
        data["classificationTruncated"] = len(classification) > 30
        while len(json.dumps(data, ensure_ascii=False)) > 29_000:
            if cleaned_alternatives:
                cleaned_alternatives.pop()
                data["alternativesTruncated"] = True
            elif data["classification"]:
                data["classification"].pop()
                data["classificationTruncated"] = True
            else:
                raise SourceError("The taxonomic source metadata exceeds the evidence limit.")
        found = matched is not None and match_type != "NONE"
        summary = (
            f"{query.scientific_name}: {match_type.lower()} name match to "
            f"{(accepted or matched or {}).get('name') or 'unnamed usage'} "
            f"({(accepted or matched or {}).get('rank') or 'unreported rank'}). "
            "This does not verify the observed organism or its native status."
            if found
            else f"No usable taxonomic name match for {query.scientific_name}. "
            "This does not show that the species is absent or that the supplied identification is invalid."
        )
        item = evidence(
            SPEC, url, f"Name lookup · {query.scientific_name}", data, LIMITATION, regional=True
        )
        item.spatial_relevance = "unresolved"
        item.record_id = str((accepted or matched or {}).get("key") or query.scientific_name)
        return SourceResult(
            SPEC.id, "available" if found else "empty", summary, [("name", item)], data
        )
    except (httpx.HTTPError, SourceError, ValueError, KeyError, TypeError) as error:
        reason = (
            f"HTTP {error.response.status_code}"
            if isinstance(error, httpx.HTTPStatusError)
            else type(error).__name__
        )
        return SourceResult(
            SPEC.id,
            "unavailable",
            f"Name lookup for {query.scientific_name} could not be retrieved ({reason}).",
        )
