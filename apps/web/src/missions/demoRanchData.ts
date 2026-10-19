/**
 * The demo ranch, read from the one file both sides use.
 *
 * `apps/api/app/seed/demo_ranch.json` is the canonical description of the place the simulated
 * fleet works: a real outline of oak savanna on California's Central Coast, Figueroa Creek from
 * the USGS NHD, grassland and oak patches from ESA WorldCover, a simulated fence, the camera
 * framing, and a `derived` block (zone footprints, geodesic areas, chip anchors, lengths) that
 * a Python build step writes (`python -m app.seed.demo_ranch --write`). The API seeds the
 * `demo-ranch` site from it; this module gives the web app the same numbers, for the built-in
 * offline copy (`api/fallback.ts`) and the simulated project (`missions/demo.ts`). Nothing here
 * is typed in by hand, so a zone's acres are its footprint's acres everywhere.
 *
 * The import is relative on purpose: apps/api is outside this package, and Vite (whose fs
 * allow-list is the workspace root), vitest, tsc and the Cloudflare build from the repository
 * root all resolve it. No imports from `fallback.ts` or `demo.ts`, which both import this.
 */

import type { LineString, MultiLineString, MultiPolygon, Polygon } from "geojson";

import raw from "../../../api/app/seed/demo_ranch.json";

import type { LonLat } from "./types";

export interface RanchLicense {
  name: string;
  spdxId: string | null;
  url: string | null;
}

/** Where one layer of the ranch came from: the credit shown wherever it is drawn. */
export interface RanchSource {
  label: string;
  text: string;
  organization: string | null;
  url: string | null;
  license: RanchLicense | null;
}

export type DemoZoneId = "Z-14" | "Z-21" | "Z-08";

export interface DemoRanchZone {
  footprint: Polygon;
  areaM2: number;
  anchor: LonLat;
  /** Key into `sources`. */
  source: string;
  /** How the outline was made from its source, in a sentence. */
  basis: string;
}

export interface DemoRanchCamera {
  target: LonLat;
  /** Ellipsoid height of the ground at `target` (m). */
  targetHeightM: number;
  targetHeightSource: string;
  headingDeg: number;
  pitchDeg: number;
  rangeM: number;
}

export interface DemoRanch {
  slug: string;
  name: string;
  purpose: string;
  region: string;
  description: string;
  sources: Record<string, RanchSource>;
  boundary: { geometry: Polygon; source: string };
  features: {
    creek: {
      name: string | null;
      geometry: LineString | MultiLineString | Polygon;
      source: string;
      corridorWidthM: number;
    };
    oakWoodland: { geometry: Polygon | MultiPolygon; source: string };
    grassland: { geometry: Polygon | MultiPolygon; source: string };
    fence: { geometry: LineString; source: string; stripWidthM: number };
  };
  bases: Record<string, LonLat>;
  camera: DemoRanchCamera;
  derived: {
    generatedBy: string;
    siteAreaM2: number;
    centroid: LonLat;
    zones: Record<DemoZoneId, DemoRanchZone>;
    creekLengthM: number;
    fenceLengthM: number;
  };
}

const ZONE_IDS: readonly DemoZoneId[] = ["Z-14", "Z-21", "Z-08"];

type Json = Record<string, unknown>;

function fail(path: string, wanted: string): never {
  throw new Error(`demo_ranch.json: ${path} should be ${wanted}`);
}

function object(value: unknown, path: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value)) fail(path, "an object");
  return value as Json;
}

function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) fail(path, "a non-empty string");
  return value;
}

function textOrNull(value: unknown, path: string): string | null {
  return value === null || value === undefined ? null : text(value, path);
}

function number(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(path, "a finite number");
  return value;
}

function lonLat(value: unknown, path: string): LonLat {
  if (!Array.isArray(value) || value.length < 2) fail(path, "[longitude, latitude]");
  const longitude = number(value[0], `${path}[0]`);
  const latitude = number(value[1], `${path}[1]`);
  if (Math.abs(longitude) > 180 || Math.abs(latitude) > 90) fail(path, "a position on Earth");
  return { longitude, latitude };
}

function geometry<T extends { type: string }>(
  value: unknown,
  path: string,
  types: readonly T["type"][],
): T {
  const shape = object(value, path);
  if (!types.includes(shape.type as T["type"])) fail(`${path}.type`, types.join(" or "));
  if (!Array.isArray(shape.coordinates)) fail(`${path}.coordinates`, "an array");
  return shape as unknown as T;
}

function license(value: unknown, path: string): RanchLicense | null {
  if (value === null || value === undefined) return null;
  const json = object(value, path);
  return {
    name: text(json.name, `${path}.name`),
    spdxId: textOrNull(json.spdxId, `${path}.spdxId`),
    url: textOrNull(json.url, `${path}.url`),
  };
}

/** Parses the ranch file, throwing a message that names the field when its shape is wrong. */
export function readDemoRanch(document: unknown): DemoRanch {
  const json = object(document, "the file");
  const sourcesJson = object(json.sources, "sources");
  const sources: Record<string, RanchSource> = {};
  for (const [id, value] of Object.entries(sourcesJson)) {
    const source = object(value, `sources.${id}`);
    sources[id] = {
      label: text(source.label, `sources.${id}.label`),
      text: text(source.text, `sources.${id}.text`),
      organization: textOrNull(source.organization, `sources.${id}.organization`),
      url: textOrNull(source.url, `sources.${id}.url`),
      license: license(source.license, `sources.${id}.license`),
    };
  }
  const sourceId = (value: unknown, path: string): string => {
    const id = text(value, path);
    if (!(id in sources)) fail(path, `one of ${Object.keys(sources).join(", ")}`);
    return id;
  };
  const boundary = object(json.boundary, "boundary");
  const features = object(json.features, "features");
  const creek = object(features.creek, "features.creek");
  const oak = object(features.oakWoodland, "features.oakWoodland");
  const grass = object(features.grassland, "features.grassland");
  const fence = object(features.fence, "features.fence");
  const camera = object(json.camera, "camera");
  const derived = object(json.derived, "derived");
  const zonesJson = object(derived.zones, "derived.zones");
  const zones = {} as Record<DemoZoneId, DemoRanchZone>;
  for (const id of ZONE_IDS) {
    const zone = object(zonesJson[id], `derived.zones.${id}`);
    zones[id] = {
      footprint: geometry<Polygon>(zone.footprint, `derived.zones.${id}.footprint`, ["Polygon"]),
      areaM2: number(zone.areaM2, `derived.zones.${id}.areaM2`),
      anchor: lonLat(zone.anchor, `derived.zones.${id}.anchor`),
      source: sourceId(zone.source, `derived.zones.${id}.source`),
      basis: text(zone.basis, `derived.zones.${id}.basis`),
    };
  }
  const bases: Record<string, LonLat> = {};
  for (const [name, value] of Object.entries(object(json.bases, "bases"))) {
    bases[name] = lonLat(value, `bases.${name}`);
  }
  return {
    slug: text(json.slug, "slug"),
    name: text(json.name, "name"),
    purpose: text(json.purpose, "purpose"),
    region: text(json.region, "region"),
    description: text(json.description, "description"),
    sources,
    boundary: {
      geometry: geometry<Polygon>(boundary.geometry, "boundary.geometry", ["Polygon"]),
      source: sourceId(boundary.source, "boundary.source"),
    },
    features: {
      creek: {
        name: textOrNull(creek.name, "features.creek.name"),
        geometry: geometry<LineString | MultiLineString | Polygon>(
          creek.geometry,
          "features.creek.geometry",
          ["LineString", "MultiLineString", "Polygon"],
        ),
        source: sourceId(creek.source, "features.creek.source"),
        corridorWidthM: number(creek.corridorWidthM, "features.creek.corridorWidthM"),
      },
      oakWoodland: {
        geometry: geometry<Polygon | MultiPolygon>(oak.geometry, "features.oakWoodland.geometry", [
          "Polygon",
          "MultiPolygon",
        ]),
        source: sourceId(oak.source, "features.oakWoodland.source"),
      },
      grassland: {
        geometry: geometry<Polygon | MultiPolygon>(grass.geometry, "features.grassland.geometry", [
          "Polygon",
          "MultiPolygon",
        ]),
        source: sourceId(grass.source, "features.grassland.source"),
      },
      fence: {
        geometry: geometry<LineString>(fence.geometry, "features.fence.geometry", ["LineString"]),
        source: sourceId(fence.source, "features.fence.source"),
        stripWidthM: number(fence.stripWidthM, "features.fence.stripWidthM"),
      },
    },
    bases,
    camera: {
      target: lonLat(camera.target, "camera.target"),
      targetHeightM: number(camera.targetHeightM, "camera.targetHeightM"),
      targetHeightSource: text(camera.targetHeightSource, "camera.targetHeightSource"),
      headingDeg: number(camera.headingDeg, "camera.headingDeg"),
      pitchDeg: number(camera.pitchDeg, "camera.pitchDeg"),
      rangeM: number(camera.rangeM, "camera.rangeM"),
    },
    derived: {
      generatedBy: text(derived.generatedBy, "derived.generatedBy"),
      siteAreaM2: number(derived.siteAreaM2, "derived.siteAreaM2"),
      centroid: lonLat(derived.centroid, "derived.centroid"),
      zones,
      creekLengthM: number(derived.creekLengthM, "derived.creekLengthM"),
      fenceLengthM: number(derived.fenceLengthM, "derived.fenceLengthM"),
    },
  };
}

export const DEMO_RANCH: DemoRanch = readDemoRanch(raw);
export const DEMO_RANCH_SLUG = DEMO_RANCH.slug;
/** The offline copy's id: the built-in catalog uses `builtin-` ids (`api/fallback.ts`). */
export const BUILTIN_RANCH_ID = "builtin-demo-ranch";
export const DEMO_RANCH_PROJECT_ID = "demo-ranch";

/** The credit for one of the ranch's sources ("© ESA WorldCover project 2021 / …"). */
export function sourceText(id: string): string {
  return DEMO_RANCH.sources[id]?.text ?? id;
}
