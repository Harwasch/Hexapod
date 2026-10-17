/**
 * The ground a dress rehearsal reads (`missions/rehearsal.ts`), gathered when "Rehearse" is
 * pressed: what the slope comes from, where the creeks are, and how high the ghosts stand.
 *
 * Slope, best first: the plan's Land's newest USGS 3DEP bare-earth raster for its current
 * outline (10 m, trees and buildings left out, which machine limits need); else its Copernicus
 * GLO-30 raster, a 30 m surface model with the canopy in it; else the globe's own terrain
 * sampled on a grid over the lanes; else none, and the rehearsal says slope was not checked.
 * Creeks: the USGS NHD flowlines the API reads around the Land (`GET /land/{id}/hydrography`);
 * on the simulated demo ranch without a saved Land, its supplied Figueroa Creek line (also
 * from NHD). The rasters' heights are NAVD88 or EGM2008, not the ellipsoid Cesium draws on, so
 * ghost heights take the raster plus the median difference to the globe's terrain at a few
 * points; slope is a difference of heights, which the datum does not change.
 */

import { Cartographic, EllipsoidTerrainProvider, sampleTerrainMostDetailed } from "cesium";

import type { components } from "@twin/contracts";
import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";

import { api, unwrap } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { rawValues } from "@/features/knowledge/grid";
import { DEMO_RANCH, DEMO_RANCH_PROJECT_ID } from "@/missions/demoRanchData";
import {
  elevationAt,
  type CreekLine,
  type CreekSource,
  type ElevationGrid,
  type ElevationSource,
} from "@/missions/rehearsal";

type LandRasterRead = components["schemas"]["LandRasterRead"];

/** How long a terrain sample may take before the rehearsal goes on without it, ms. */
const TERRAIN_WAIT_MS = 10_000;
/** Cells a side, at most, of the globe-terrain grid sampled when no raster is at hand. */
const TERRAIN_MAX_CELLS = 64;
/** Its cells are never finer than this, metres. */
const TERRAIN_MIN_CELL_M = 15;

export interface RehearsalGround {
  elevation: ElevationSource | null;
  elevationMissing?: string;
  creeks: CreekSource | null;
  creeksMissing?: string;
  /** Ellipsoid height of the ground at a point, for the ghosts and hands to stand on. */
  heightAt: (longitude: number, latitude: number) => number;
  /** A better slope source the Land could have, said beside the result; null when none. */
  better: string | null;
}

function within<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  return Promise.race([
    promise,
    new Promise<undefined>((resolve) => window.setTimeout(() => resolve(undefined), ms)),
  ]);
}

function median(values: readonly number[]): number | null {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const mid = sorted.length >> 1;
  return sorted.length % 2
    ? (sorted[mid] ?? null)
    : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
}

/** The newest raster of a dataset made for the Land's current outline. */
export function freshestRaster(
  rasters: readonly LandRasterRead[],
  dataset: "usgs-3dep-13" | "cop-dem-glo-30",
): LandRasterRead | null {
  return (
    rasters
      .filter((r) => (r.request.dataset ?? "cop-dem-glo-30") === dataset && !r.stale)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0] ?? null
  );
}

function elevationLabel(raster: LandRasterRead): string {
  const metres = Math.round(raster.metadata.resolutionM);
  return raster.request.dataset === "usgs-3dep-13"
    ? `USGS 3DEP 1/3″ bare-earth DEM, ${metres} m`
    : `Copernicus GLO-30, a ${metres} m surface model incl. canopy`;
}

async function rasterElevation(
  landId: string,
): Promise<{ source: ElevationSource; raster: LandRasterRead } | null> {
  const rasters = await unwrap<LandRasterRead[]>(
    api.GET("/api/v1/land/{land_id}/rasters", {
      params: { path: { land_id: landId }, query: { limit: 50 } },
    }),
  );
  const raster =
    freshestRaster(rasters, "usgs-3dep-13") ?? freshestRaster(rasters, "cop-dem-glo-30");
  if (!raster) return null;
  const grid = await unwrap(
    api.GET("/api/v1/land/rasters/{raster_id}/grid", {
      params: { path: { raster_id: raster.id }, query: { band: 1 } },
    }),
  );
  const values = rawValues(grid);
  return {
    raster,
    source: {
      grid: {
        west: grid.west,
        south: grid.south,
        east: grid.east,
        north: grid.north,
        columns: grid.columns,
        rows: grid.rows,
        values: values instanceof Float32Array ? values : Float32Array.from(values),
      },
      label: elevationLabel(raster),
      ref: `raster ${raster.id.slice(0, 8)}`,
    },
  };
}

async function landCreeks(
  landId: string,
): Promise<{ creeks: CreekSource | null; missing?: string }> {
  const answer = await unwrap(
    api.GET("/api/v1/land/{land_id}/hydrography", { params: { path: { land_id: landId } } }),
  );
  // "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000"
  const scale = answer.source.resolution.split(/[;(]/)[0]?.trim();
  const label = scale ? `${answer.source.name}, ${scale}` : answer.source.name;
  if (answer.status === "available" || answer.status === "empty") {
    return {
      creeks: {
        label,
        lines: answer.features.map((feature) => ({
          id: feature.permanentIdentifier,
          name: feature.name,
          kind: feature.kind,
          coordinates: feature.geometry.coordinates.map(([longitude, latitude]) => ({
            longitude: longitude ?? 0,
            latitude: latitude ?? 0,
          })),
        })),
      },
    };
  }
  return { creeks: null, missing: `Creek buffer not checked: ${answer.message}` };
}

/** The demo ranch's supplied creek (USGS NHD, Figueroa Creek), for the ranch without a Land. */
function demoCreeks(): CreekSource {
  const creek = DEMO_RANCH.features.creek;
  const geometry = creek.geometry;
  const parts: number[][][] =
    geometry.type === "LineString"
      ? [geometry.coordinates]
      : geometry.type === "MultiLineString"
        ? geometry.coordinates
        : geometry.coordinates;
  const source = DEMO_RANCH.sources[creek.source];
  const lines: CreekLine[] = parts.map((coordinates, index) => ({
    id: `demo-creek-${index + 1}`,
    name: creek.name,
    // The supplied line does not carry NHD's permanence; a channel of unknown permanence counts.
    kind: "unclassified",
    coordinates: coordinates.map(([longitude, latitude]) => ({
      longitude: longitude ?? 0,
      latitude: latitude ?? 0,
    })),
  }));
  return { label: `${source?.label ?? "USGS NHD"} (demo ranch data)`, lines };
}

function boundsOf(points: readonly LonLat[], padM: number) {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const p of points) {
    west = Math.min(west, p.longitude);
    east = Math.max(east, p.longitude);
    south = Math.min(south, p.latitude);
    north = Math.max(north, p.latitude);
  }
  const midLat = (south + north) / 2;
  const padLon = padM / (Math.cos(midLat * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M);
  const padLat = padM / (DEG_TO_RAD * EARTH_RADIUS_M);
  return { west: west - padLon, south: south - padLat, east: east + padLon, north: north + padLat };
}

function hasTerrain(scene: CesiumSceneManager): boolean {
  return !(scene.viewer.terrainProvider instanceof EllipsoidTerrainProvider);
}

async function terrainHeights(
  scene: CesiumSceneManager,
  points: readonly LonLat[],
): Promise<(number | undefined)[] | null> {
  if (!hasTerrain(scene) || points.length === 0) return null;
  const samples = await within(
    sampleTerrainMostDetailed(
      scene.viewer.terrainProvider,
      points.map((p) => Cartographic.fromDegrees(p.longitude, p.latitude)),
    ).catch(() => undefined),
    TERRAIN_WAIT_MS,
  );
  if (!samples) return null;
  const heights = samples.map((s) =>
    Number.isFinite(s.height) && s.height > -500 && s.height < 9000 ? s.height : undefined,
  );
  return heights.some((h) => h !== undefined) ? heights : null;
}

/** The globe's terrain on a grid over the lanes: the last slope source, labelled as such. */
async function terrainGrid(
  scene: CesiumSceneManager,
  points: readonly LonLat[],
): Promise<ElevationSource | null> {
  if (!hasTerrain(scene) || points.length === 0) return null;
  const box = boundsOf(points, 40);
  const midLat = (box.south + box.north) / 2;
  const widthM =
    (box.east - box.west) * Math.cos(midLat * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
  const heightM = (box.north - box.south) * DEG_TO_RAD * EARTH_RADIUS_M;
  const cellM = Math.max(TERRAIN_MIN_CELL_M, Math.max(widthM, heightM) / TERRAIN_MAX_CELLS);
  const columns = Math.max(2, Math.ceil(widthM / cellM));
  const rows = Math.max(2, Math.ceil(heightM / cellM));
  const dlon = (box.east - box.west) / columns;
  const dlat = (box.north - box.south) / rows;
  const centres: LonLat[] = [];
  for (let row = 0; row < rows; row++)
    for (let col = 0; col < columns; col++)
      centres.push({
        longitude: box.west + (col + 0.5) * dlon,
        latitude: box.north - (row + 0.5) * dlat,
      });
  const heights = await terrainHeights(scene, centres);
  if (!heights) return null;
  const grid: ElevationGrid = {
    ...box,
    columns,
    rows,
    values: Float32Array.from(heights, (h) => h ?? Number.NaN),
  };
  return {
    grid,
    label: `the globe's terrain (Cesium), sampled ${Math.round(cellM)} m apart`,
    ref: null,
  };
}

/** Heights for ghosts: the raster's on the ellipsoid, through the globe's terrain at a few points. */
async function standOn(
  scene: CesiumSceneManager,
  elevation: ElevationSource | null,
  ellipsoidal: boolean,
  points: readonly LonLat[],
): Promise<(longitude: number, latitude: number) => number> {
  if (!elevation) return () => 0;
  const grid = elevation.grid;
  const typical = median(Array.from(grid.values)) ?? 0;
  let offset = 0;
  if (!ellipsoidal) {
    const step = Math.max(1, Math.floor(points.length / 9));
    const probes = points.filter((_, i) => i % step === 0).slice(0, 9);
    const heights = await terrainHeights(scene, probes);
    const differences = (heights ?? []).flatMap((h, i) => {
      const p = probes[i];
      const z = p ? elevationAt(grid, p.longitude, p.latitude) : Number.NaN;
      return h !== undefined && Number.isFinite(z) ? [h - z] : [];
    });
    offset = median(differences) ?? 0;
  }
  return (longitude, latitude) => {
    const z = elevationAt(grid, longitude, latitude);
    return (Number.isFinite(z) ? z : typical) + offset;
  };
}

/** Everything the rehearsal reads, for the lanes it will drive. */
export async function rehearsalGround(
  scene: CesiumSceneManager,
  options: { landId: string | null; projectId: string; points: readonly LonLat[] },
): Promise<RehearsalGround> {
  const { landId, projectId, points } = options;
  let elevation: ElevationSource | null = null;
  let elevationMissing: string | undefined;
  let better: string | null = null;
  let creeks: CreekSource | null = null;
  let creeksMissing: string | undefined;
  if (landId) {
    const [raster, water] = await Promise.all([
      rasterElevation(landId).catch(() => null),
      landCreeks(landId).catch((error: unknown) => ({
        creeks: null,
        missing: `Creek buffer not checked: USGS NHD flowlines could not be read (${error instanceof Error ? error.message : "no answer"}). Try again.`,
      })),
    ]);
    elevation = raster?.source ?? null;
    creeks = water.creeks;
    creeksMissing = water.missing;
    if (raster?.raster.request.dataset !== "usgs-3dep-13")
      better =
        "For 10 m bare-earth slope, analyze bare-earth terrain (USGS 3DEP) for this land in Land, then rehearse again.";
  } else if (projectId === DEMO_RANCH_PROJECT_ID) {
    creeks = demoCreeks();
    better =
      "Open the Oak Bench Ranch land in Land to rehearse against its own rasters and USGS NHD flowlines.";
  } else {
    creeksMissing =
      "Creek buffer not checked: creeks come from USGS NHD around a saved Land, and this ground is not on one.";
  }
  const ellipsoidal = !elevation;
  if (!elevation) {
    elevation = await terrainGrid(scene, points);
    if (!elevation)
      elevationMissing = landId
        ? "Slope not checked: this land has no elevation raster yet and the globe has no terrain here."
        : "Slope not checked: no terrain for this ground.";
  }
  const heightAt = await standOn(scene, elevation, ellipsoidal, points);
  return { elevation, elevationMissing, creeks, creeksMissing, heightAt, better };
}
