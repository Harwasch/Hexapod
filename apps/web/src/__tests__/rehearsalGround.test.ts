import { EllipsoidTerrainProvider } from "cesium";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { components } from "@twin/contracts";

import { api } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { freshestRaster, rehearsalGround } from "@/features/mission/rehearsalGround";
import { DEMO_RANCH_PROJECT_ID } from "@/missions/demoRanchData";

type LandRasterRead = components["schemas"]["LandRasterRead"];
type LandHydrography = components["schemas"]["LandHydrography"];

function raster(
  id: string,
  dataset: "usgs-3dep-13" | "cop-dem-glo-30",
  createdAt: string,
  stale = false,
): LandRasterRead {
  return {
    id,
    landId: "land",
    runId: "run",
    boundaryRevision: 1,
    request: { dataset },
    sha256: "a".repeat(64),
    byteSize: 100,
    stale,
    createdAt,
    metadata: {
      algorithm: "x",
      method: "x",
      bounds: [-120.05, 34.69, -120.03, 34.71],
      crs: "EPSG:4326",
      resolutionM: dataset === "usgs-3dep-13" ? 10 : 30,
      width: 2,
      height: 2,
      boundaryCells: 4,
      validCells: 4,
      coverageFraction: 1,
      sampledAreaM2: 400,
      downloadedBytes: 1,
      warnings: [],
      sources: [],
      bands: [],
    },
  };
}

/** A 2 x 2 float32 grid, row 0 at the north, as the API sends it. */
const GRID = {
  west: -120.05,
  south: 34.69,
  east: -120.03,
  north: 34.71,
  columns: 2,
  rows: 2,
  cellM: 10,
  encoding: "f32" as const,
  scale: 1,
  offset: 0,
  nodata: null,
  data: Buffer.from(Float32Array.from([410, 420, 400, Number.NaN]).buffer).toString("base64"),
};

function hydrography(status: LandHydrography["status"] = "available"): LandHydrography {
  return {
    landId: "land",
    boundaryRevision: 1,
    status,
    message: status === "uncovered" ? "USGS NHD covers the United States only." : "Found.",
    bufferM: 100,
    truncated: false,
    retrievedAt: "2026-10-17T00:00:00Z",
    features:
      status === "available"
        ? [
            {
              id: "{ABC}",
              permanentIdentifier: "{ABC}",
              name: "Figueroa Creek",
              fcode: 46003,
              kind: "intermittent",
              description: "Stream/river, intermittent",
              lengthM: 100,
              geometry: {
                type: "LineString",
                coordinates: [
                  [-120.04, 34.7],
                  [-120.039, 34.701],
                ],
              },
            },
          ]
        : [],
    source: {
      provider: "usgs-nhd",
      name: "USGS National Hydrography Dataset (NHD) flowlines",
      url: "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query",
      linkUrl: "https://hydro.nationalmap.gov/",
      documentationUrl: "https://www.usgs.gov/",
      license: "Public domain",
      attribution: "USGS",
      resolution: "1:24,000 (Flowline - Large Scale); mapped channels, not surveyed banks",
    },
  };
}

function answer(rasters: LandRasterRead[], water: LandHydrography) {
  return vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/rasters") ? rasters : path.endsWith("/grid") ? GRID : water,
      response: new Response(),
    })) as typeof api.GET);
}

/** A globe without terrain: no samples, so heights and slope come from the rasters only. */
const flat = {
  viewer: { terrainProvider: new EllipsoidTerrainProvider() },
} as unknown as CesiumSceneManager;
const points = [
  { longitude: -120.045, latitude: 34.705 },
  { longitude: -120.035, latitude: 34.695 },
];

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the ground a rehearsal reads", () => {
  it("takes the newest fresh bare-earth raster, then the surface model", () => {
    const rasters = [
      raster("stale-3dep", "usgs-3dep-13", "2026-10-16T00:00:00Z", true),
      raster("old-3dep", "usgs-3dep-13", "2026-10-01T00:00:00Z"),
      raster("new-3dep", "usgs-3dep-13", "2026-10-10T00:00:00Z"),
      raster("cop", "cop-dem-glo-30", "2026-10-15T00:00:00Z"),
    ];
    expect(freshestRaster(rasters, "usgs-3dep-13")?.id).toBe("new-3dep");
    expect(freshestRaster(rasters, "cop-dem-glo-30")?.id).toBe("cop");
    expect(freshestRaster(rasters.slice(0, 1), "usgs-3dep-13")).toBeNull();
  });

  it("reads a Land's 3DEP grid and NHD creeks, cited", async () => {
    const get = answer(
      [
        raster("cop-1", "cop-dem-glo-30", "2026-10-15T00:00:00Z"),
        raster("3dep-12345678", "usgs-3dep-13", "2026-10-10T00:00:00Z"),
      ],
      hydrography(),
    );
    const ground = await rehearsalGround(flat, { landId: "land", projectId: "demo", points });
    expect(get).toHaveBeenCalledWith(
      "/api/v1/land/rasters/{raster_id}/grid",
      expect.objectContaining({
        params: { path: { raster_id: "3dep-12345678" }, query: { band: 1 } },
      }),
    );
    expect(ground.elevation).toMatchObject({
      label: "USGS 3DEP 1/3″ bare-earth DEM, 10 m",
      ref: "raster 3dep-123",
    });
    expect(Array.from(ground.elevation?.grid.values ?? []).slice(0, 3)).toEqual([410, 420, 400]);
    expect(ground.better).toBeNull();
    expect(ground.creeks?.label).toBe(
      "USGS National Hydrography Dataset (NHD) flowlines, 1:24,000",
    );
    expect(ground.creeks?.lines[0]).toMatchObject({
      id: "{ABC}",
      name: "Figueroa Creek",
      kind: "intermittent",
    });
    // No globe terrain to tie the datum to: the raster's own heights.
    expect(ground.heightAt(-120.0475, 34.705)).toBeCloseTo(410, 0);
  });

  it("falls back to the surface model, and says how to do better", async () => {
    answer([raster("cop-1", "cop-dem-glo-30", "2026-10-15T00:00:00Z")], hydrography("uncovered"));
    const ground = await rehearsalGround(flat, { landId: "land", projectId: "demo", points });
    expect(ground.elevation?.label).toBe("Copernicus GLO-30, a 30 m surface model incl. canopy");
    expect(ground.better).toContain("analyze bare-earth terrain (USGS 3DEP)");
    expect(ground.creeks).toBeNull();
    expect(ground.creeksMissing).toBe(
      "Creek buffer not checked: USGS NHD covers the United States only.",
    );
  });

  it("without a raster or terrain, says slope was not checked", async () => {
    answer([], hydrography("empty"));
    const ground = await rehearsalGround(flat, { landId: "land", projectId: "demo", points });
    expect(ground.elevation).toBeNull();
    expect(ground.elevationMissing).toContain("Slope not checked");
    expect(ground.creeks?.lines).toEqual([]);
    expect(ground.heightAt(-120.04, 34.7)).toBe(0);
  });

  it("on the demo ranch without a Land, uses its supplied NHD creek", async () => {
    const get = vi.spyOn(api, "GET");
    const ground = await rehearsalGround(flat, {
      landId: null,
      projectId: DEMO_RANCH_PROJECT_ID,
      points,
    });
    expect(get).not.toHaveBeenCalled();
    expect(ground.creeks?.label).toContain("(demo ranch data)");
    expect(ground.creeks?.lines[0]?.name).toBe("Figueroa Creek");
    expect(ground.elevationMissing).toBe("Slope not checked: no terrain for this ground.");
    const elsewhere = await rehearsalGround(flat, { landId: null, projectId: "site:x", points });
    expect(elsewhere.creeks).toBeNull();
    expect(elsewhere.creeksMissing).toContain("not on one");
  });
});
