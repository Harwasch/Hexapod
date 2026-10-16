/**
 * The catalog used when the API is unreachable.
 *
 * **What "offline" means changed in A9, because what is reachable changed.** This file
 * used to hold nothing but Cesium ion asset ids, and that was not an oversight: a
 * capture's tiles were a `StaticFiles` mount *on the API*, so an API that was down took
 * its tiles down with it. Listing a local capture here would have produced a site whose
 * every asset 404s — a worse answer than not listing it.
 *
 * Now the tiles live in object storage, behind a different host that does not go down
 * with this API. So offline splits into two questions that used to have one answer:
 *
 *  * **are the bytes reachable?** Yes — the bucket, or the CDN in front of it, is
 *    independent of the catalog service.
 *  * **is the list of captures knowable?** Only if something published it, because that
 *    list lives in the database, which is exactly what is unreachable. So
 *    `app/seed/publish.py` writes `catalog.json` next to the tiles, and this module
 *    fetches it. It is data published by the same step that publishes the tiles, not a
 *    table someone keeps in step by hand.
 *
 * What stays hard-coded is what is true with no deployment at all: the demo ranch the
 * simulated fleet works (read from the same file the API seeds it from,
 * `missions/demoRanchData.ts`), the Redmond scan showcase, and the open-data layer stack
 * (mirroring apps/api/app/seed/data.py), so a fresh checkout with no bucket still shows a
 * globe with something on it. The scan showcase's own Cesium ion splat is withdrawn there
 * (the account's ion token cannot read it); `VITE_DEFAULT_SPLAT_ASSET_ID` gives it one again.
 * The UI labels this whole state explicitly, and `builtin: true` keeps every write form
 * disabled — an offline capture is readable, never editable.
 */

import type { Layer, Site, SiteSummary } from "@twin/contracts";
import { destination, footprintAreaM2 } from "@twin/geo";

import { env } from "@/app/env";
import { eyeOf } from "@/missions/framing";
import { BUILTIN_RANCH_ID, DEMO_RANCH } from "@/missions/demoRanchData";

/** The Redmond scan showcase: Cesium's Gaussian splat Sandcastle site. */
export const SCAN_SHOWCASE_SLUG = "cesium-splat-demo";
export const BUILTIN_SCAN_SITE_ID = "builtin-demo-site";
const DEMO_CENTER = { longitude: -122.13810992689156, latitude: 47.644519699638366 };
const DEMO_HEIGHT = 120;
const NOW = "2026-09-15T00:00:00Z";

/** The seed's Overview (apps/api/app/seed/data.py, default_bookmark_for_demo). */
function overviewBookmark() {
  const range = 420;
  const pitch = -45;
  const heading = 100;
  const horizontal = range * Math.cos((-pitch * Math.PI) / 180);
  const vertical = range * Math.sin((-pitch * Math.PI) / 180);
  const eye = destination(DEMO_CENTER, heading + 180, horizontal);
  return {
    id: "builtin-bookmark-overview",
    siteId: BUILTIN_SCAN_SITE_ID,
    name: "Overview",
    longitude: eye.longitude,
    latitude: eye.latitude,
    height: DEMO_HEIGHT + vertical,
    heading,
    pitch,
    roll: 0,
    isDefault: true,
    createdAt: NOW,
  };
}

function envAsset(
  id: string,
  name: string,
  representation: Site["assets"][number]["representation"],
  assetId: number | undefined,
): Site["assets"][number] | null {
  if (!assetId) return null;
  return {
    id,
    siteId: BUILTIN_SCAN_SITE_ID,
    provider: "cesium-ion",
    name,
    representation,
    source: { type: "cesium-ion", assetId },
    footprint: null,
    observedAt: null,
    validFrom: null,
    validTo: null,
    resolution: null,
    crs: null,
    license: null,
    attribution: [],
    provenance: {
      sourceOrganization: "Configured via environment",
      notes: "VITE_DEFAULT_*_ASSET_ID",
    },
    renderConfig: {
      maximumScreenSpaceError: 16,
      pointCloudShading: null,
      clipsWorld: true,
      clipFootprint: "tileset",
      heightOffsetM: 0,
    },
    defaultVisible: false,
    sidecarFlags: [],
    createdAt: NOW,
    updatedAt: NOW,
  };
}

/** Horizontal extent of the demo tileset's root bounding box (~1.7 km × 2.8 km). */
const DEMO_BOUNDARY = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-122.143272, 47.635222],
      [-122.120532, 47.635222],
      [-122.120526, 47.660476],
      [-122.143272, 47.660476],
      [-122.143272, 47.635222],
    ],
  ],
};

export function builtinScanSite(): Site {
  const footprint = DEMO_BOUNDARY;
  const assets: Site["assets"] = [];
  const splat = envAsset(
    "builtin-demo-splat",
    "Gaussian splat (configured)",
    "gaussian-splat",
    env.defaultSplatAssetId,
  );
  if (splat) assets.push({ ...splat, defaultVisible: true });
  const mesh = envAsset("builtin-demo-mesh", "Mesh (configured)", "mesh", env.defaultMeshAssetId);
  const points = envAsset(
    "builtin-demo-points",
    "Point cloud (configured)",
    "point-cloud",
    env.defaultPointCloudAssetId,
  );
  if (mesh) assets.push(mesh);
  if (points) assets.push(points);
  return {
    id: BUILTIN_SCAN_SITE_ID,
    slug: SCAN_SHOWCASE_SLUG,
    name: "Cesium Gaussian splat demo",
    description:
      "Cesium's '3D Tiles Gaussian splats with LOD' Sandcastle site in Redmond, WA: the 3D-scan showcase. Its splat (Cesium ion 4547222) is withdrawn in this deployment; VITE_DEFAULT_SPLAT_ASSET_ID or adding the asset from the ion Asset Depot brings it back.",
    boundary: footprint,
    centroid: { ...DEMO_CENTER, height: DEMO_HEIGHT },
    areaM2: footprintAreaM2(footprint),
    thumbnailUrl: null,
    metadata: { origin: "builtin" },
    attribution: [],
    license: null,
    assets,
    cameraBookmarks: [overviewBookmark()],
    createdAt: NOW,
    updatedAt: NOW,
  };
}

/**
 * The demo ranch as the API seeds it (apps/api/app/seed/demo_ranch.py), from the same file:
 * the same outline, area, credits and Overview, so offline and online agree to the acre.
 */
export function builtinRanchSite(): Site {
  const ranch = DEMO_RANCH;
  const camera = ranch.camera;
  const eye = eyeOf(
    {
      target: camera.target,
      headingDeg: camera.headingDeg,
      pitchDeg: camera.pitchDeg,
      rangeM: camera.rangeM,
    },
    camera.targetHeightM,
  );
  const used = new Set([
    ranch.boundary.source,
    ...Object.values(ranch.features).map((feature) => feature.source),
  ]);
  const sources = Object.entries(ranch.sources).filter(([id]) => used.has(id));
  return {
    id: BUILTIN_RANCH_ID,
    slug: ranch.slug,
    name: ranch.name,
    description: ranch.description,
    boundary: ranch.boundary.geometry,
    centroid: { ...ranch.derived.centroid, height: null },
    // The number the API serves (PostGIS ST_Area on the spheroid), not a spherical estimate.
    areaM2: ranch.derived.siteAreaM2,
    thumbnailUrl: null,
    metadata: {
      origin: "demo-ranch",
      simulated: ["fleet", "zones", "plans", "work log"],
      sources: Object.fromEntries(sources),
    },
    attribution: sources.map(([, source]) => ({
      text: source.text,
      organization: source.organization,
      url: source.url,
    })),
    license: {
      name: "CC BY 4.0",
      spdxId: "CC-BY-4.0",
      url: "https://creativecommons.org/licenses/by/4.0/",
      requiresAttribution: true,
      notes:
        "Zone outlines derive from ESA WorldCover 2021 (CC BY 4.0) and the USGS National Hydrography Dataset (public domain). The outline is a demo area, and the fleet, zones, plans and work log are simulated.",
    },
    assets: [],
    cameraBookmarks: [
      {
        id: "builtin-ranch-bookmark-overview",
        siteId: BUILTIN_RANCH_ID,
        name: "Overview",
        ...eye,
        roll: 0,
        isDefault: true,
        createdAt: NOW,
      },
    ],
    createdAt: NOW,
    updatedAt: NOW,
  };
}

/** Every built-in site: the demo ranch first (the app's default demo), then the scan showcase. */
export function builtinSites(): Site[] {
  return [builtinRanchSite(), builtinScanSite()];
}

/** The built-in site with this id, or null. */
export function builtinSite(id: string): Site | null {
  return builtinSites().find((site) => site.id === id) ?? null;
}

export function toSummary(site: Site): SiteSummary {
  const observed = site.assets.map((a) => a.observedAt).filter((v): v is string => Boolean(v));
  return {
    id: site.id,
    slug: site.slug,
    name: site.name,
    description: site.description ?? null,
    centroid: site.centroid,
    areaM2: site.areaM2,
    thumbnailUrl: site.thumbnailUrl ?? null,
    representations: Array.from(new Set(site.assets.map((a) => a.representation))),
    latestObservedAt: observed.sort().at(-1) ?? null,
    quality: (site.metadata?.quality as SiteSummary["quality"]) ?? null,
    createdAt: site.createdAt,
    updatedAt: site.updatedAt,
  };
}

const world = { west: -180, south: -90, east: 180, north: 90 };
const baseLayer = {
  description: null,
  temporalExtent: null,
  observedAt: null,
  legend: null,
  provenance: null,
  builtin: true,
  createdAt: NOW,
  updatedAt: NOW,
  render: {
    opacity: 1,
    minimumAltitudeM: null,
    maximumAltitudeM: null,
    maximumScreenSpaceError: null,
    exclusiveGroup: null,
  },
};

export function builtinLayers(): Layer[] {
  return [
    {
      ...baseLayer,
      id: "builtin-terrain",
      slug: "cesium-world-terrain",
      name: "Cesium World Terrain",
      category: "terrain",
      sourceType: "cesium-ion-terrain",
      source: {
        type: "cesium-ion-terrain",
        assetId: 1,
        requestVertexNormals: true,
        requestWaterMask: false,
      },
      spatialExtent: world,
      resolution: "~30 m global",
      coverage: "Global",
      render: { ...baseLayer.render, exclusiveGroup: "terrain" },
      attribution: [
        {
          text: "Cesium World Terrain",
          organization: "Cesium GS, Inc.",
          url: "https://cesium.com/",
        },
      ],
      license: {
        name: "Cesium ion Terms of Service",
        url: "https://cesium.com/legal/terms-of-service/",
        requiresAttribution: true,
        spdxId: null,
        notes: null,
      },
      defaultVisible: true,
    },
    {
      ...baseLayer,
      id: "builtin-bing",
      slug: "bing-maps-aerial",
      name: "Bing Maps Aerial",
      category: "imagery",
      sourceType: "cesium-ion-imagery",
      source: { type: "cesium-ion-imagery", assetId: 2 },
      spatialExtent: world,
      resolution: "~0.3 m urban",
      coverage: "Global",
      render: { ...baseLayer.render, exclusiveGroup: "basemap" },
      attribution: [
        {
          text: "© Microsoft, Bing Maps",
          organization: "Microsoft",
          url: "https://www.bing.com/maps",
        },
      ],
      license: {
        name: "Cesium ion Terms of Service",
        url: "https://cesium.com/legal/terms-of-service/",
        requiresAttribution: true,
        spdxId: null,
        notes: null,
      },
      defaultVisible: true,
    },
    {
      ...baseLayer,
      id: "builtin-osm",
      slug: "openstreetmap",
      name: "OpenStreetMap",
      category: "imagery",
      sourceType: "xyz",
      source: {
        type: "xyz",
        urlTemplate: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        minimumLevel: 0,
        maximumLevel: 19,
        subdomains: null,
        tileWidth: 256,
      },
      spatialExtent: world,
      resolution: "Zoom 19",
      coverage: "Global",
      render: { ...baseLayer.render, exclusiveGroup: "basemap" },
      attribution: [
        {
          text: "© OpenStreetMap contributors",
          organization: "OpenStreetMap Foundation",
          url: "https://www.openstreetmap.org/copyright",
        },
      ],
      license: {
        name: "ODbL 1.0",
        spdxId: "ODbL-1.0",
        url: "https://operations.osmfoundation.org/policies/tiles/",
        requiresAttribution: true,
        notes: null,
      },
      defaultVisible: false,
    },
  ];
}

/**
 * The published catalog: every site the API knew about when `catalog.json` was last
 * written, reachable while the API is not.
 *
 * Parsed defensively rather than cast. It is JSON from a bucket, written by a different
 * process at a different time, and the one thing worse than showing no captures offline
 * would be crashing the globe over a stale document — so anything unrecognisable yields
 * an empty list and the built-in sites stand alone.
 */
export async function fetchOfflineCatalog(url: string, signal?: AbortSignal): Promise<Site[]> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`offline catalog: ${String(response.status)}`);
  return readOfflineCatalog(await response.json());
}

/** The parsing half of {@link fetchOfflineCatalog}, separated so it can be tested directly. */
export function readOfflineCatalog(document: unknown): Site[] {
  if (typeof document !== "object" || document === null) return [];
  const sites: unknown = (document as { sites?: unknown }).sites;
  if (!Array.isArray(sites)) return [];
  return sites.filter(isSiteish);
}

function isSiteish(value: unknown): value is Site {
  if (typeof value !== "object" || value === null) return false;
  const site = value as Partial<Site>;
  return (
    typeof site.id === "string" &&
    typeof site.slug === "string" &&
    typeof site.name === "string" &&
    Array.isArray(site.assets) &&
    typeof site.boundary === "object" &&
    site.boundary !== null &&
    typeof site.centroid === "object" &&
    site.centroid !== null
  );
}
