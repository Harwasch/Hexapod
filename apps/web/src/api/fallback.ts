/**
 * Built-in catalog used when the API is unreachable.
 *
 * This mirrors the database seed (apps/api/app/seed/data.py) for the public
 * Cesium demo and a minimal open-data stack, so the globe stays useful and
 * the demo still works. The UI labels this state explicitly.
 */

import type { Layer, Site, SiteSummary } from "@twin/contracts";
import { destination, footprintAreaM2 } from "@twin/geo";

import { env } from "@/app/env";

export const DEMO_SITE_SLUG = "cesium-splat-demo";
export const DEMO_SPLAT_ASSET_ID = 4547222;
const DEMO_CENTER = { longitude: -122.13810992689156, latitude: 47.644519699638366 };
const DEMO_HEIGHT = 120;
const NOW = "2026-09-15T00:00:00Z";

function overviewBookmark() {
  const range = 500;
  const pitch = -25;
  const heading = 100;
  const horizontal = range * Math.cos((-pitch * Math.PI) / 180);
  const vertical = range * Math.sin((-pitch * Math.PI) / 180);
  const eye = destination(DEMO_CENTER, heading + 180, horizontal);
  return {
    id: "builtin-bookmark-overview",
    siteId: "builtin-demo-site",
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

const cesiumAttribution = {
  text: "Cesium sample data",
  organization: "Cesium GS, Inc.",
  url: "https://cesium.com/",
};

function envAsset(
  id: string,
  name: string,
  representation: Site["assets"][number]["representation"],
  assetId: number | undefined,
): Site["assets"][number] | null {
  if (!assetId) return null;
  return {
    id,
    siteId: "builtin-demo-site",
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

export function builtinDemoSite(): Site {
  const footprint = DEMO_BOUNDARY;
  const assets: Site["assets"] = [
    {
      id: "builtin-demo-splat",
      siteId: "builtin-demo-site",
      provider: "cesium-ion",
      name: "Gaussian splat (LOD)",
      representation: "gaussian-splat",
      source: { type: "cesium-ion", assetId: env.defaultSplatAssetId ?? DEMO_SPLAT_ASSET_ID },
      footprint,
      observedAt: null,
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [cesiumAttribution],
      provenance: {
        sourceOrganization: "Cesium GS, Inc.",
        sourceUrl: "https://sandcastle.cesium.com/?id=3d-tiles-gaussian-splats-with-lod",
        notes: "Sample asset referenced by the official CesiumJS Sandcastle.",
      },
      renderConfig: {
        maximumScreenSpaceError: 16,
        pointCloudShading: null,
        clipsWorld: true,
        clipFootprint: "tileset",
        heightOffsetM: 0,
      },
      defaultVisible: true,
      createdAt: NOW,
      updatedAt: NOW,
    },
  ];
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
    id: "builtin-demo-site",
    slug: DEMO_SITE_SLUG,
    name: "Cesium Gaussian splat demo",
    description:
      "Public 3D Gaussian splat tileset with hierarchical level of detail, published by Cesium for the '3D Tiles Gaussian splats with LOD' Sandcastle.",
    boundary: footprint,
    centroid: { ...DEMO_CENTER, height: DEMO_HEIGHT },
    areaM2: footprintAreaM2(footprint),
    thumbnailUrl: null,
    metadata: {
      quality: { resolutionDescription: "Sub-decimetre splat detail (visual)" },
      origin: "builtin",
    },
    attribution: [cesiumAttribution],
    license: {
      name: "Cesium ion sample asset",
      url: "https://cesium.com/legal/terms-of-service/",
      requiresAttribution: true,
      spdxId: null,
      notes: "Provided by Cesium for evaluation. Access depends on the ion token in use.",
    },
    assets,
    cameraBookmarks: [overviewBookmark()],
    createdAt: NOW,
    updatedAt: NOW,
  };
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
