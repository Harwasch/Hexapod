import { test as base, expect, type Page } from "@playwright/test";

/** Mirrors the API seed closely enough for the UI; keeps e2e independent of PostGIS. */
export const demoSite = {
  id: "11111111-1111-4111-8111-111111111111",
  slug: "cesium-splat-demo",
  name: "Cesium Gaussian splat demo",
  description: "Public 3D Gaussian splat tileset with hierarchical level of detail.",
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-122.143272, 47.635222],
        [-122.120532, 47.635222],
        [-122.120526, 47.660476],
        [-122.143272, 47.660476],
        [-122.143272, 47.635222],
      ],
    ],
  },
  centroid: { longitude: -122.13810992689156, latitude: 47.644519699638366, height: 120 },
  areaM2: 4_797_797,
  thumbnailUrl: null,
  metadata: { quality: { resolutionDescription: "Sub-decimetre splat detail (visual)" } },
  attribution: [
    { text: "Cesium sample data", organization: "Cesium GS, Inc.", url: "https://cesium.com/" },
  ],
  license: {
    name: "Cesium ion sample asset",
    spdxId: null,
    url: "https://cesium.com/legal/terms-of-service/",
    requiresAttribution: true,
    notes: null,
  },
  assets: [
    {
      id: "22222222-2222-4222-8222-222222222222",
      siteId: "11111111-1111-4111-8111-111111111111",
      provider: "cesium-ion",
      name: "Gaussian splat (LOD)",
      representation: "gaussian-splat",
      source: { type: "cesium-ion", assetId: 4547222 },
      footprint: null,
      observedAt: null,
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [
        { text: "Cesium sample data", organization: "Cesium GS, Inc.", url: "https://cesium.com/" },
      ],
      provenance: null,
      renderConfig: {
        maximumScreenSpaceError: 16,
        pointCloudShading: null,
        clipsWorld: true,
        clipFootprint: "tileset",
        heightOffsetM: 0,
      },
      defaultVisible: true,
      createdAt: "2026-09-15T00:00:00Z",
      updatedAt: "2026-09-15T00:00:00Z",
    },
    {
      id: "33333333-3333-4333-8333-333333333333",
      siteId: "11111111-1111-4111-8111-111111111111",
      provider: "3d-tiles-url",
      name: "Mesh (test)",
      representation: "mesh",
      source: { type: "3d-tiles-url", url: "https://example.invalid/mesh/tileset.json" },
      footprint: null,
      observedAt: "2025-05-01T00:00:00Z",
      validFrom: null,
      validTo: null,
      resolution: null,
      crs: null,
      license: null,
      attribution: [],
      provenance: null,
      renderConfig: {
        maximumScreenSpaceError: 16,
        pointCloudShading: null,
        clipsWorld: true,
        clipFootprint: "catalog",
        heightOffsetM: 0,
      },
      defaultVisible: false,
      createdAt: "2026-09-15T00:00:00Z",
      updatedAt: "2026-09-15T00:00:00Z",
    },
  ],
  cameraBookmarks: [
    {
      id: "44444444-4444-4444-8444-444444444444",
      siteId: "11111111-1111-4111-8111-111111111111",
      name: "Overview",
      longitude: -122.14406,
      latitude: 47.64523,
      height: 331.3,
      heading: 100,
      pitch: -25,
      roll: 0,
      isDefault: true,
      createdAt: "2026-09-15T00:00:00Z",
    },
  ],
  createdAt: "2026-09-15T00:00:00Z",
  updatedAt: "2026-09-15T00:00:00Z",
};

export const demoSummary = {
  id: demoSite.id,
  slug: demoSite.slug,
  name: demoSite.name,
  description: demoSite.description,
  centroid: demoSite.centroid,
  areaM2: demoSite.areaM2,
  thumbnailUrl: null,
  representations: ["gaussian-splat", "mesh"],
  latestObservedAt: "2025-05-01T00:00:00Z",
  quality: {
    resolutionDescription: "Sub-decimetre splat detail (visual)",
    groundSampleDistanceM: null,
  },
  createdAt: demoSite.createdAt,
  updatedAt: demoSite.updatedAt,
};

const baseLayer = {
  description: null,
  temporalExtent: null,
  observedAt: null,
  legend: null,
  provenance: null,
  builtin: true,
  createdAt: "2026-09-15T00:00:00Z",
  updatedAt: "2026-09-15T00:00:00Z",
  render: {
    opacity: 1,
    minimumAltitudeM: null,
    maximumAltitudeM: null,
    maximumScreenSpaceError: null,
    exclusiveGroup: null,
  },
  spatialExtent: { west: -180, south: -90, east: 180, north: 90 },
  coverage: "Global",
  resolution: null,
  license: {
    name: "ODbL 1.0",
    spdxId: "ODbL-1.0",
    url: "https://opendatacommons.org/licenses/odbl/",
    requiresAttribution: true,
    notes: null,
  },
};

export const layers = [
  {
    ...baseLayer,
    id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    slug: "openstreetmap",
    name: "OpenStreetMap",
    description: "Community-maintained street map raster tiles.",
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
    render: { ...baseLayer.render, exclusiveGroup: "basemap" },
    attribution: [
      {
        text: "© OpenStreetMap contributors",
        organization: "OpenStreetMap Foundation",
        url: "https://www.openstreetmap.org/copyright",
      },
    ],
    defaultVisible: false,
  },
  {
    ...baseLayer,
    id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    slug: "esa-worldcover-2021",
    name: "ESA WorldCover 2021",
    description: "Global 10 m land-cover map.",
    category: "land-cover",
    sourceType: "wms",
    source: {
      type: "wms",
      url: "https://services.terrascope.be/wms/v2",
      layers: "WORLDCOVER_2021_MAP",
      parameters: { transparent: "true" },
    },
    observedAt: "2021-01-01T00:00:00Z",
    resolution: "10 m",
    legend: {
      title: "Land cover class",
      entries: [{ label: "Tree cover", color: "#006400" }],
      imageUrl: null,
    },
    attribution: [
      {
        text: "© ESA WorldCover project 2021",
        organization: "ESA",
        url: "https://esa-worldcover.org",
      },
    ],
    license: {
      name: "CC BY 4.0",
      spdxId: "CC-BY-4.0",
      url: "https://creativecommons.org/licenses/by/4.0/",
      requiresAttribution: true,
      notes: null,
    },
    defaultVisible: false,
  },
];

export interface MockOptions {
  /** Simulate the catalog API being down. */
  apiDown?: boolean;
  onCreateSite?: (body: unknown) => void;
}

export async function mockApi(page: Page, options: MockOptions = {}): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (options.apiDown) {
      await route.abort("connectionrefused");
      return;
    }
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (path === "/api/v1/health")
      return json({
        status: "ok",
        database: true,
        postgisVersion: "3.4",
        objectStorage: "none",
        objectStorageAvailable: false,
      });
    if (path === "/api/v1/sites" && request.method() === "GET") return json([demoSummary]);
    if (path === "/api/v1/sites" && request.method() === "POST") {
      const body = request.postDataJSON() as { name: string; boundary: unknown };
      options.onCreateSite?.(body);
      if (typeof body.name !== "string" || body.name.length < 2) {
        return json(
          {
            title: "Validation error",
            status: 422,
            errors: [{ loc: ["body", "name"], msg: "too short", type: "value_error" }],
          },
          422,
        );
      }
      return json(
        {
          ...demoSite,
          id: "55555555-5555-4555-8555-555555555555",
          slug: "new-site",
          name: body.name,
          assets: [],
          cameraBookmarks: [],
        },
        201,
      );
    }
    if (path === `/api/v1/sites/${demoSite.id}`) return json(demoSite);
    if (path === "/api/v1/layers" && request.method() === "GET") return json(layers);
    if (path === "/api/v1/layers" && request.method() === "POST") {
      const body = request.postDataJSON() as { name: string; source: { type: string } };
      return json(
        {
          ...layers[0],
          id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
          slug: "new-layer",
          name: body.name,
          sourceType: body.source.type,
          source: body.source,
          builtin: false,
          defaultVisible: false,
        },
        201,
      );
    }
    if (path === "/api/v1/ion/status")
      return json({
        configured: false,
        apiBase: "https://api.cesium.com",
        reconstruction: {
          monitorJobs: false,
          registerAssets: true,
          createJobs: false,
          createJobsReason: "sourceType for photo inputs is not documented",
        },
      });
    return json({ title: "Not found", status: 404 }, 404);
  });
  // Keep the globe offline-friendly: block the heavy external tile traffic so runs are deterministic.
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.route(/https:\/\/tile\.openstreetmap\.org\/.*/, (route) => route.abort());
  await page.route(/https:\/\/services\.terrascope\.be\/.*/, (route) => route.abort());
  await page.route(
    /https:\/\/(dev\.virtualearth|ecn\.t[0-9]\.tiles\.virtualearth)\.net\/.*/,
    (route) => route.abort(),
  );
}

export const test = base.extend<{ app: Page }>({
  app: async ({ page }, use) => {
    await mockApi(page);
    await page.addInitScript(() => {
      window.localStorage.setItem(
        "twin.settings.v1",
        JSON.stringify({
          state: { onboardingDismissed: false, quality: "performance" },
          version: 1,
        }),
      );
    });
    await page.goto("/");
    await expect(page.getByTestId("cesium-viewport")).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible();
    await expect(page.getByTestId("status-bar")).toContainText("Alt");
    await use(page);
  },
});

export { expect };
