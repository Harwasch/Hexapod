import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { LandRasterBridge } from "@/cesium/LandRasterBridge";
import { useLandContext } from "@/state/landContext";
import type * as IdentityModule from "@/state/landIdentity";
import { useLandIdentity } from "@/state/landIdentity";

const fixture = vi.hoisted(() => ({
  requests: [] as { url: { url: string; headers: Record<string, string> }; rectangle: number[] }[],
  add: vi.fn(),
  remove: vi.fn(),
  render: vi.fn(),
}));
vi.mock("cesium", () => ({
  Credit: class {
    constructor(public text: string) {}
  },
  GeographicTilingScheme: vi.fn(),
  TextureMagnificationFilter: { NEAREST: 9728 },
  TextureMinificationFilter: { NEAREST: 9728 },
  Rectangle: { fromDegrees: (...values: number[]) => values },
  Resource: class {
    url: string;
    headers: Record<string, string>;
    constructor(value: { url: string; headers: Record<string, string> }) {
      this.url = value.url;
      this.headers = value.headers;
    }
  },
  UrlTemplateImageryProvider: class {
    errorEvent = { addEventListener: vi.fn() };
    constructor(options: (typeof fixture.requests)[number]) {
      fixture.requests.push(options);
    }
  },
  ImageryLayer: class {
    constructor(
      public provider: unknown,
      public options: unknown,
    ) {}
  },
}));
vi.mock("@/cesium/SceneContext", () => ({
  useScene: () => ({
    isDestroyed: false,
    scene: { requestRender: fixture.render },
    viewer: {
      imageryLayers: {
        length: 1,
        add: fixture.add,
        remove: fixture.remove,
        layerAdded: { addEventListener: () => () => undefined },
      },
    },
    layers: { ensureFallbackBasemap: () => Promise.resolve() },
  }),
}));
vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<typeof IdentityModule>()),
  landUsesOidc: true,
}));
afterEach(() => {
  cleanup();
  useLandContext.getState().clear();
  useLandIdentity.getState().setSession(null, null);
  fixture.requests.length = 0;
  vi.clearAllMocks();
});
it("carries workspace credentials in headers and rebuilds private tiles after token refresh", () => {
  useLandIdentity.getState().setSession("initial-test-token", "alice");
  useLandIdentity.getState().selectWorkspace("workspace", "owner");
  useLandContext.getState().setRaster({
    id: "raster",
    band: 2,
    categorical: true,
    bounds: [-77.055, 38.886, -77.045, 38.891],
    attribution: "Public reference fixture",
    opacity: 0.6,
  });
  const mounted = render(<LandRasterBridge />);
  expect(fixture.requests[0]?.url.headers).toEqual({
    Authorization: "Bearer initial-test-token",
    "X-Workspace-ID": "workspace",
  });
  expect(fixture.requests[0]?.url.url).not.toContain("token");
  expect(fixture.add).toHaveBeenCalledWith(
    expect.objectContaining({ magnificationFilter: 9728, minificationFilter: 9728 }),
  );
  expect(fixture.requests[0]?.rectangle).toEqual([-77.055, 38.886, -77.045, 38.891]);
  useLandIdentity.getState().setSession("refreshed-test-token", "alice");
  expect(fixture.requests).toHaveLength(2);
  expect(fixture.requests[1]?.url.headers.Authorization).toBe("Bearer refreshed-test-token");
  expect(fixture.remove).toHaveBeenCalledTimes(1);
  useLandContext.getState().clear();
  expect(fixture.remove).toHaveBeenCalledTimes(2);
  mounted.unmount();
  useLandIdentity.getState().setSession("later-test-token", "alice");
  expect(fixture.requests).toHaveLength(2);
});
it("routes archive overlays through the private alignment tiles and rebuilds when the layer kind changes", () => {
  const layer = {
    id: "saved-map",
    band: 1,
    bounds: [-77.05, 38.88, -77.04, 38.89],
    attribution: "USGS public domain",
    opacity: 0.6,
  };
  useLandContext.getState().setRaster(layer);
  render(<LandRasterBridge />);
  useLandContext.getState().setRaster({ ...layer, kind: "archive-alignment" });
  expect(fixture.requests).toHaveLength(2);
  expect(fixture.requests[1]?.url.url).toContain(
    "/research/image-registrations/saved-map/tiles/{z}/{x}/{y}.png",
  );
  expect(fixture.remove).toHaveBeenCalledOnce();
});
