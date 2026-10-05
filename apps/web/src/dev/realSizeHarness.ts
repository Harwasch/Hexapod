/**
 * A real `CesiumSceneManager` with one pipeline-placed splat site at a runtime scale, and the
 * Set real size tool beside it: the driver for e2e/realSize.spec.ts.
 *
 * The site is the committed synthetic tree, registered as a pipeline run would have it: its
 * metadata records the run and the origin its tiles are placed at (the root transform's), so
 * the tool is offered for it, and its `renderConfig.scale` is the harness's `scale`. The site's
 * record lives in a react-query client, as in the app, and `watchSiteRecords` hands every
 * record to the scene, so a Save that the spec answers re-places the scan as the app would. The
 * spec routes the API (`PUT /assets/{id}/scale`, `GET /sites/{id}`); this only points the
 * camera and reads back what is drawn: the pixels, the bounding sphere, the collider.
 *
 * The globe, the sky and the fog are off, so the scan is drawn over black above the horizon and
 * navy below it, and its greens and browns are told from both by colour. Loaded dynamically
 * by the spec; nothing imports it, so it never reaches the production bundle.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Cartesian2,
  Cartesian3,
  Cartographic,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  Ray,
  SceneTransforms,
} from "cesium";
import { createElement } from "react";
import { createRoot } from "react-dom/client";

import type { Site, SiteSummary } from "@twin/contracts";

import { queryKeys, useSite, watchSiteRecords } from "@/api/queries";
import { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { sceneRegistry } from "@/cesium/SceneContext";
import { splatTilesetOf } from "@/cesium/splatInternals";
import { RealSize } from "@/features/inspector/RealSize";
// The styles the tool is drawn with in the app: the glass, and the app's own.
import "@twin/ui/styles.css";
import "@/styles/app.css";

export const SITE_ID = "77777777-7777-4777-8777-777777777777";
export const ASSET_ID = "88888888-8888-4888-8888-888888888888";

export interface RealSizeHarnessOptions {
  readonly container: HTMLElement;
  /** Where the tool is mounted. */
  readonly panel: HTMLElement;
  readonly tilesetUrl: string;
  /** `renderConfig.scale`, as the catalog has it. */
  readonly scale: number;
}

export interface RealSizeHarness {
  /** The scan's bounding sphere radius on the globe, metres. */
  radius(): number;
  /** The scale the scan is drawn at now. */
  shownScale(): number | null;
  previewScale(scale: number | null): boolean;
  /**
   * How far east of the origin's vertical a ray from the west meets the scan, metres on the
   * globe: the ray is `heightM` above the origin, from 30 m west, through the collider.
   */
  edgeEast(heightM: number): number | null;
  /** Where a point of the scan's own frame is on screen now, CSS px. */
  screenOf(local: [number, number, number]): { x: number; y: number } | null;
  /** The drawn scan's box on screen, CSS px: the tree's colours, not the background's. */
  extent(): Promise<{ width: number; height: number; pixels: number }>;
  /** Waits for the drawn scan to settle after a change: tiles in, re-baked, sorted. */
  settle(): Promise<void>;
  /** The length the Set real size tool measured, as drawn now. */
  measuredM(): number | null;
  /** The site's record as the page has it now. */
  record(): Site | undefined;
}

/** The site a pipeline run would have registered for the tiles at `url`, placed at `origin`. */
function siteFixture(url: string, origin: Cartographic, scale: number): Site {
  const lon = CesiumMath.toDegrees(origin.longitude);
  const lat = CesiumMath.toDegrees(origin.latitude);
  const d = 0.0001 * scale;
  const ring = [
    [lon - d, lat - d],
    [lon + d, lat - d],
    [lon + d, lat + d],
    [lon - d, lat + d],
    [lon - d, lat - d],
  ];
  const timestamps = { createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" };
  return {
    id: SITE_ID,
    slug: "synthetic-tree",
    name: "Synthetic tree",
    description: null,
    boundary: { type: "Polygon", coordinates: [ring] },
    centroid: { longitude: lon, latitude: lat, height: origin.height },
    areaM2: 400 * scale * scale,
    thumbnailUrl: null,
    metadata: {
      captureId: "99999999-9999-4999-8999-999999999999",
      registration: {
        georef: { lat, lon, height: origin.height },
      },
    },
    attribution: [],
    license: null,
    assets: [
      {
        id: ASSET_ID,
        siteId: SITE_ID,
        provider: "3d-tiles-url",
        name: "Gaussian splat",
        representation: "gaussian-splat",
        source: { type: "3d-tiles-url", url },
        footprint: null,
        observedAt: null,
        validFrom: null,
        validTo: null,
        resolution: null,
        crs: null,
        license: null,
        attribution: [],
        provenance: { georefMethod: "exif-gps", scaleSource: "unresolved", uncertaintyM: 10 },
        renderConfig: {
          maximumScreenSpaceError: 1,
          pointCloudShading: null,
          // Nothing under it to clip or clamp to: the globe is hidden.
          clipsWorld: false,
          clipFootprint: "catalog",
          heightOffsetM: 0,
          ...(scale === 1 ? {} : { scale }),
        },
        defaultVisible: true,
        ...timestamps,
      },
    ],
    cameraBookmarks: [],
    ...timestamps,
  } as unknown as Site;
}

export async function startRealSizeHarness(
  options: RealSizeHarnessOptions,
): Promise<RealSizeHarness> {
  const document = (await (await fetch(options.tilesetUrl)).json()) as {
    root: { transform: number[] };
  };
  const rootTransform = Matrix4.fromArray(document.root.transform);
  const origin = Matrix4.getTranslation(rootTransform, new Cartesian3());
  const originCarto = Cartographic.fromCartesian(origin);
  const site = siteFixture(options.tilesetUrl, originCarto, options.scale);

  const scene = new CesiumSceneManager(options.container, {
    ionToken: undefined,
    home: {
      longitude: CesiumMath.toDegrees(originCarto.longitude),
      latitude: CesiumMath.toDegrees(originCarto.latitude),
      height: originCarto.height + 60,
    },
    nightSky: false,
  });
  const viewer = scene.viewer;
  const gl = viewer.scene;
  gl.globe.show = false;
  if (gl.skyAtmosphere) gl.skyAtmosphere.show = false;
  if (gl.skyBox) gl.skyBox.show = false;
  if (gl.sun) gl.sun.show = false;
  gl.fog.enabled = false;
  scene.performance.configure({
    preset: "balanced",
    manualScreenSpaceError: null,
    adaptive: false,
  });
  // CesiumJS draws the scan here, so its pixels are the canvas's.
  scene.setSplatRenderer("cesium");

  // The site's record as the app keeps it: in the query cache, handed to the scene on change.
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  client.setQueryData(queryKeys.site(SITE_ID), site);
  watchSiteRecords(client, (record) => scene.sites.updateRecord(record));
  const summary = {
    id: SITE_ID,
    slug: site.slug,
    name: site.name,
    description: null,
    centroid: site.centroid,
    areaM2: site.areaM2,
    thumbnailUrl: null,
    representations: ["gaussian-splat"],
    latestObservedAt: null,
    quality: { resolutionDescription: null, groundSampleDistanceM: null },
    createdAt: site.createdAt,
    updatedAt: site.updatedAt,
  } as unknown as SiteSummary;
  scene.sites.setCatalog([summary], () => Promise.resolve(site));
  await scene.sites.activate(SITE_ID);
  sceneRegistry.set(scene);

  // Side on to the tree, level with the middle of it at its registered size, a way off.
  const target = Matrix4.multiplyByPoint(rootTransform, new Cartesian3(0, 0, 3), new Cartesian3());
  viewer.camera.lookAt(
    target,
    new HeadingPitchRange(CesiumMath.toRadians(90), CesiumMath.toRadians(-5), 28),
  );
  viewer.camera.lookAtTransform(Matrix4.IDENTITY);

  const Tool = () => {
    const record = useSite(SITE_ID).data;
    return record ? createElement(RealSize, { site: record }) : null;
  };
  createRoot(options.panel).render(
    createElement(QueryClientProvider, { client }, createElement(Tool)),
  );

  const tileset = () => scene.sites.scalableTileset(SITE_ID);

  const nextFrame = (): Promise<void> =>
    new Promise((resolve) => {
      const remove = gl.postRender.addEventListener(() => {
        remove();
        resolve();
      });
      gl.requestRender();
    });

  /** The frame, read inside postRender: the drawing buffer is not kept between frames. */
  const grab = (): Promise<ImageData | null> =>
    new Promise((resolve) => {
      const remove = gl.postRender.addEventListener(() => {
        remove();
        const source = viewer.canvas;
        const copy = window.document.createElement("canvas");
        copy.width = source.width;
        copy.height = source.height;
        const context = copy.getContext("2d");
        if (!context) {
          resolve(null);
          return;
        }
        context.drawImage(source, 0, 0);
        resolve(context.getImageData(0, 0, copy.width, copy.height));
      });
      gl.requestRender();
    });

  const extent = async (): Promise<{ width: number; height: number; pixels: number }> => {
    const image = await grab();
    if (!image) return { width: 0, height: 0, pixels: 0 };
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    let pixels = 0;
    const { data, width, height } = image;
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const at = (y * width + x) * 4;
        // The tree's greens and browns; not the black sky, nor the navy below the horizon.
        const warm = Math.max(data[at] ?? 0, data[at + 1] ?? 0);
        if (warm <= 50 || warm <= (data[at + 2] ?? 0)) continue;
        pixels += 1;
        x0 = Math.min(x0, x);
        y0 = Math.min(y0, y);
        x1 = Math.max(x1, x);
        y1 = Math.max(y1, y);
      }
    const ratio = viewer.canvas.clientWidth / Math.max(1, width);
    if (pixels === 0) return { width: 0, height: 0, pixels };
    return { width: (x1 - x0 + 1) * ratio, height: (y1 - y0 + 1) * ratio, pixels };
  };

  /**
   * Whether the splats drawn are baked for the placement now: CesiumJS re-bakes every splat
   * when the model matrix changes, about an east/north/up frame at the tileset's centre then.
   */
  const rebaked = (): boolean => {
    const current = tileset();
    const primitive = current ? splatTilesetOf(current).gaussianSplatPrimitive : undefined;
    const frame = primitive?._rootTransform;
    if (!current?.tilesLoaded || !primitive?._numSplats || !frame || primitive._pendingSnapshot)
      return false;
    const at = new Cartesian3(frame[12] ?? 0, frame[13] ?? 0, frame[14] ?? 0);
    return Cartesian3.distance(at, current.boundingSphere.center) < 1e-3;
  };

  const settle = async (): Promise<void> => {
    // Tiles in and re-baked for the current model matrix, then the drawn box the same three
    // frames running: the engine adds re-baked tiles back a batch at a time.
    let last: { width: number; height: number } | null = null;
    let steady = 0;
    for (let frame = 0; frame < 120; frame++) {
      await nextFrame();
      if (!rebaked()) {
        last = null;
        continue;
      }
      const now = await extent();
      const same =
        last !== null &&
        now.pixels > 0 &&
        Math.abs(now.width - last.width) < 1 &&
        Math.abs(now.height - last.height) < 1;
      steady = same ? steady + 1 : 0;
      if (steady >= 3) return;
      last = now;
    }
  };

  await settle();

  return {
    radius: () => tileset()?.boundingSphere.radius ?? 0,
    shownScale: () => scene.sites.shownScale(SITE_ID),
    previewScale: (scale) => scene.sites.previewScale(SITE_ID, scale),
    edgeEast: (heightM) => {
      const east = Matrix4.multiplyByPointAsVector(
        rootTransform,
        Cartesian3.UNIT_X,
        new Cartesian3(),
      );
      Cartesian3.normalize(east, east);
      const from = Matrix4.multiplyByPoint(
        rootTransform,
        new Cartesian3(-30, 0, heightM),
        new Cartesian3(),
      );
      const hit = scene.collider.raycast(new Ray(from, east));
      return hit ? hit.distance - 30 : null;
    },
    screenOf: (local) => {
      const current = tileset();
      if (!current) return null;
      const toWorld = Matrix4.multiply(current.modelMatrix, rootTransform, new Matrix4());
      const world = Matrix4.multiplyByPoint(toWorld, Cartesian3.fromArray(local), new Cartesian3());
      const at = SceneTransforms.worldToWindowCoordinates(gl, world, new Cartesian2());
      return at ? { x: at.x, y: at.y } : null;
    },
    extent,
    settle,
    measuredM: () => scene.scaleMeasure.lengthM(),
    record: () => client.getQueryData<Site>(queryKeys.site(SITE_ID)),
  };
}
