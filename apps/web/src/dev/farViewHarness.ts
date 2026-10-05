/**
 * A real `CesiumSceneManager` with one splat site, for the far view end-to-end check
 * (e2e/scanFarView.spec.ts): is the scan drawn -- by CesiumJS, PlayCanvas, Spark or PlayCanvas
 * on WebGPU -- from a few hundred metres, where its site is no longer engaged, and gone from a
 * few kilometres; and does zooming out from up close keep the dedicated renderer's session?
 *
 * The site is the synthetic LOD tree (some 8 m in radius), served by the spec, with no rig, so
 * every renderer may draw it. The globe, sky and sun are hidden: the frame is the scan alone.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production bundle.
 */

import {
  Cartesian3,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  Matrix4,
  SceneTransforms,
  type BoundingSphere,
} from "cesium";

import type { Site, SiteSummary } from "@twin/contracts";

import { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { ScanRendererStatus } from "@/cesium/scanView/ScanRendererHost";
import type { SplatRendererKind } from "@/cesium/scanView/types";

const SITE_ID = "00000000-0000-4000-8000-0000000000f1";
const ASSET_ID = "00000000-0000-4000-8000-0000000000f2";

/** What the spec reads after each move. */
export interface FarViewState {
  /** Camera to the scan's centre (m). */
  distanceM: number;
  /** CesiumJS draws the scan's tileset itself. */
  tilesetShown: boolean;
  /** What a dedicated renderer is asked to draw (SiteManager.scanTarget), or null. */
  target: { key: string; far: boolean } | null;
  /** The dedicated renderer's status. */
  scan: ScanRendererStatus;
  /** Overlay canvases in the page, and whether the one there carries the spec's mark. */
  overlays: number;
  marked: boolean;
}

export interface FarViewHarness {
  /** Chooses who draws the scan, as Settings › Advanced does. */
  use(kind: SplatRendererKind): void;
  /** Puts the camera `rangeM` from the scan's centre, `pitchDeg` below level. */
  view(rangeM: number, pitchDeg: number): void;
  state(): FarViewState;
  /** Marks the overlay canvas there now, to tell later whether it is still the same one. */
  mark(): boolean;
  /** Where the scan's centre is drawn (CSS pixels), or null. */
  centre(): [number, number] | null;
}

function siteFixture(
  tilesetUrl: string,
  longitude: number,
  latitude: number,
): {
  site: Site;
  summary: SiteSummary;
} {
  const d = 0.00005;
  const ring = [
    [longitude - d, latitude - d],
    [longitude + d, latitude - d],
    [longitude + d, latitude + d],
    [longitude - d, latitude + d],
    [longitude - d, latitude - d],
  ];
  const timestamps = { createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" };
  const site = {
    id: SITE_ID,
    slug: "far-view-tree",
    name: "Far view fixture",
    description: "",
    boundary: { type: "Polygon", coordinates: [ring] },
    centroid: { longitude, latitude, height: 0 },
    areaM2: 100,
    thumbnailUrl: null,
    metadata: {},
    attribution: [],
    license: null,
    assets: [
      {
        id: ASSET_ID,
        siteId: SITE_ID,
        provider: "3d-tiles-url",
        name: "Gaussian splat (synthetic tree)",
        representation: "gaussian-splat",
        source: { type: "3d-tiles-url", url: tilesetUrl },
        footprint: null,
        observedAt: null,
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
          clipsWorld: false,
          clipFootprint: "catalog",
          heightOffsetM: 0,
        },
        defaultVisible: true,
        ...timestamps,
      },
    ],
    cameraBookmarks: [],
    ...timestamps,
  };
  const summary = {
    id: SITE_ID,
    slug: site.slug,
    name: site.name,
    description: "",
    centroid: site.centroid,
    areaM2: site.areaM2,
    thumbnailUrl: null,
    representations: ["gaussian-splat"],
    latestObservedAt: null,
    quality: { resolutionDescription: null, groundSampleDistanceM: null },
    ...timestamps,
  };
  return { site: site as unknown as Site, summary: summary as unknown as SiteSummary };
}

export async function startFarViewHarness(options: {
  container: HTMLElement;
  tilesetUrl: string;
  longitude: number;
  latitude: number;
}): Promise<FarViewHarness> {
  // A dependency first met mid-test on a fresh dev server -- the splat decode worker's, a
  // renderer's chunk -- has Vite optimise it and reload the page. Met here, before the harness
  // is ready (the spec waits for it across a reload), it only makes the start slower.
  await Promise.all([
    import("@spz-loader/core"),
    import("@/cesium/scanView/playcanvasBackend"),
    import("@/cesium/scanView/sparkBackend"),
  ]);
  const scene = new CesiumSceneManager(options.container, {
    ionToken: undefined,
    // Out past the far view's limit, so the site loads hidden and the spec moves in.
    home: { longitude: options.longitude, latitude: options.latitude, height: 30_000 },
    nightSky: false,
  });
  const viewer = scene.viewer;
  const gl = viewer.scene;
  gl.globe.show = false;
  if (gl.skyAtmosphere) gl.skyAtmosphere.show = false;
  if (gl.skyBox) gl.skyBox.show = false;
  if (gl.sun) gl.sun.show = false;
  if (gl.moon) gl.moon.show = false;
  gl.fog.enabled = false;
  gl.backgroundColor = Color.fromCssColorString("#10141a");
  scene.performance.configure({
    preset: "balanced",
    manualScreenSpaceError: null,
    adaptive: false,
  });

  const { site, summary } = siteFixture(options.tilesetUrl, options.longitude, options.latitude);
  scene.sites.setCatalog([summary], () => Promise.resolve(site));
  await scene.sites.activate(SITE_ID);
  // Proximity may have started the same load first; it is in once the tileset is.
  for (let i = 0; i < 600 && !scene.sites.tilesetFor(SITE_ID, "gaussian-splat"); i++) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  const tileset = scene.sites.tilesetFor(SITE_ID, "gaussian-splat");
  if (!tileset) throw new Error("The far view fixture's tileset did not load.");
  const sphere = (): BoundingSphere => tileset.boundingSphere;
  const overlay = (): HTMLCanvasElement | null =>
    document.querySelector<HTMLCanvasElement>("canvas[data-scan-renderer]");

  return {
    use(kind) {
      scene.setSplatRenderer(kind);
      gl.requestRender();
    },
    view(rangeM, pitchDeg) {
      viewer.camera.lookAt(
        sphere().center,
        new HeadingPitchRange(0.4, CesiumMath.toRadians(-pitchDeg), rangeM),
      );
      viewer.camera.lookAtTransform(Matrix4.IDENTITY);
      gl.requestRender();
    },
    state() {
      const target = scene.sites.scanTarget();
      const canvas = overlay();
      return {
        distanceM: Math.round(Cartesian3.distance(viewer.camera.positionWC, sphere().center)),
        tilesetShown: tileset.show,
        target: target ? { key: target.key, far: target.far } : null,
        scan: scene.scanRendererStatus,
        overlays: document.querySelectorAll("canvas[data-scan-renderer]").length,
        marked: canvas?.dataset.farViewMark === "1",
      };
    },
    mark() {
      const canvas = overlay();
      if (!canvas) return false;
      canvas.dataset.farViewMark = "1";
      return true;
    },
    centre() {
      const at = SceneTransforms.worldToWindowCoordinates(gl, sphere().center);
      return at ? [at.x, at.y] : null;
    },
  };
}
