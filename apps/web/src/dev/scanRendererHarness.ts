/**
 * A bare CesiumJS page that draws one splat tileset with each splat renderer in turn --
 * CesiumJS, Spark, PlayCanvas (cesium/scanView) -- from the same camera: the driver for
 * e2e/scanRenderers.spec.ts, which checks that the dedicated renderers put the scan where
 * CesiumJS does (their camera follows Cesium's) and that each streams and draws its tiles.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production bundle.
 */

import { CesiumWidget, Color, HeadingPitchRange, Math as CesiumMath, Matrix4 } from "cesium";
import type { SiteAsset } from "@twin/contracts";

import { createSiteTileset } from "@/cesium/providers/tiles";
import { ScanRendererHost, type ScanRendererStatus } from "@/cesium/scanView/ScanRendererHost";
import type { SplatRendererKind } from "@/cesium/scanView/types";
import { installSplatDecoder } from "@/cesium/splatDecoder";
import { installSplatSorter } from "@/cesium/splatSorter";

export interface ScanRendererHarness {
  /** Draws the scan with `kind` and waits until its tiles are up (or `timeoutS` pass). */
  use(kind: SplatRendererKind, timeoutS: number): Promise<ScanRendererStatus>;
  status(): ScanRendererStatus;
}

export async function startScanRendererHarness(options: {
  container: HTMLElement;
  tilesetUrl: string;
  rangeM: number;
}): Promise<ScanRendererHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: false,
    msaaSamples: 1,
  });
  const { scene, camera } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  if (scene.sun) scene.sun.show = false;
  scene.backgroundColor = Color.fromCssColorString("#10141a");
  installSplatSorter();
  installSplatDecoder();
  const asset = {
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: options.tilesetUrl },
    renderConfig: {},
  } as unknown as SiteAsset;
  const tileset = await createSiteTileset(asset, { maximumScreenSpaceError: 2 });
  tileset.show = true;
  scene.primitives.add(tileset);
  camera.lookAt(
    tileset.boundingSphere.center,
    new HeadingPitchRange(0.4, CesiumMath.toRadians(-20), options.rangeM),
  );
  camera.lookAtTransform(Matrix4.IDENTITY);
  const host = new ScanRendererHost(widget);
  const frame = (): Promise<void> => new Promise((done) => requestAnimationFrame(() => done()));

  return {
    status: () => host.status(),
    async use(kind, timeoutS) {
      tileset.show = kind === "cesium";
      tileset.preloadWhenHidden = kind === "cesium";
      host.setRenderer(kind);
      host.setTarget(kind === "cesium" ? null : { key: "harness", tileset });
      const until = performance.now() + timeoutS * 1000;
      let settled = 0;
      let lastTiles = -1;
      while (performance.now() < until) {
        await frame();
        const status = host.status();
        const tiles = kind === "cesium" ? (tileset.tilesLoaded ? 1 : 0) : status.tiles;
        const ready = kind === "cesium" ? tileset.tilesLoaded : status.frames > 5 && tiles > 0;
        settled = ready && tiles === lastTiles ? settled + 1 : 0;
        lastTiles = tiles;
        if (settled > 60) break;
      }
      return host.status();
    },
  };
}
