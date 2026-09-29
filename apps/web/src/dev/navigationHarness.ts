/**
 * A bare CesiumJS page that loads one splat tileset the way the console does and flies a
 * scripted camera path through it, reporting what each frame cost the main thread -- the
 * driver for the navigation measurement (e2e/navigationPerf.spec.ts).
 *
 * What it answers: while the camera moves, does streaming (tile decode, the splat
 * primitive's snapshot rebuilds) take main-thread time away from the camera? Headless GL is
 * SwiftShader, which rasterises on the CPU, so the harness separates the frame's JavaScript
 * (`scene.preUpdate` to `scene.postUpdate`, where tiles are selected, decoded and aggregated)
 * from the frame as a whole.
 *
 * Loaded dynamically by the spec; nothing imports it, so it never reaches the production bundle.
 */

import { BoundingSphere, CesiumWidget, Color, HeadingPitchRange, Math as CesiumMath } from "cesium";
import type { SiteAsset } from "@twin/contracts";

import { Emitter } from "@/lib/emitter";

import { createSiteTileset } from "@/cesium/providers/tiles";
import { splatTilesetOf } from "@/cesium/splatInternals";
import { SplatMotionGate } from "@/cesium/splatMotionGate";
import type { SceneEvents } from "@/cesium/types";

export interface NavigationHarnessOptions {
  readonly container: HTMLElement;
  readonly tilesetUrl: string;
  /** The motion gate and off-screen selection on (as the console runs), or both off. */
  readonly motionFirst: boolean;
}

export interface MotionReport {
  /** Main-thread milliseconds of each frame's update (selection, decode, aggregation). */
  updateMs: number[];
  /** Milliseconds between frames. */
  intervalMs: number[];
  /** Snapshot rebuilds the splat primitive committed during the motion. */
  rebuilds: number;
  /** Tiles whose content arrived during the motion. */
  tilesLoaded: number;
  /** Gaussians drawn at the end. */
  drawn: number;
}

export interface NavigationHarness {
  /** Orbits the tileset's centre at `rangeM`, one full turn in `seconds`, reporting frames. */
  orbit(rangeM: number, seconds: number): Promise<MotionReport>;
  /** Renders until nothing is loading, or `seconds` pass. */
  settle(seconds: number): Promise<void>;
}

export async function startNavigationHarness(
  options: NavigationHarnessOptions,
): Promise<NavigationHarness> {
  const widget = new CesiumWidget(options.container, {
    baseLayer: false,
    requestRenderMode: false,
    msaaSamples: 1,
  });
  const { scene, camera } = widget;
  scene.globe.show = false;
  if (scene.skyBox) scene.skyBox.show = false;
  if (scene.skyAtmosphere) scene.skyAtmosphere.show = false;
  scene.backgroundColor = Color.fromCssColorString("#10141a");
  const asset = {
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: options.tilesetUrl },
    renderConfig: {},
  } as unknown as SiteAsset;
  const tileset = await createSiteTileset(asset, { maximumScreenSpaceError: 16 });
  tileset.show = true;
  scene.primitives.add(tileset);
  const events = new Emitter<SceneEvents>();
  if (options.motionFirst) new SplatMotionGate(scene, events);
  else (tileset as unknown as { selectOffscreen: boolean }).selectOffscreen = false;
  let tilesLoaded = 0;
  tileset.tileLoad.addEventListener(() => (tilesLoaded += 1));
  const centre = BoundingSphere.clone(tileset.boundingSphere);
  camera.lookAt(centre.center, new HeadingPitchRange(0, CesiumMath.toRadians(-25), 60));

  let updateStart = 0;
  const updateMs: number[] = [];
  let recording = false;
  scene.preUpdate.addEventListener(() => {
    updateStart = performance.now();
  });
  scene.postUpdate.addEventListener(() => {
    if (recording) updateMs.push(performance.now() - updateStart);
  });
  const generation = (): number =>
    splatTilesetOf(tileset).gaussianSplatPrimitive?._snapshot?.generation ?? 0;

  const frame = (): Promise<number> => new Promise((done) => requestAnimationFrame(done));

  return {
    async orbit(rangeM, seconds) {
      updateMs.length = 0;
      const intervalMs: number[] = [];
      const loadedBefore = tilesLoaded;
      const generationBefore = generation();
      recording = true;
      events.emit("motion", true);
      const start = await frame();
      let last = start;
      for (;;) {
        const now = await frame();
        intervalMs.push(now - last);
        last = now;
        const t = (now - start) / 1000 / seconds;
        if (t >= 1) break;
        camera.lookAt(
          centre.center,
          new HeadingPitchRange(t * CesiumMath.TWO_PI, CesiumMath.toRadians(-20), rangeM),
        );
      }
      recording = false;
      events.emit("motion", false);
      const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive;
      return {
        updateMs: [...updateMs],
        intervalMs,
        rebuilds: generation() - generationBefore,
        tilesLoaded: tilesLoaded - loadedBefore,
        drawn: primitive?._numSplats ?? 0,
      };
    },
    async settle(seconds) {
      const until = performance.now() + seconds * 1000;
      while (performance.now() < until) {
        await frame();
        if (tileset.tilesLoaded) break;
      }
    },
  };
}
