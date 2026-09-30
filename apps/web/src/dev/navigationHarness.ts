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

import {
  BoundingSphere,
  Cartesian3,
  CesiumWidget,
  Color,
  HeadingPitchRange,
  Math as CesiumMath,
  type Cesium3DTile,
} from "cesium";
import * as CesiumBarrel from "cesium";
import type { SiteAsset } from "@twin/contracts";

import { Emitter } from "@/lib/emitter";

import { createSiteTileset } from "@/cesium/providers/tiles";
import { splatTilesetOf } from "@/cesium/splatInternals";
import { SplatMotionGate } from "@/cesium/splatMotionGate";
import { installSplatDecoder } from "@/cesium/splatDecoder";
import { installSplatSorter } from "@/cesium/splatSorter";
import type { SceneEvents } from "@/cesium/types";

export interface NavigationHarnessOptions {
  readonly container: HTMLElement;
  readonly tilesetUrl: string;
  /** The console's splat settings (motion gate, off-screen selection, focus, incremental
   *  slots), or the engine as it was. */
  readonly motionFirst: boolean;
  /** Stream, pack, upload and sort as usual but never draw the splats: under SwiftShader a
   *  frame of a large scan takes seconds, and what is measured is the main thread's work. */
  readonly skipDraw?: boolean;
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

/** Main-thread GPU traffic and network over one scenario. */
export interface Traffic {
  /** Synchronous GPU read-backs (`readPixels`). */
  readbacks: number;
  /** `bufferData` calls (buffer (re)allocations) and `bufferSubData` bytes (sort orders). */
  bufferAllocations: number;
  bufferUploadMB: number;
  /** `texSubImage2D` bytes (splat slots uploaded). */
  textureUploadMB: number;
  /** Tile requests (.glb) started, and how many of them were for a tile already fetched. */
  tileRequests: number;
  tileRefetches: number;
  /** Sorts the splat primitive completed. */
  sorts: number;
}

export interface LookReport extends Traffic {
  /** Gaussians drawn looking at view A once settled. */
  settledA: number;
  /** After looking away and back: ms, and frames, until 95% of view A's tiles are drawn. */
  restoreMs: number | null;
  restoreFrames: number | null;
  /** Share of view A's tiles drawn on the first frame back. */
  firstFrameShare: number;
  /** Share of view A's tiles still drawn just before turning back (looking the other way). */
  awayShare: number;
}

export interface WalkReport extends Traffic {
  updateMs: number[];
  intervalMs: number[];
}

export interface NavigationHarness {
  /** From `eyeM` above the centre, look at heading A, turn 180 degrees, turn back. */
  lookAround(settleS: number, eyeM: number): Promise<LookReport>;
  /** Walks straight ahead from the centre at `speed` m/s for `seconds`. */
  walk(speed: number, seconds: number, eyeM: number): Promise<WalkReport>;
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
  if (options.skipDraw) {
    type Update = (
      this: { _drawCommand?: unknown },
      frameState: { commandList: unknown[] },
    ) => void;
    const prototype = (
      (CesiumBarrel as unknown as Record<string, unknown>).GaussianSplatPrimitive as
        { prototype: Record<string, Update | undefined> } | undefined
    )?.prototype;
    const update = prototype?.update;
    if (prototype && update) {
      prototype.update = function (frameState) {
        update.call(this, frameState);
        const at = frameState.commandList.indexOf(this._drawCommand);
        if (at >= 0) frameState.commandList.splice(at, 1);
      };
    }
  }
  const tileset = await createSiteTileset(asset, { maximumScreenSpaceError: 16 });
  tileset.show = true;
  scene.primitives.add(tileset);
  const events = new Emitter<SceneEvents>();
  if (options.motionFirst) {
    new SplatMotionGate(scene, events);
    installSplatSorter();
    installSplatDecoder();
  } else {
    // The engine as it was: every selected splat re-aggregated on each change, frustum only.
    const patched = tileset as unknown as {
      selectOffscreen: boolean;
      splatIncremental: boolean;
    };
    patched.selectOffscreen = false;
    patched.splatIncremental = false;
  }
  // Count the GPU traffic that stalls the main thread, and every tile request.
  const counters = {
    readbacks: 0,
    bufferAllocations: 0,
    bufferUpload: 0,
    textureUpload: 0,
    sorts: 0,
  };
  const gl = (scene as unknown as { context: { _gl: WebGL2RenderingContext } }).context._gl;
  const wrap = (name: string, count: (args: unknown[]) => void) => {
    const methods = gl as unknown as Record<string, (...args: unknown[]) => unknown>;
    const original = methods[name];
    if (!original) return;
    (gl as unknown as Record<string, unknown>)[name] = function (
      this: unknown,
      ...args: unknown[]
    ) {
      count(args);
      return original.apply(gl, args);
    };
  };
  const bytesOf = (value: unknown): number => (ArrayBuffer.isView(value) ? value.byteLength : 0);
  wrap("readPixels", () => (counters.readbacks += 1));
  wrap("bufferData", () => (counters.bufferAllocations += 1));
  wrap("bufferSubData", (args) => (counters.bufferUpload += bytesOf(args[2])));
  wrap("texSubImage2D", (args) => (counters.textureUpload += bytesOf(args[8] ?? args[6])));
  const fetched = new Map<string, number>();
  let tileRequests = 0;
  let tileRefetches = 0;
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (!entry.name.includes(".glb")) continue;
      tileRequests += 1;
      const seen = fetched.get(entry.name) ?? 0;
      if (seen > 0) tileRefetches += 1;
      fetched.set(entry.name, seen + 1);
    }
  });
  observer.observe({ type: "resource", buffered: false });
  const traffic = () => ({
    readbacks: counters.readbacks,
    bufferAllocations: counters.bufferAllocations,
    bufferUploadMB: counters.bufferUpload / 1e6,
    textureUploadMB: counters.textureUpload / 1e6,
    tileRequests,
    tileRefetches,
    sorts: counters.sorts,
  });
  const delta = (before: ReturnType<typeof traffic>): ReturnType<typeof traffic> => {
    const now = traffic();
    return {
      readbacks: now.readbacks - before.readbacks,
      bufferAllocations: now.bufferAllocations - before.bufferAllocations,
      bufferUploadMB: Math.round((now.bufferUploadMB - before.bufferUploadMB) * 10) / 10,
      textureUploadMB: Math.round((now.textureUploadMB - before.textureUploadMB) * 10) / 10,
      tileRequests: now.tileRequests - before.tileRequests,
      tileRefetches: now.tileRefetches - before.tileRefetches,
      sorts: now.sorts - before.sorts,
    };
  };
  let lastIndexes: unknown;
  scene.postRender.addEventListener(() => {
    const indexes = (
      splatTilesetOf(tileset).gaussianSplatPrimitive as unknown as
        { _indexes?: unknown } | undefined
    )?._indexes;
    if (indexes && indexes !== lastIndexes) counters.sorts += 1;
    lastIndexes = indexes;
  });

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

  /** Tiles drawn now: live incremental slots, or the snapshot's tiles. */
  const drawnTiles = (): Set<Cesium3DTile> => {
    const primitive = splatTilesetOf(tileset).gaussianSplatPrimitive as unknown as
      { _tileSlots?: Map<Cesium3DTile, unknown>; _selectedTileSet?: Set<Cesium3DTile> } | undefined;
    return new Set(primitive?._tileSlots?.keys() ?? primitive?._selectedTileSet ?? []);
  };
  const gaussiansOf = (tiles: Iterable<Cesium3DTile>): number => {
    let n = 0;
    for (const tile of tiles)
      n += (tile.content as { pointsLength?: number } | undefined)?.pointsLength ?? 0;
    return n;
  };
  const eye = (eyeM: number): Cartesian3 => {
    const up = Cartesian3.normalize(centre.center, new Cartesian3());
    return Cartesian3.add(
      centre.center,
      Cartesian3.multiplyByScalar(up, eyeM, up),
      new Cartesian3(),
    );
  };
  const look = (position: Cartesian3, headingDeg: number): void => {
    camera.setView({
      destination: position,
      orientation: {
        heading: CesiumMath.toRadians(headingDeg),
        pitch: CesiumMath.toRadians(-8),
        roll: 0,
      },
    });
  };
  const settleFor = async (seconds: number): Promise<void> => {
    const until = performance.now() + seconds * 1000;
    while (performance.now() < until) await frame();
  };

  return {
    async lookAround(settleS, eyeM) {
      const at = eye(eyeM);
      look(at, 0);
      events.emit("motion", false);
      await settleFor(settleS);
      const viewA = drawnTiles();
      const settledA = gaussiansOf(viewA);
      look(at, 180);
      await settleFor(settleS);
      const awayTiles = drawnTiles();
      let awayHit = 0;
      for (const tile of viewA)
        if (awayTiles.has(tile))
          awayHit += (tile.content as { pointsLength?: number } | undefined)?.pointsLength ?? 0;
      const awayShare = settledA > 0 ? awayHit / settledA : 1;
      const before = traffic();
      // Turn back over half a second, as a person would, then wait for view A to return.
      events.emit("motion", true);
      for (let i = 0; i <= 15; i++) {
        look(at, 180 + (180 * i) / 15);
        await frame();
      }
      events.emit("motion", false);
      const back = performance.now();
      const share = (): number => {
        const now = drawnTiles();
        let hit = 0;
        for (const tile of viewA)
          if (now.has(tile))
            hit += (tile.content as { pointsLength?: number } | undefined)?.pointsLength ?? 0;
        return settledA > 0 ? hit / settledA : 1;
      };
      const firstFrameShare = share();
      let restoreMs: number | null = null;
      let restoreFrames: number | null = null;
      for (let f = 0; performance.now() - back < settleS * 1000; f++) {
        if (share() >= 0.95) {
          restoreMs = Math.round(performance.now() - back);
          restoreFrames = f;
          break;
        }
        await frame();
      }
      return {
        settledA,
        restoreMs,
        restoreFrames,
        firstFrameShare,
        awayShare,
        ...delta(before),
      };
    },
    async walk(speed, seconds, eyeM) {
      const at = eye(eyeM);
      look(at, 0);
      await settleFor(3);
      const before = traffic();
      updateMs.length = 0;
      const intervalMs: number[] = [];
      recording = true;
      events.emit("motion", true);
      const start = await frame();
      let last = start;
      for (;;) {
        const now = await frame();
        intervalMs.push(now - last);
        last = now;
        const t = (now - start) / 1000;
        if (t >= seconds) break;
        camera.moveForward((speed * (intervalMs[intervalMs.length - 1] ?? 0)) / 1000);
      }
      recording = false;
      events.emit("motion", false);
      return { updateMs: [...updateMs], intervalMs, ...delta(before) };
    },
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
        drawn: primitive?._liveSplats ?? primitive?._numSplats ?? 0,
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
