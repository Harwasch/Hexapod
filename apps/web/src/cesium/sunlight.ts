/**
 * One splat model's sunlight relight (`splatSunlight.ts`), on while the sun is (state/lighting.ts)
 * and As captured is not -- `sunlightSlot.ts` says which and loads this on the first switch, so
 * the globe's first load carries none of it -- and while the splat renderer draws it
 * (lib/viewModes.ts `sunlightDrawable`): CesiumJS in its splat primitive, PlayCanvas in its
 * per-frame hook (scanView/sunHook.ts). Under Spark or the WebGPU trial nothing is relit, as
 * the control, which is disabled there, and the badge, which says nothing there, have it.
 *
 * Under CesiumJS the part joins the primitive's colour chain while on and leaves it when off
 * (the draw command is rebuilt either way); the primitive appears with the first tile, so a
 * tile load places it again. Under PlayCanvas the tileset is kept hidden and the renderer draws
 * the scan: this hands it the sun instead (scanView/scanSunlight.ts), in the scan's frame.
 *
 * Each frame drawn while on, the sun at the scan's centre is worked out for that frame's scene
 * time (`sunSine`, `sunInFrame`: the engine's own sun) and set -- a uniform of the part, or the
 * sun handed over -- and what was drawn is reported (state/sunReport.ts) for the badge: the
 * minute and the sun's height, while the model is on screen (a hidden tileset under CesiumJS,
 * a scan no relighting renderer draws under PlayCanvas, withdraws its report). The scene
 * renders on request, so while on it is asked for a frame each minute: "now" keeps up with the
 * sun even when nothing else moves.
 */

import { JulianDate, Matrix4, type Cesium3DTileset, type Scene } from "cesium";

import { sunUpFactor } from "@/lib/splatNormals";
import { sunlightDrawable } from "@/lib/viewModes";
import { useRendererOverride, useSettings } from "@/state/settings";
import { useSunReport, type SunReport } from "@/state/sunReport";

import { onScanSunDrawers, publishScanSun, scanSunDrawn } from "./scanView/scanSunlight";
import type { ColorPrimitive } from "./splatColor";
import { splatTilesetOf } from "./splatInternals";
import { SplatSunlight, sunInFrame, sunSine, type Vec4Factory } from "./splatSunlight";
import { inverseScaledTransformation } from "./tilesetScale";

export interface SunlightHandle {
  sync(on: boolean): void;
  dispose(): void;
}

/** How often, while on, the scene is asked for a frame so the sun of now moves on (ms). */
export const SUN_REFRESH_MS = 60_000;

/** Who draws the relight now: CesiumJS's part, a dedicated renderer's hook, or nobody. */
type Drawer = "cesium" | "dedicated" | null;

/** `time` to the minute, ISO 8601: what the badge names. */
function minuteOf(time: JulianDate): string {
  const ms = JulianDate.toDate(time).getTime();
  return new Date(Math.floor(ms / 60_000) * 60_000).toISOString();
}

/** The photograph: what a renderer is handed for a frame with no sun to light by. */
const UNLIT = { dir: [0, 0, 1], sunUp: 1, strength: 0 } as const;

export function startSunlight(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "requestRender" | "preRender">,
  assetId: string,
  options: { vec4?: Vec4Factory } = {},
): SunlightHandle {
  const part = new SplatSunlight(options.vec4);
  const report = useSunReport.getState().report;
  const drawer = (): Drawer => {
    const renderer =
      useRendererOverride.getState().renderer ?? useSettings.getState().splatRenderer;
    if (!sunlightDrawable(renderer)) return null;
    return renderer === "cesium" ? "cesium" : "dedicated";
  };
  const toLocal = new Matrix4();
  let asked = false;
  let on: Drawer = null;
  let disposed = false;
  let last: SunReport | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;

  const primitive = (): ColorPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;

  const place = (): void => {
    if (disposed) return;
    if (on !== "cesium") {
      if (part.installed) {
        part.uninstall();
        scene.requestRender();
      }
      return;
    }
    const target = primitive();
    if (target && !part.installed && part.install(target)) scene.requestRender();
  };

  const withdraw = (): void => {
    if (!last) return;
    last = undefined;
    report(assetId, null);
  };

  /** The sun in the scan's frame, as the renderer's scene has it (scanView/pose.ts). */
  const scanFrameSun = (time: JulianDate): readonly [number, number, number] | null => {
    const root = (tileset as { root?: { computedTransform?: Matrix4 } }).root;
    if (!root?.computedTransform) return null;
    inverseScaledTransformation(root.computedTransform, toLocal);
    return sunInFrame(time, toLocal);
  };

  const frame = (_scene: unknown, time: JulianDate): void => {
    if (!on) return;
    if (on === "cesium" && !part.installed) return;
    // A model not on screen draws no relight, so it says none: the badge and the clock it holds
    // follow only what is on screen -- under CesiumJS its tileset shown (another date shown,
    // the site out of reach hide it), under PlayCanvas a renderer drawing it. Nor does a frame
    // with no sun to light by (NaN: not placed, the engine's sun not a number), which draws the
    // colours untouched (strength 0).
    const shown = on === "cesium" ? tileset.show : scanSunDrawn(assetId);
    const sine = shown ? sunSine(time, tileset.boundingSphere.center) : Number.NaN;
    const dir = on === "dedicated" && Number.isFinite(sine) ? scanFrameSun(time) : null;
    const lit = Number.isFinite(sine) && (on === "cesium" || dir !== null);
    if (on === "cesium") {
      part.sunUp = lit ? sunUpFactor(sine) : Number.NaN;
    } else {
      publishScanSun(assetId, lit && dir ? { dir, sunUp: sunUpFactor(sine), strength: 1 } : UNLIT);
    }
    if (!lit) {
      withdraw();
      return;
    }
    const at = minuteOf(time);
    if (last?.at !== at || last.sine > 0 !== sine > 0) {
      last = { at, sine };
      report(assetId, last);
    }
  };

  const stopTimer = (): void => {
    if (timer !== undefined) clearInterval(timer);
    timer = undefined;
  };

  const apply = (): void => {
    const next = asked && !disposed ? drawer() : null;
    if (disposed || next === on) return;
    const before = on;
    on = next;
    place();
    if (before === "dedicated") publishScanSun(assetId, null);
    stopTimer();
    withdraw();
    if (on) timer = setInterval(() => scene.requestRender(), SUN_REFRESH_MS);
    scene.requestRender();
  };

  const removeTileLoad = tileset.tileLoad.addEventListener(place);
  const removeFrame = scene.preRender.addEventListener(frame);
  // A renderer that starts drawing the scan gets its sun in the next frame, and the badge with
  // it; one that stops takes its report with it at the next frame.
  const removeDrawers = onScanSunDrawers(assetId, () => {
    if (on === "dedicated") scene.requestRender();
  });
  const unsubscribe = [useSettings.subscribe(apply), useRendererOverride.subscribe(apply)];
  return {
    sync(next) {
      asked = next;
      apply();
    },
    dispose() {
      asked = false;
      for (const off of unsubscribe) off();
      apply();
      on = null;
      place();
      disposed = true;
      stopTimer();
      removeTileLoad();
      removeFrame();
      removeDrawers();
      report(assetId, null);
    },
  };
}
