/**
 * One splat model's sunlight relight (`splatSunlight.ts`), on while the sun is (state/lighting.ts)
 * and As captured is not -- `sunlightSlot.ts` says which and loads this on the first switch, so
 * the globe's first load carries none of it.
 *
 * The part joins the primitive's colour chain while on and leaves it when off (the draw command
 * is rebuilt either way); the primitive appears with the first tile, so a tile load places it
 * again. Each frame drawn while on, the sun's height at the scan's centre is worked out for
 * that frame's scene time (`sunSine`, the engine's own sun) and set as a uniform, and what was
 * drawn is reported (state/sunReport.ts) for the badge: the minute and the sun's height. The
 * scene renders on request, so while on it is asked for a frame each minute: "now" keeps up
 * with the sun even when nothing else moves.
 */

import { JulianDate, type Cesium3DTileset, type Scene } from "cesium";

import { sunUpFactor } from "@/lib/splatNormals";
import { useSunReport, type SunReport } from "@/state/sunReport";

import type { ColorPrimitive } from "./splatColor";
import { splatTilesetOf } from "./splatInternals";
import { SplatSunlight, sunSine, type Vec4Factory } from "./splatSunlight";

export interface SunlightHandle {
  sync(on: boolean): void;
  dispose(): void;
}

/** How often, while on, the scene is asked for a frame so the sun of now moves on (ms). */
export const SUN_REFRESH_MS = 60_000;

/** `time` to the minute, ISO 8601: what the badge names. */
function minuteOf(time: JulianDate): string {
  const ms = JulianDate.toDate(time).getTime();
  return new Date(Math.floor(ms / 60_000) * 60_000).toISOString();
}

export function startSunlight(
  tileset: Cesium3DTileset,
  scene: Pick<Scene, "requestRender" | "preRender">,
  assetId: string,
  options: { vec4?: Vec4Factory } = {},
): SunlightHandle {
  const part = new SplatSunlight(options.vec4);
  const report = useSunReport.getState().report;
  let on = false;
  let disposed = false;
  let last: SunReport | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;

  const primitive = (): ColorPrimitive | undefined =>
    splatTilesetOf(tileset).gaussianSplatPrimitive;

  const place = (): void => {
    if (disposed) return;
    if (!on) {
      if (part.installed) {
        part.uninstall();
        scene.requestRender();
      }
      return;
    }
    const target = primitive();
    if (target && !part.installed && part.install(target)) scene.requestRender();
  };

  const frame = (_scene: unknown, time: JulianDate): void => {
    if (!on || !part.installed) return;
    const sine = sunSine(time, tileset.boundingSphere.center);
    part.sunUp = Number.isFinite(sine) ? sunUpFactor(sine) : 1;
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

  const removeTileLoad = tileset.tileLoad.addEventListener(place);
  const removeFrame = scene.preRender.addEventListener(frame);
  return {
    sync(next) {
      if (disposed || next === on) return;
      on = next;
      place();
      stopTimer();
      if (on) {
        timer = setInterval(() => scene.requestRender(), SUN_REFRESH_MS);
      } else {
        last = undefined;
        report(assetId, null);
      }
      scene.requestRender();
    },
    dispose() {
      on = false;
      place();
      disposed = true;
      stopTimer();
      removeTileLoad();
      removeFrame();
      report(assetId, null);
    },
  };
}
