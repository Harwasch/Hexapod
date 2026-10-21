/**
 * Where two dates compared side by side (SiteManager `setCompareDates`; ADR 0017 "Compare two
 * dates") meet PlayCanvas when PlayCanvas draws the site's scan: per asset id -- the right-hand
 * date, which is the scan the renderer draws (SiteManager `scanTarget`) -- the left-hand date
 * to stream beside it and where the divider is, or none. The comparison's session puts it here
 * (cesium/compareSession.ts); the renderer drawing that asset watches it (playcanvasBackend.ts).
 *
 * **One app, one cut.** The left date is streamed into the same app as the scan (ruleDates.ts,
 * as a per-cell rule's dates are), within the budget the scan draws to, and every splat of
 * either is sorted with the other's. Each tile's copy into the work buffer writes which date it
 * is of -- one byte a splat, `SIDE_STREAM`, from one uniform a tile, needing neither its
 * position nor its shape -- and the frame's own splat hook hands that byte to the fragment
 * shader (a flat varying), which discards a fragment of the left date at or right of the
 * divider and of the right date left of it, by `gl_FragCoord.x`. The divider is a uniform of the
 * frame: swiping copies nothing. A splat straddling the divider is cut exactly there, as
 * CesiumJS's split cuts it (`czm_splitPosition`).
 *
 * Both sides load on demand (the session with the first comparison, the renderer when chosen),
 * and so does this: nothing of it is on the globe's first load.
 */

import type { Matrix4 } from "cesium";

import type { RuleDate } from "./scanRule";

/** The work buffer's extra stream (R8U): PlayCanvas names its accessors `load/writeHexapodSide`. */
export const SIDE_STREAM = "hexapodSide";
/** The varying the frame's hook hands the fragment shader (`set/getHexapodHalf`). */
export const SIDE_VARYING = "hexapodHalf";
/** What a tile's copy writes: no comparison (or a tile not yet copied since), left, right. */
export const SIDE_NONE = 0;
export const SIDE_LEFT = 1;
export const SIDE_RIGHT = 2;

/** Two dates compared, as the renderer drawing the right-hand one is handed them. */
export interface ScanCompareView {
  /** The left-hand date, streamed beside the scan, placed in its frame. */
  left: RuleDate;
  /** The divider: the share of the canvas's width left of it (`scene.splitPosition`). */
  split: number;
  /** The scan's frame to Earth-fixed now (its root's computed transform): for the picks. */
  toWorld(): Matrix4 | undefined;
}

/**
 * What every tile's copy writes under a comparison (playcanvasBackend.ts modifiers): its side,
 * from `uHexapodSide` (`SIDE_LEFT` or `SIDE_RIGHT`; `SIDE_NONE` with none). The colour pass of a
 * natively streamed scan's refresh does not write it (PlayCanvas guards every other output),
 * so the byte its full copy wrote stays.
 */
export const SIDE_WRITE_GLSL = `
uniform float uHexapodSide;
`;
export const SIDE_WRITE_STEP = "writeHexapodSide(uvec4(uint(uHexapodSide + 0.5), 0u, 0u, 0u));";

/**
 * The frame's hook in the fragment shader (`gsplatModifyPS` on `app.scene.gsplat.material`):
 * `uHexapodDivider` is (divider x in the canvas's pixels, on). A fragment of the left date is
 * kept left of the divider only, the right date's right of it only; a splat of neither (no
 * comparison) everywhere.
 */
export const COMPARE_HOOK_PS = `
uniform vec4 uHexapodDivider;
void modifySplatColor(vec2 gaussianUV, inout vec4 color) {
#ifdef GSPLAT_USER_VARYINGS
    if (uHexapodDivider.y < 0.5) {
        return;
    }
    uint side = getHexapodHalf();
    bool left = gl_FragCoord.x < uHexapodDivider.x;
    if ((side == ${String(SIDE_LEFT)}u && !left) || (side == ${String(SIDE_RIGHT)}u && left)) {
        discard;
    }
#endif
}
`;

/**
 * The frame's vertex hook (`gsplatModifyVS`) under a comparison: hands each splat's side to the
 * fragment shader, then does what `inner` does (the sunlight's hook, or nothing). `inner` is a
 * whole hook -- its three functions -- whose `modifySplatColor` is renamed and called.
 */
export function compareHookVs(inner: string | null): string {
  const own = inner?.replace("void modifySplatColor(", "void hexapodInnerColor(") ?? null;
  return `
${
  own ??
  `void modifySplatCenter(inout vec3 center) {
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
}`
}
void modifySplatColor(vec3 center, inout vec4 color) {
#ifdef GSPLAT_USER_VARYINGS
    setHexapodHalf(loadHexapodSide().r);
#endif
${own ? "    hexapodInnerColor(center, color);" : ""}
}
`;
}

type Listener = (view: ScanCompareView | null) => void;

const views = new Map<string, ScanCompareView>();
const listeners = new Map<string, Set<Listener>>();

/** Puts asset `assetId`'s comparison (null: none) where the renderer drawing it reads it. */
export function publishScanCompare(assetId: string, view: ScanCompareView | null): void {
  if ((views.get(assetId) ?? null) === view) return;
  if (view) views.set(assetId, view);
  else views.delete(assetId);
  for (const listener of listeners.get(assetId) ?? []) listener(view);
}

/** Asset `assetId`'s comparison now, or null. */
export function scanCompareOf(assetId: string): ScanCompareView | null {
  return views.get(assetId) ?? null;
}

/** Calls `listener` with asset `assetId`'s comparison now and on every change; the remover. */
export function watchScanCompare(assetId: string, listener: Listener): () => void {
  let set = listeners.get(assetId);
  if (!set) listeners.set(assetId, (set = new Set()));
  set.add(listener);
  listener(views.get(assetId) ?? null);
  return () => {
    set.delete(listener);
    if (set.size === 0) listeners.delete(assetId);
  };
}

/** What the renderer drew of a comparison, last time it was asked. */
export interface ScanCompareReading {
  renderer: string;
  /** Whether it cuts two dates now. */
  on: boolean;
  /** The divider in the canvas's pixels, as the fragment shader has it. */
  dividerPx: number;
  /** Its tiles by the side each was last told (scan tiles, layers and objects alike). */
  sides: { left: number; right: number };
  /** The left date it streams: its tiles on screen, gaussians, fetches. */
  left: { dates: number; tiles: number; gaussians: number; loading: number };
}

const readings = new Map<string, () => ScanCompareReading>();

/** Registers how asset `assetId`'s renderer reads out its comparison; returns the remover. */
export function reportScanCompare(assetId: string, read: () => ScanCompareReading): () => void {
  readings.set(assetId, read);
  return () => {
    if (readings.get(assetId) === read) readings.delete(assetId);
  };
}

/** What the renderer drawing asset `assetId` says of its comparison, or null when none does. */
export function scanCompareReading(assetId: string): ScanCompareReading | null {
  return readings.get(assetId)?.() ?? null;
}
