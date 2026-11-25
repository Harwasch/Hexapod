/**
 * The phone's "Detail" choice, read wherever a splat is drawn.
 *
 * It used to be a packaging cut: `package.max_gaussians` kept the 200k / 400k / 800k most
 * opaque gaussians of a scan and threw the rest away, for every viewer, for good. The
 * pipeline now packs every gaussian as a level-of-detail tileset
 * (tools/captures/splat_tiles.py), so how many are drawn is decided on the device that pays
 * for drawing them, and the choice is this device's, not the scan's:
 *
 * - the scan viewer (view.html, Spark) hands it to Spark's own level of detail as
 *   `SparkRenderer.lodSplatCount`, the most splats drawn a frame, and downloads up to
 *   `LOAD_FACTOR` times as many (`view/tiles.ts`);
 * - the globe (CesiumJS) scales a splat tileset's `maximumScreenSpaceError` by
 *   `detailScreenSpaceScale`, since Cesium refines by screen-space error, and counts the
 *   splats it has loaded against it as a memory budget (`cesium/splatCount.ts`), since
 *   Cesium's own byte count misses splats.
 *
 * The upload page is where it is chosen (`upload/options.ts`), and the three pages share an
 * origin, so the choice saved there is read here.
 */

import { readJson } from "./storage";

/** Where `upload/options.ts` saves the phone's choices, `detail` among them. */
export const OPTIONS_STORAGE = "twin.phoneOptions.v3";

/** All scan-detail tuning lives here. Counts are splats, times milliseconds, caches bytes.
 * iPhone classification is a conservative capability heuristic, not a model guarantee.
 * Unknown/older handhelds retain their starting budgets. Growth is bounded by the chosen
 * Detail tier; phones cut sooner and harder than desktops. Desktop policy is unchanged.
 * Embed pressure is sticky for the page/session: one halving, never automatic regrowth.
 */
export const SCAN_DETAIL_CONFIG = {
  budgets: {
    handheld: [200_000, 400_000, 800_000],
    recentIPhone: [750_000, 1_500_000, 2_500_000],
    desktop: [
      [8, 3_000_000],
      [4, 1_500_000],
      [0, 800_000],
    ],
  },
  // Allow four reported logical cores; require both the screen and GPU signals too.
  detection: {
    handheldShortSide: 900,
    iphoneShortSide: 390,
    iphoneLongSide: 844,
    pixelRatio: 3,
    cores: 4,
    maxTextureSize: 16_384,
    defaultDesktopMemoryGb: 8,
  },
  growth: {
    factor: 2,
    desktopCeiling: 8_000_000,
    handheldCeiling: 1_600_000,
    recentIPhoneCeiling: 3_500_000,
  },
  // Slow decisions use consecutive phone frames; growth uses a median window.
  adaptive: {
    windowFrames: 45,
    phoneSlowFrames: 12,
    slowFps: 30,
    fastFps: 50,
    cut: 0.8,
    phoneCut: 0.5,
    raise: 1.15,
    floorShare: 0.3,
    minSplats: 300_000,
    phoneMinSplats: 100_000,
    slowDrawShare: 0.6,
    fastDrawShare: 0.8,
    maxIntervalMs: 2000,
  },
  embed: {
    cacheBytes: 320 * 1024 * 1024,
    maximumCacheOverflowBytes: 80 * 1024 * 1024,
    maximumScreenSpaceError: 8,
    pressureCacheFactor: 0.5,
    pressureErrorFactor: 2,
    heapLimitShare: 0.85,
    heapPollMs: 1000,
  },
} as const;

// Stored wire values remain stable: they identify the user's Light/Standard/Full choice.
export const DETAIL_BUDGETS = SCAN_DETAIL_CONFIG.budgets.handheld;
export const DEFAULT_SPLAT_BUDGET = DETAIL_BUDGETS[1];
export function splatBudget(): number {
  const stored = readJson(OPTIONS_STORAGE) as { detail?: unknown } | undefined;
  const budget = Number(stored?.detail);
  return (DETAIL_BUDGETS as readonly number[]).includes(budget) ? budget : DEFAULT_SPLAT_BUDGET;
}
export function detailScreenSpaceScale(budget: number): number {
  return Math.sqrt(DEFAULT_SPLAT_BUDGET / Math.max(1, budget));
}
export type DeviceClass = "desktop" | "handheld" | "recentIPhone";
export interface DeviceSignals {
  handheld: boolean;
  iphone: boolean;
  shortSide: number;
  longSide: number;
  pixelRatio: number;
  cores: number;
  maxTextureSize: number;
}
export function classifyDevice(s: DeviceSignals): DeviceClass {
  if (!s.handheld) return "desktop";
  const d = SCAN_DETAIL_CONFIG.detection;
  return s.iphone &&
    s.shortSide >= d.iphoneShortSide &&
    s.longSide >= d.iphoneLongSide &&
    s.pixelRatio >= d.pixelRatio &&
    s.cores >= d.cores &&
    s.maxTextureSize >= d.maxTextureSize
    ? "recentIPhone"
    : "handheld";
}
export function isHandheld(): boolean {
  try {
    return (
      window.matchMedia("(pointer: coarse)").matches &&
      Math.min(screen.width, screen.height) < SCAN_DETAIL_CONFIG.detection.handheldShortSide
    );
  } catch {
    return false;
  }
}
let textureSize: number | undefined;
export function deviceClass(): DeviceClass {
  if (!isHandheld()) return "desktop";
  const iphone = navigator.userAgent.includes("iPhone");
  // Probe once, only on iPhones, and release the temporary GPU context immediately.
  if (iphone && textureSize === undefined) {
    textureSize = 0;
    try {
      const gl = document.createElement("canvas").getContext("webgl2");
      if (gl) {
        textureSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE));
        gl.getExtension("WEBGL_lose_context")?.loseContext();
      }
    } catch {
      /* Missing capabilities retain the conservative class. */
    }
  }
  return classifyDevice({
    handheld: true,
    iphone,
    shortSide: Math.min(screen.width, screen.height),
    longSide: Math.max(screen.width, screen.height),
    pixelRatio: window.devicePixelRatio,
    cores: navigator.hardwareConcurrency ?? 0,
    maxTextureSize: textureSize ?? 0,
  });
}
export function deviceSplatBudget(kind = deviceClass()): number {
  const detail = splatBudget();
  if (kind !== "desktop") {
    const index = DETAIL_BUDGETS.findIndex((value) => value === detail);
    return SCAN_DETAIL_CONFIG.budgets[kind][index] ?? SCAN_DETAIL_CONFIG.budgets[kind][1];
  }
  const memory =
    (typeof navigator !== "undefined"
      ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory
      : undefined) ?? SCAN_DETAIL_CONFIG.detection.defaultDesktopMemoryGb;
  const standard =
    SCAN_DETAIL_CONFIG.budgets.desktop.find(([minGb]) => memory >= minGb)?.[1] ??
    SCAN_DETAIL_CONFIG.budgets.desktop[2][1];
  return Math.round(standard * (detail / DEFAULT_SPLAT_BUDGET));
}
export function deviceSplatCeiling(kind = deviceClass()): number {
  const g = SCAN_DETAIL_CONFIG.growth;
  const cap =
    kind === "desktop"
      ? g.desktopCeiling
      : kind === "recentIPhone"
        ? g.recentIPhoneCeiling
        : g.handheldCeiling;
  return Math.min(deviceSplatBudget(kind) * g.factor, cap);
}
