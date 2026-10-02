/**
 * How finely the splat overlay draws: its resolution, the smallest splat it keeps, and the
 * spherical-harmonic bands a tile keeps -- held to the same rules as the globe under it.
 *
 * Resolution used to be the overlay's own: up to 2 device pixels per CSS pixel (1.5 on a
 * phone or tablet), cut to 60% while the camera moved, whatever the quality preset and the
 * adaptive ladder said. So on the performance preset, where the globe renders one device
 * pixel per CSS pixel, a 2x display blended every splat over four times the globe's pixels,
 * and a machine the ladder had already cut to half resolution kept paying full price for the
 * scan. The globe's resolution is the PerformanceManager's whole policy in one number --
 * CSS or device pixels by preset, Balanced's 1.5 and short-side caps while moving, the full
 * device ratio once a still frame is sharpened, the ladder's 0.8 / 0.65 / 0.5 -- and CesiumJS
 * applies it as `(useBrowserRecommendedResolution ? 1 : devicePixelRatio) * resolutionScale`.
 * The overlay now takes that number as its own ceiling, under its own caps, and keeps its
 * motion cut where that is lower.
 */

/** Most device pixels per CSS pixel the overlay draws: desktop, and a phone or tablet. */
export const OVERLAY_MAX_PIXEL_RATIO = { desktop: 2, handheld: 1.5 } as const;
/** Resolution while the camera moves, as a share of the resting one (never below
 *  `MIN_MOTION_PIXEL_RATIO`): blending splats is per pixel, and the GPU is what runs out. */
export const MOTION_RESOLUTION = 0.6;
export const MIN_MOTION_PIXEL_RATIO = 0.75;

/** What the globe renders at, as CesiumWidget computes it. */
export interface GlobeResolution {
  devicePixelRatio: number;
  /** `Viewer.useBrowserRecommendedResolution`: one device pixel per CSS pixel. */
  browserRecommended: boolean;
  /** `Viewer.resolutionScale`: the preset's base scale times the ladder's step. */
  resolutionScale: number;
}

/** Device pixels per CSS pixel the globe renders at now. */
export function globePixelRatio(globe: GlobeResolution): number {
  const base = globe.browserRecommended ? 1 : Math.max(globe.devicePixelRatio || 1, 0.25);
  return base * (globe.resolutionScale > 0 ? globe.resolutionScale : 1);
}

/**
 * Device pixels per CSS pixel the overlay draws at: never more than the globe renders at now
 * (preset, ladder, sharpened still frame) nor than its own cap, and while the camera moves its
 * own motion cut -- 60% of the preset's full ratio, at least 0.75 -- where that is lower.
 */
export function overlayPixelRatio(
  globe: GlobeResolution,
  options: { handheld: boolean; moving: boolean },
): number {
  const cap = options.handheld ? OVERLAY_MAX_PIXEL_RATIO.handheld : OVERLAY_MAX_PIXEL_RATIO.desktop;
  const now = Math.min(globePixelRatio(globe), cap);
  if (!options.moving) return now;
  const full = Math.min(globePixelRatio({ ...globe, resolutionScale: 1 }), cap);
  return Math.min(now, Math.max(MIN_MOTION_PIXEL_RATIO, full * MOTION_RESOLUTION));
}

/**
 * The smallest splat PlayCanvas keeps, in device pixels, at `pixelRatio`: half a CSS pixel,
 * whatever the resolution. PlayCanvas measures `minPixelSize` in the pixels it renders, so the
 * one device pixel that suited a 2x canvas would drop every splat under a whole CSS pixel at
 * 1x -- fine detail (grass, wires) thinning out at the lower preset instead of only softening.
 */
export const MIN_SPLAT_CSS_PX = 0.5;
export function splatMinPixelSize(pixelRatio: number): number {
  return MIN_SPLAT_CSS_PX * pixelRatio;
}

/**
 * Spherical-harmonic bands a tile keeps in a dedicated renderer: all three on a desktop, the
 * first on a phone or tablet. Each band costs memory on every splat (degree 3 is 45 floats, 180
 * bytes a splat, more than everything else together) and shading work on every pixel; a
 * phone's memory and fill rate are a fraction of a desktop's, and band 1 keeps most of the
 * view-dependent shine. Tiles carry none today; the pipeline may soon write degree 1 to 3.
 */
export const MAX_SH_DEGREE = { desktop: 3, handheld: 1 } as const;
export function maxShDegree(handheld: boolean): number {
  return handheld ? MAX_SH_DEGREE.handheld : MAX_SH_DEGREE.desktop;
}
