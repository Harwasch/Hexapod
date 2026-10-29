/**
 * Spherical-harmonic bands a tile keeps in a dedicated renderer: all three on a desktop, the
 * first on a phone or tablet. Each band costs memory on every splat (degree 3 is 45 floats, 180
 * bytes a splat, more than everything else together) and shading work on every pixel; a
 * phone's memory and fill rate are a fraction of a desktop's, and band 1 keeps most of the
 * view-dependent shine. Captures ship degree 3 (the pipeline's recipes), so this cap is what
 * keeps a phone at band 1.
 *
 * Here rather than in `cesium/scanView/quality.ts` because the scan gallery (`view.html`)
 * applies it too, and that page must not request anything under a path naming Cesium.
 */
export const MAX_SH_DEGREE = { desktop: 3, handheld: 1 } as const;
export function maxShDegree(handheld: boolean): number {
  return handheld ? MAX_SH_DEGREE.handheld : MAX_SH_DEGREE.desktop;
}
