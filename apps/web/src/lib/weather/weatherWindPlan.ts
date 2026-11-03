/**
 * Which scan on the active site weather wind moves, and from which sidecar. Pure, so the
 * choice is tested without a renderer (features/living/WeatherWindDriver.tsx drives it).
 *
 * Order, most detailed first:
 * 1. a baked branch rig beside the tiles (`articulation.json`): its header's detail score picks
 *    Tier B or C (`motionTier`, cesium/scanView/livingDetail.ts);
 * 2. published segmentation (`instances.json`): Tier A, whole-plant sway of vegetation clumps;
 * 3. neither: off. There is deliberately no colour-only fallback: green tables moved in the
 *    research round (docs/LIVING_WEATHER_WIND.md, "Detail check").
 *
 * Only the PlayCanvas WebGL2 renderer has the hook. A Living Survey scan (`renderConfig.rigUrl`)
 * is drawn by CesiumJS with its own Simulated wind, so weather wind leaves it alone.
 */

export type WeatherWindTier = "off" | "A" | "B" | "C";

export interface WindScan {
  assetId: string;
  name: string;
  /** The tileset's address; empty for a scan not served as 3D Tiles by URL. */
  tileset: string;
  splat: boolean;
  /** Has a Living Survey rig (CesiumJS draws it). */
  rigged: boolean;
  /** Its `instances.json` loaded. */
  segmented: boolean;
}

export type WindPlan =
  | {
      kind: "target";
      assetId: string;
      name: string;
      /** The articulation address the renderer loads (before the weather query is added). */
      url: string;
      source: "rig" | "segmentation";
    }
  /** A rig probe is still out: wait rather than settle for a lower tier. */
  | { kind: "pending" }
  | { kind: "none"; reason: string };

export const ARTICULATION_FILE = "articulation.json";

/** `file` beside the tileset, or "" when the address cannot be read. */
export function besideTileset(tileset: string, file: string): string {
  try {
    const base = typeof window === "undefined" ? undefined : window.location.href;
    return new URL(file, new URL(tileset, base)).toString();
  } catch {
    return "";
  }
}

export function planWeatherWind(
  scans: readonly WindScan[],
  /** Per asset id: true when its `articulation.json` exists, false when not, absent while probing. */
  rigFound: Readonly<Record<string, boolean | undefined>>,
  renderer: string,
): WindPlan {
  if (renderer !== "playcanvas")
    return { kind: "none", reason: "needs the PlayCanvas renderer (Settings › Advanced)" };
  const usable = scans.filter((s) => s.splat && !s.rigged && s.tileset !== "");
  if (usable.length === 0)
    return {
      kind: "none",
      reason: scans.some((s) => s.rigged)
        ? "this site moves with Simulated wind above"
        : "no splat scan on this site",
    };
  for (const scan of usable)
    if (rigFound[scan.assetId] === true) {
      const url = besideTileset(scan.tileset, ARTICULATION_FILE);
      if (url)
        return { kind: "target", assetId: scan.assetId, name: scan.name, url, source: "rig" };
    }
  if (usable.some((s) => rigFound[s.assetId] === undefined)) return { kind: "pending" };
  for (const scan of usable)
    if (scan.segmented) {
      const url = besideTileset(scan.tileset, "instances.json");
      if (url)
        return {
          kind: "target",
          assetId: scan.assetId,
          name: scan.name,
          url: `${url}?tier-a=1`,
          source: "segmentation",
        };
    }
  return { kind: "none", reason: "no vegetation data for this scan" };
}

export const TIER_TEXT: Readonly<Record<WeatherWindTier, string>> = {
  A: "Tier A · whole plants sway",
  B: "Tier B · branches move",
  C: "Tier C · branches and leaves move",
  off: "Off · no vegetation found",
};
