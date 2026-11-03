import { ARTICULATION_FILE, besideTileset } from "./weatherWindPlan";

/** Per asset id: whether a baked rig sits beside its tiles. Probed once per page. */
const rigProbes = new Map<string, Promise<boolean>>();

/**
 * Asks whether `articulation.json` is published beside the tiles. A host that answers every path
 * with its HTML page (a single-page fallback) is not a rig, so the type must be JSON.
 */
export function probeRig(assetId: string, tileset: string): Promise<boolean> {
  let probe = rigProbes.get(assetId);
  if (!probe) {
    const url = besideTileset(tileset, ARTICULATION_FILE);
    probe = url
      ? fetch(url, { method: "HEAD" })
          .then((r) => r.ok && (r.headers.get("content-type") ?? "").includes("json"))
          .catch(() => false)
      : Promise.resolve(false);
    rigProbes.set(assetId, probe);
  }
  return probe;
}

/** Test hook: forget earlier probes. */
export function forgetRigProbes(): void {
  rigProbes.clear();
}
