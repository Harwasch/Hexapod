import type { components } from "@twin/contracts";

type Run = Pick<components["schemas"]["RunRead"], "status" | "error">;
type Observation = Pick<
  components["schemas"]["VegetationObservation"],
  "validCells" | "acquiredAt" | "explanation"
>;

export const FETCHING_LOOK = "Fetching the latest satellite look…";

/**
 * What the fog legend says about a "Fetch latest satellite look" run. A finished run is told
 * from what it stored, never from its status: a 30-day window under cloud or snow still
 * "succeeds" with a raster of no clear cells, which the fog does not count, and a "partial"
 * run may have stopped before anything was saved.
 *
 * `stored` is the run's dated observations: undefined while they are read, null when the run
 * stored no raster. `readError` is why they could not be read.
 */
export function satelliteLookNote(
  run: Run | undefined,
  stored: readonly Observation[] | null | undefined,
  readError: string | null = null,
): string | null {
  if (!run) return null;
  if (run.status === "queued" || run.status === "running") return FETCHING_LOOK;
  if (run.status === "failed")
    return `The satellite look failed: ${run.error ?? "no reason given"}.`;
  if (run.status === "cancelled") return null;
  if (readError)
    return `The satellite look finished, but what it stored could not be read: ${readError}`;
  if (stored === undefined) return "Checking what the satellite look stored…";
  if (stored === null) {
    return "The satellite look stopped before a scene was stored. Nothing was added to the fog.";
  }
  const clear = stored.find((item) => item.validCells > 0);
  if (clear) {
    const day = clear.acquiredAt?.slice(0, 10);
    return day
      ? `Sentinel-2 scene of ${day} stored. The fog now counts it.`
      : "Satellite look stored. The fog now counts it.";
  }
  const why = stored[0]?.explanation;
  return `No clear satellite scene in the last 30 days, so nothing was added to the fog.${
    why ? ` ${why}` : ""
  }`;
}
