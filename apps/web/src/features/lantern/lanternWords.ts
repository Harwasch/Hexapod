import type { DatedImageryProviderStatus } from "@twin/contracts";

import type { LanternUnavailable } from "@/state/lantern";

/** Why the lens cannot show, as the Look back button says it. Orbit and altitude hide it. */
export const UNAVAILABLE: Record<LanternUnavailable, string> = {
  "open-world": "Needs the 3D world (Settings › World)",
  webgl1: "Needs WebGL 2, which this browser does not offer",
  "clipping-off": "Turned off with the developer clipping switch",
  explore: "Not while exploring on foot or in flight",
  paint: "Not while painting objects with the brush",
  orbit: "Come closer to the ground",
  altitude: "Come closer to the ground",
};

export const LANTERN_HINT =
  "Hold Alt to peek · scroll over the map to travel in time · Esc to close";

const plural = (count: number, one: string, many: string) => (count === 1 ? one : many);

/**
 * The providers that could not answer and those with nothing here, counted apart: a source
 * that could not be reached is not a source with nothing to show.
 */
export function providersLine(providers: DatedImageryProviderStatus[]): string {
  const unreachable = providers.filter((p) => p.status === "unavailable").length;
  const empty = providers.filter((p) => p.status === "empty").length;
  const parts: string[] = [];
  if (unreachable)
    parts.push(`${unreachable} ${plural(unreachable, "source", "sources")} could not be reached`);
  if (empty)
    parts.push(
      unreachable
        ? `${empty} ${plural(empty, "has", "have")} nothing here`
        : `${empty} ${plural(empty, "source has", "sources have")} nothing to show here`,
    );
  return parts.join(", ");
}
