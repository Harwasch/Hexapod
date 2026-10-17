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
