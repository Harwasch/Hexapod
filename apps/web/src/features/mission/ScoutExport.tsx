import { lazy, Suspense } from "react";

import { env } from "@/app/env";
import type { Zone } from "@/missions/types";

/** The drone card (DJI route, heights, batteries) loads only with drone missions on. */
const DroneScoutExport = lazy(() =>
  import("./DroneScoutExport").then((m) => ({ default: m.DroneScoutExport })),
);

/**
 * The scout route of a draft or plan's scout areas. Drone missions are a stub until there is
 * a drone (`env.droneExportEnabled`, off; docs/DRONES.md): the card then offers no export,
 * heights or battery figures, only a note, and loads and samples nothing. With them on it is
 * `DroneScoutExport`, the DJI route.
 */
export function ScoutExport(props: {
  zones: readonly Zone[];
  ground?: readonly Zone[];
  onAltitude?: (altitudeM: number) => void;
}) {
  if (!props.zones.some((z) => z.scout)) return null;
  if (env.droneExportEnabled)
    return (
      <Suspense fallback={null}>
        <DroneScoutExport {...props} />
      </Suspense>
    );
  return (
    <section className="mc-scout" data-testid="scout-export" aria-label="Scout route">
      <p className="mc-muted" data-testid="scout-drone-stub">
        Drone missions: not available yet
      </p>
    </section>
  );
}
