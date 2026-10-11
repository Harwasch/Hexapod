import type { LandArea } from "@twin/contracts";

import { LandScenarios } from "../../LandScenarios";
import { LensCard } from "../LensCard";

/** Interim: the tools this lens replaces, until its map-first card lands. */
export function SolarLens({ land }: { land: LandArea }) {
  return (
    <LensCard lens="solar">
      <LandScenarios land={land} />
    </LensCard>
  );
}
