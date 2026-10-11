import type { LandArea } from "@twin/contracts";

import { LandResearch } from "../../LandResearch";
import { LensCard } from "../LensCard";

/** Interim: the tools this lens replaces, until its map-first card lands. */
export function SoilLens({ land }: { land: LandArea }) {
  return (
    <LensCard lens="soil">
      <LandResearch land={land} />
    </LensCard>
  );
}
