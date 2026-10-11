import type { LandArea } from "@twin/contracts";

import { LandActions } from "../../LandActions";
import { LensCard } from "../LensCard";

/** Interim: the tools this lens replaces, until its map-first card lands. */
export function PlansLens({ land }: { land: LandArea }) {
  return (
    <LensCard lens="plans">
      <LandActions land={land} />
    </LensCard>
  );
}
