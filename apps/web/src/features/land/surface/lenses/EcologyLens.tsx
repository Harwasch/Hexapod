import type { LandArea } from "@twin/contracts";

import { LandEcology } from "../../LandEcology";
import { LensCard } from "../LensCard";

/** Interim: the tools this lens replaces, until its map-first card lands. */
export function EcologyLens({ land }: { land: LandArea }) {
  return (
    <LensCard lens="ecology">
      <LandEcology land={land} />
    </LensCard>
  );
}
