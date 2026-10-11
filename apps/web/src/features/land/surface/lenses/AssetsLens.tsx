import type { LandArea } from "@twin/contracts";

import { LandInventory } from "../../LandInventory";
import { LensCard } from "../LensCard";

/** Interim: the tools this lens replaces, until its map-first card lands. */
export function AssetsLens({ land }: { land: LandArea }) {
  return (
    <LensCard lens="assets">
      <LandInventory land={land} />
    </LensCard>
  );
}
