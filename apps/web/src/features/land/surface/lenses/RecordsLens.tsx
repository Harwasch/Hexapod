import type { LandArea } from "@twin/contracts";

import { LandDocuments } from "../../LandDocuments";
import { LensCard } from "../LensCard";

/** Interim: the tools this lens replaces, until its map-first card lands. */
export function RecordsLens({ land }: { land: LandArea }) {
  return (
    <LensCard lens="records">
      <LandDocuments land={land} />
    </LensCard>
  );
}
