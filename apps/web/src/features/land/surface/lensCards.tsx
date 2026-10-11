import type { ComponentType } from "react";

import type { LandArea } from "@twin/contracts";

import type { LensId } from "./lenses";
import { AssetsLens } from "./lenses/AssetsLens";
import { EcologyLens } from "./lenses/EcologyLens";
import { OverviewLens } from "./lenses/OverviewLens";
import { PlansLens } from "./lenses/PlansLens";
import { RecordsLens } from "./lenses/RecordsLens";
import { SoilLens } from "./lenses/SoilLens";
import { SolarLens } from "./lenses/SolarLens";
import { WaterLens } from "./lenses/WaterLens";

/** Each lens's card, by lens. A lens module owns its card, its map layers and its pins. */
export const LENS_CARDS: Record<LensId, ComponentType<{ land: LandArea }>> = {
  overview: OverviewLens,
  water: WaterLens,
  soil: SoilLens,
  ecology: EcologyLens,
  solar: SolarLens,
  records: RecordsLens,
  assets: AssetsLens,
  plans: PlansLens,
};
