import type { LandCreate } from "@twin/contracts";

import { OSM_ATTRIBUTION, type OsmCandidate } from "@/missions/osm";

/**
 * A mapped OpenStreetMap area as an unsaved land: what "Find the mapped area here" proposes
 * after its click (LandPanel's pick mode), and what the Descent's "Outline this land" proposes
 * on arrival. One builder, so both say the same thing about where the outline came from: a
 * physical feature someone mapped, not a parcel or a property line.
 */
export function mappedFeatureDraft(candidate: OsmCandidate, description = ""): LandCreate {
  return {
    name: candidate.name,
    description,
    boundary: candidate.footprint,
    source: {
      method: "mapped-feature",
      meaning: "physical-feature",
      label: "OpenStreetMap feature",
      attribution: OSM_ATTRIBUTION,
      url: "https://www.openstreetmap.org/copyright",
    },
  };
}
