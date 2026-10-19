import type { LandCreate } from "@twin/contracts";

type Source = LandCreate["source"];

/** Where the outline came from, in a few words. */
export function sourceLabel(source: Source): string {
  switch (source.method) {
    case "drawn":
      return source.label === "Adjusted on the map" ? "Adjusted by you" : "Drawn by you";
    case "parcel":
      return "From parcel records";
    case "mapped-feature":
      return "From the map";
    case "corridor":
      return "Corridor";
    case "imported":
      return "Imported";
    default:
      return source.label;
  }
}

/** What the outline does and does not mean, for the info line. */
export function sourceMeaning(meaning: Source["meaning"]): string {
  if (meaning === "recorded-parcel")
    return "A county tax-parcel record. It is not a survey, so check legal lines with the county.";
  if (meaning === "physical-feature")
    return "Follows something on the ground, such as a field edge. It is not a property line.";
  return "An area you chose to study. It is not a property line.";
}
