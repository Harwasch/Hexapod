import type { KnowledgeClassId } from "@twin/contracts";

/** The fog-v1 classes, finest first (the API's policy table carries the same order). */
export const CLASS_ORDER: readonly KnowledgeClassId[] = [
  "centimetre",
  "field",
  "aerial",
  "satellite",
  "regional",
];

const LABELS: Record<KnowledgeClassId | "none", string> = {
  centimetre: "Centimetre capture",
  field: "Field inspection",
  aerial: "Aerial imagery",
  satellite: "Satellite",
  regional: "Regional or unlocated",
  none: "No record",
};

export function classLabel(id: string): string {
  return LABELS[id as KnowledgeClassId | "none"] ?? id;
}

/** A record's age as the card says it: "undated", "today", "1 day old", "2.1 years old". */
export function agePhrase(days: number | null | undefined): string {
  if (days == null) return "undated";
  if (days < 1) return "today";
  if (days < 365) {
    const whole = Math.round(days);
    return `${whole} day${whole === 1 ? "" : "s"} old`;
  }
  return `${(days / 365.25).toFixed(1)} years old`;
}
