import type { SoilComponent } from "../overviewBundle";

/**
 * USDA soil survey terms in plain words. Everything here is a fixed reading of the survey's
 * own classes (drainage class, hydrologic group, texture in the map-unit name, slope); none
 * of it is a measurement of this land.
 */

const TEXTURES = [
  "very fine sandy loam",
  "loamy very fine sand",
  "coarse sandy loam",
  "fine sandy loam",
  "loamy coarse sand",
  "loamy fine sand",
  "silty clay loam",
  "sandy clay loam",
  "very fine sand",
  "coarse sand",
  "fine sand",
  "silty clay",
  "sandy clay",
  "clay loam",
  "silt loam",
  "sandy loam",
  "loamy sand",
  "mucky peat",
  "muck",
  "peat",
  "loam",
  "sand",
  "silt",
  "clay",
];
const MODIFIER =
  "(?:(?:very|extremely)\\s+)?(?:gravelly|cobbly|stony|channery|shaly|rocky|bouldery|flaggy|cherty|mucky)\\s+";
const TEXTURE = new RegExp(`\\b((?:${MODIFIER})?(?:${TEXTURES.join("|")}))\\b`, "i");

/** The map unit's own name, without its slope range: "Hagerstown silt loam". */
export function soilName(mapUnit: string): string {
  return mapUnit.split(",")[0]?.trim() ?? "";
}

/** The texture named in a map unit ("Hagerstown silt loam, 3 to 8 percent slopes" → "silt loam"). */
export function soilTexture(mapUnit: string): string | null {
  const match = TEXTURE.exec(soilName(mapUnit));
  return match?.[1]?.toLowerCase().replace(/\s+/g, " ") ?? null;
}

/** Map units that are not soil at all. */
export function notSoil(mapUnit: string): "water" | "urban" | null {
  const name = soilName(mapUnit).toLowerCase();
  if (/^water\b|^open water/.test(name)) return "water";
  if (/^urban land\b|^pits\b|^dumps\b|^made land\b/.test(name)) return "urban";
  return null;
}

type Drainage = "very-fast" | "fast" | "well" | "fair" | "slow" | "wet" | "waterlogged" | "under";

function drainageOf(drainage: string | null): Drainage | null {
  const value = (drainage ?? "").trim().toLowerCase();
  if (!value) return null;
  if (value.startsWith("subaqueous")) return "under";
  if (value.startsWith("very poorly")) return "waterlogged";
  if (value.startsWith("somewhat poorly")) return "slow";
  if (value.startsWith("poorly")) return "wet";
  if (value.startsWith("moderately well")) return "fair";
  if (value.startsWith("somewhat excessively")) return "fast";
  if (value.startsWith("excessively")) return "very-fast";
  if (value.startsWith("well")) return "well";
  return null;
}

const DRAINAGE: Record<Drainage, { short: string; fact: string }> = {
  "very-fast": { short: "drains very fast", fact: "Drains very fast and dries out quickly" },
  fast: { short: "drains fast", fact: "Drains fast" },
  well: { short: "drains well", fact: "Drains well" },
  fair: { short: "drains fairly well", fact: "Drains fairly well; wet for short spells" },
  slow: { short: "drains slowly", fact: "Drains slowly" },
  wet: { short: "stays wet", fact: "Drains poorly and is often wet" },
  waterlogged: { short: "stays waterlogged", fact: "Waterlogged much of the year" },
  under: { short: "lies under water", fact: "Lies under water" },
};

/** "drains slowly", for sentences. */
export function drainageShort(drainage: string | null): string | null {
  const key = drainageOf(drainage);
  return key ? DRAINAGE[key].short : null;
}

/** "Drains slowly", for a fact. */
export function drainageFact(drainage: string | null): string | null {
  const key = drainageOf(drainage);
  return key ? DRAINAGE[key].fact : null;
}

/** What rain does on this soil, from its hydrologic group (A sandy … D clay). */
export function rainSoak(group: string | null): string | null {
  const value = (group ?? "").trim().toUpperCase();
  if (/^[ABC]\/D$/.test(value)) return "Runs off while waterlogged; soaks in once drained";
  if (value === "A") return "Soaks in fast; little runs off";
  if (value === "B") return "Soaks in fairly well";
  if (value === "C") return "Soaks in slowly; some runs off";
  if (value === "D") return "Mostly runs off; soaks in very slowly";
  return null;
}

/** The hydrologic groups, for the Details guide. */
export const HYDROLOGIC_GROUPS: readonly [string, string][] = [
  ["A", "Sandy or gravelly: water soaks in fast"],
  ["B", "Loamy: water soaks in fairly well"],
  ["C", "Fine-textured: water soaks in slowly"],
  ["D", "Clay or shallow soil: water runs off"],
  ["A/D, B/D, C/D", "Runs off while the water table is high; soaks in once drained"],
];

/** "Gentle, about 5%". */
export function slopeWords(percent: number | null): string | null {
  if (percent === null) return null;
  const name =
    percent < 2
      ? "Nearly flat"
      : percent < 8
        ? "Gentle"
        : percent < 15
          ? "Moderate"
          : percent < 30
            ? "Steep"
            : "Very steep";
  return `${name}, about ${Math.round(percent).toLocaleString()}%`;
}

/**
 * A rule of thumb for what the soil suits, from drainage and slope only. Shown as such in
 * Details: it is not a land capability class or a soil test.
 */
export function soilUse(drainage: string | null, slopePercent: number | null): string | null {
  const key = drainageOf(drainage);
  if (!key) return null;
  if (key === "under") return null;
  if (key === "waterlogged") return "best left as wetland or wet pasture";
  if (key === "wet") return "best for pasture";
  if (key === "slow") return "best for pasture or hay";
  if (slopePercent !== null && slopePercent >= 15) return "best kept in grass or trees";
  if (key === "very-fast" || key === "fast") return "suits drought-tolerant plants";
  if (slopePercent !== null && slopePercent >= 8) return "suits pasture, orchards or vines";
  return "suits most crops";
}

/** The main component of the map unit: the largest share with the data a lens needs. */
export function mainComponent(components: readonly SoilComponent[]): SoilComponent | null {
  return components[0] ?? null;
}

/** The first value among the components (largest first) that the survey filled in. */
export function firstKnown<K extends "drainage" | "hydrologicGroup" | "slopePercent">(
  components: readonly SoilComponent[],
  key: K,
): SoilComponent[K] | null {
  for (const component of components) if (component[key] !== null) return component[key];
  return null;
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * The Soil lens's sentence: "Clay loam near the middle of the land: drains slowly, best for
 * pasture." `where` is the sample point's place on the land ("the middle", "the north").
 */
export function soilHeadline(components: readonly SoilComponent[], where: string): string | null {
  const main = mainComponent(components);
  if (!main) return null;
  const odd = notSoil(main.mapUnit);
  if (odd === "water") return "The sample point is mapped as open water.";
  if (odd === "urban") return "The sample point is mapped as built-over land, not described soil.";
  const place = where === "the middle" ? "near the middle of the land" : `in ${where} of the land`;
  const texture = soilTexture(main.mapUnit);
  const subject = texture
    ? capitalise(texture)
    : capitalise(soilName(main.mapUnit) || main.component || "Mapped soil");
  const drainage = firstKnown(components, "drainage");
  const short = drainageShort(drainage);
  const use = soilUse(drainage, firstKnown(components, "slopePercent"));
  if (!short) return `${subject} ${place}.`;
  return `${subject} ${place}: ${short}${use ? `, ${use}` : ""}.`;
}
