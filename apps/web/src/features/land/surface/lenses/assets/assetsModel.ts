import type { components, LandCandidate, LandMapGeometry } from "@twin/contracts";
import { footprintAreaM2, formatLength, haversineDistance, type UnitSystem } from "@twin/geo";

import { areaLabel } from "../../landFlow";
import type { Vertex } from "./assetsGeometry";

export type Asset = components["schemas"]["LandFeatureRead"];
export type AssetCategory = Asset["category"];
export type AssetStatus = NonNullable<Asset["status"]>;
export type Inspection = components["schemas"]["FeatureInspectionRead"];
export type Condition = Inspection["condition"];
export type ShapeKind = "Point" | "LineString" | "Polygon";

/** How each category reads and shows on the map. Gold is kept free for the selected asset. */
export const CATEGORIES: Record<AssetCategory, { label: string; one: string; color: string }> = {
  building: { label: "Buildings", one: "Building", color: "#e0956e" },
  water: { label: "Water", one: "Water", color: "#5aaef0" },
  power: { label: "Power", one: "Power", color: "#b99af5" },
  transport: { label: "Roads and access", one: "Access", color: "#c9c2b2" },
  equipment: { label: "Equipment", one: "Equipment", color: "#ef8fb4" },
  vegetation: { label: "Trees and plants", one: "Plant", color: "#7ccf7c" },
  other: { label: "Other", one: "Other", color: "#7fd8c0" },
};

export const CATEGORY_ORDER: readonly AssetCategory[] = [
  "building",
  "water",
  "power",
  "transport",
  "equipment",
  "vegetation",
  "other",
];

/** The newest inspection's verdict, and "none" for an asset nobody has inspected. */
export const CONDITIONS: Record<Condition | "none", { label: string; color: string }> = {
  good: { label: "Good", color: "#7ccf7c" },
  fair: { label: "Fair", color: "#d9e26a" },
  poor: { label: "Poor", color: "#f0964a" },
  critical: { label: "Critical", color: "#e8605a" },
  unknown: { label: "Unknown", color: "#a3a9b0" },
  none: { label: "Not inspected", color: "#6f7780" },
};

export const CONDITION_ORDER: readonly Condition[] = [
  "good",
  "fair",
  "poor",
  "critical",
  "unknown",
];

export const STATUSES: Record<AssetStatus, string> = {
  candidate: "Not yet confirmed",
  confirmed: "Confirmed",
  retired: "Retired",
};

export function needsAttention(condition: Condition | null | undefined): boolean {
  return condition === "poor" || condition === "critical";
}

/** A common thing to map: what it is called, where it belongs, and how it is drawn. */
export interface AssetPreset {
  id: string;
  label: string;
  /** The name a new asset starts with; empty asks the person for one. */
  name: string;
  category: AssetCategory;
  kind: ShapeKind;
}

export const PRESETS: readonly AssetPreset[] = [
  { id: "building", label: "Building", name: "Building", category: "building", kind: "Polygon" },
  { id: "fence", label: "Fence", name: "Fence", category: "other", kind: "LineString" },
  { id: "gate", label: "Gate", name: "Gate", category: "transport", kind: "Point" },
  { id: "track", label: "Track", name: "Track", category: "transport", kind: "LineString" },
  { id: "well", label: "Well", name: "Well", category: "water", kind: "Point" },
  { id: "pond", label: "Pond", name: "Pond", category: "water", kind: "Polygon" },
  { id: "pipe", label: "Pipe", name: "Pipe", category: "water", kind: "LineString" },
  { id: "culvert", label: "Culvert", name: "Culvert", category: "water", kind: "Point" },
  { id: "power", label: "Power line", name: "Power line", category: "power", kind: "LineString" },
  { id: "solar", label: "Solar panels", name: "Solar panels", category: "power", kind: "Polygon" },
  { id: "tree", label: "Tree", name: "Tree", category: "vegetation", kind: "Point" },
  { id: "equipment", label: "Equipment", name: "", category: "equipment", kind: "Point" },
];

/** Anything else, by shape alone. */
export const SHAPES: readonly AssetPreset[] = [
  { id: "point", label: "A point", name: "", category: "other", kind: "Point" },
  { id: "line", label: "A line", name: "", category: "other", kind: "LineString" },
  { id: "area", label: "An area", name: "", category: "other", kind: "Polygon" },
];

export const SHAPE_NAMES: Record<LandMapGeometry["type"], string> = {
  Point: "Point",
  LineString: "Line",
  Polygon: "Area",
  MultiPolygon: "Areas",
};

/** "Well", or "Well 3" when wells already carry the plain name. */
export function nextName(base: string, names: readonly string[]): string {
  if (!base) return "";
  const taken = new Set(names.map((name) => name.trim().toLowerCase()));
  if (!taken.has(base.toLowerCase())) return base;
  for (let n = 2; ; n++) if (!taken.has(`${base} ${n}`.toLowerCase())) return `${base} ${n}`;
}

export function lineLengthM(points: readonly (readonly number[])[]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1] ?? [];
    const b = points[i] ?? [];
    total += haversineDistance(
      { longitude: a[0] ?? 0, latitude: a[1] ?? 0 },
      { longitude: b[0] ?? 0, latitude: b[1] ?? 0 },
    );
  }
  return total;
}

/** Length for a line, area for an area, nothing for a point: "420 m", "0.4 acres". */
export function sizeLabel(geometry: LandMapGeometry, units: UnitSystem): string | null {
  if (geometry.type === "Point") return null;
  if (geometry.type === "LineString") return formatLength(lineLengthM(geometry.coordinates), units);
  return areaLabel(footprintAreaM2(geometry));
}

/** Where an asset sits against the land: inside it, or how far outside. */
export function whereLabel(asset: Pick<Asset, "intersectsLand" | "distanceM">, units: UnitSystem) {
  return asset.intersectsLand
    ? "On your land"
    : `${formatLength(asset.distanceM, units)} outside your land`;
}

/** "today", "yesterday", "5 days ago", then a date. */
export function whenLabel(iso: string, now = new Date()): string {
  const then = new Date(iso);
  if (!Number.isFinite(then.getTime())) return "";
  const day = (value: Date) => Date.UTC(value.getFullYear(), value.getMonth(), value.getDate());
  const days = Math.round((day(now) - day(then)) / 86_400_000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days > 1 && days < 14) return `${days} days ago`;
  return then.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(then.getFullYear() !== now.getFullYear() ? { year: "numeric" } : {}),
  });
}

export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count.toLocaleString()} ${count === 1 ? one : many}`;
}

export interface AssetSummary {
  /** Assets still in use (retired ones are history). */
  count: number;
  /** True when the list stopped at its page size, so the count is a floor. */
  more: boolean;
  categories: { category: AssetCategory; count: number }[];
  /** Newest inspection poor or critical, worst first. */
  attention: { asset: Asset; inspection: Inspection }[];
  /** True once every asset's newest inspection is known. */
  conditionsKnown: boolean;
  inspected: number;
  last: { asset: Asset; inspection: Inspection } | null;
  unconfirmed: number;
}

/**
 * The few numbers the card leads with, from the asset list and each asset's newest inspection
 * (`undefined` while it loads, `null` when there is none).
 */
export function summarizeAssets(
  assets: readonly Asset[],
  latest: Readonly<Record<string, Inspection | null | undefined>>,
  more = false,
): AssetSummary {
  const active = assets.filter((asset) => asset.status !== "retired");
  const counts = new Map<AssetCategory, number>();
  for (const asset of active) counts.set(asset.category, (counts.get(asset.category) ?? 0) + 1);
  const categories = [...counts]
    .map(([category, count]) => ({ category, count }))
    .sort(
      (a, b) =>
        b.count - a.count ||
        CATEGORY_ORDER.indexOf(a.category) - CATEGORY_ORDER.indexOf(b.category),
    );
  const attention: AssetSummary["attention"] = [];
  let last: AssetSummary["last"] = null;
  let inspected = 0;
  let conditionsKnown = true;
  for (const asset of active) {
    const inspection = latest[asset.id];
    if (inspection === undefined) {
      conditionsKnown = false;
      continue;
    }
    if (!inspection) continue;
    inspected++;
    if (needsAttention(inspection.condition)) attention.push({ asset, inspection });
    if (!last || inspection.observedAt > last.inspection.observedAt) last = { asset, inspection };
  }
  attention.sort(
    (a, b) =>
      Number(b.inspection.condition === "critical") -
        Number(a.inspection.condition === "critical") || a.asset.name.localeCompare(b.asset.name),
  );
  return {
    count: active.length,
    more,
    categories,
    attention,
    conditionsKnown,
    inspected,
    last,
    unconfirmed: active.filter((asset) => (asset.status ?? "candidate") === "candidate").length,
  };
}

/** One plain sentence: how many things, and whether any need a look. */
export function assetsHeadline(summary: AssetSummary): string {
  const total = `${summary.count.toLocaleString()}${summary.more ? "+" : ""} ${
    summary.count === 1 && !summary.more ? "asset" : "assets"
  }`;
  const attention = summary.attention.length;
  if (attention) return `${total}: ${attention} ${attention === 1 ? "needs" : "need"} attention.`;
  if (!summary.conditionsKnown) return `${total}.`;
  if (!summary.inspected) return `${total}, none inspected yet.`;
  return `${total}, none needing attention.`;
}

/** A pin label that fits: the name cut at a word near 22 characters. */
export function shortLabel(name: string, max = 22): string {
  const trimmed = name.trim();
  if (trimmed.length <= max) return trimmed;
  const cut = trimmed.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${(space > max / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

function distinct(points: Vertex[]): number {
  return new Set(points.map((point) => point.join(","))).size;
}

/** Drawn points as a shape the server can check, or null while there are too few. */
export function shapeOf(kind: ShapeKind, points: Vertex[]): LandMapGeometry | null {
  const first = points[0];
  if (!first) return null;
  if (kind === "Point") return { type: "Point", coordinates: first };
  if (kind === "LineString")
    return distinct(points) >= 2 ? { type: "LineString", coordinates: points } : null;
  return distinct(points) >= 3 ? { type: "Polygon", coordinates: [[...points, first]] } : null;
}

function categoryOf(candidate: LandCandidate): AssetCategory {
  const properties = candidate.properties ?? {};
  if (properties.power) return "power";
  if (properties.building) return "building";
  if (properties.waterway) return "water";
  if (properties.highway || properties.railway) return "transport";
  return candidate.geometry.type === "Point" ? "equipment" : "other";
}

/** A record found in open map data as an unconfirmed asset, keeping its tags as attributes. */
export function draftOfCandidate(
  candidate: LandCandidate,
): components["schemas"]["LandFeatureCreate"] {
  return {
    name: candidate.label,
    category: categoryOf(candidate),
    geometry: candidate.geometry,
    source: candidate.source,
    status: "candidate",
    attributes: Object.fromEntries(
      Object.entries(candidate.properties ?? {}).map(([key, value]) => [key, String(value)]),
    ),
  };
}
