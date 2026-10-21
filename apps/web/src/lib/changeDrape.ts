/**
 * What the change drape says and draws (ADR 0018; features/change, cesium/changeDrape.ts),
 * from a site's change map between two dates (`GET /sites/{id}/change`, `change-v1`). Pure and
 * Cesium-free: the legend, the Compare › Dates offer and the drawing all read it.
 *
 * The map is derived from the two dates' height surfaces, never observed and never from splat
 * pixels, and uncalibrated: the drape says so, with the least change each layer can show
 * between those dates, and says when a date's placement is not survey-grade, so a phone video
 * never reads as a survey. It comes in two layers (ADR 0018 § API 2): the ground, the primary
 * product, and vegetation (the top surface), at its own higher threshold and never ground
 * change; each is keyed and drawn on its own.
 */
import type { Position } from "geojson";

import type { ChangeArea, ChangeMap, Epoch, EpochSurface } from "@twin/contracts";
import { polygonsOf } from "@twin/geo";

import { formatEpochDate } from "@/lib/epochs";

/** One layer of a change map: the ground, or vegetation and objects. */
export type ChangeLayer = ChangeMap["ground"];
export type ChangeLayerName = ChangeLayer["layer"];

/** How the vegetation layer is named wherever it is shown, apart from the ground. */
export const VEGETATION_TITLE =
  "Vegetation: canopy height change, higher threshold, wind and growth not ground change";

/** The classes a change map outlines; unchanged cells are counted, never outlined. */
export type ChangeClass = ChangeArea["class"];

export const CHANGE_CLASSES: readonly ChangeClass[] = ["raised", "lowered", "insufficient"];

/**
 * Each layer's class colours, on the ground and in its legend. The ground: warm up, cool down,
 * grey unknown. Vegetation in hues of its own (yellow up, violet down; not green, which a
 * scan's grass already is), so the two are never read as one where both are drawn.
 */
export const CHANGE_COLORS: Record<ChangeLayerName, Record<ChangeClass, string>> = {
  ground: { raised: "#f08c3a", lowered: "#3a9cf0", insufficient: "#a3a7ad" },
  vegetation: { raised: "#f2c94c", lowered: "#b37be0", insufficient: "#a3a7ad" },
};

export const CHANGE_LABELS: Record<ChangeClass | "unchanged", string> = {
  raised: "Raised",
  lowered: "Lowered",
  insufficient: "Not comparable",
  unchanged: "Unchanged",
};

/** The words for a 401 or 403: the map is behind the write token (ADR 0018 § API 8). */
export const NEEDS_TOKEN =
  "The change map needs the pilot's write token: enter it in Settings › API access and it loads.";

/** An epoch's height surface, or null: not measured (a splat upload, or an older run). */
export function surfaceOf(epoch: Epoch | undefined): EpochSurface | null {
  // `?? null`: a record from before surfaces were listed carries no key at all.
  return epoch?.surface ?? null;
}

function time(iso: string): number {
  const at = Date.parse(iso);
  return Number.isFinite(at) ? at : Number.NEGATIVE_INFINITY;
}

export type ChangeOffer =
  { offered: true; from: string; to: string } | { offered: false; why: string };

/**
 * Whether the change map is offered for two chosen dates: both listed, not one twice, and both
 * with a height surface measured. From the older to the newer, whichever side each is on, so
 * "raised" always means higher later.
 */
export function changeOffer(epochs: readonly Epoch[], left: string, right: string): ChangeOffer {
  if (left === right) return { offered: false, why: "Not offered: choose two different dates" };
  const a = epochs.find((e) => e.id === left);
  const b = epochs.find((e) => e.id === right);
  if (!a || !b) return { offered: false, why: "Not offered: a date chosen is not listed" };
  const later = time(a.observedAt) - time(b.observedAt) || a.ordinal - b.ordinal;
  const [from, to] = later > 0 ? [b, a] : [a, b];
  const missing = [from, to].filter((e) => !surfaceOf(e));
  const [first, second] = missing.map((e) => formatEpochDate(e.observedAt));
  if (second)
    return {
      offered: false,
      why: `Not offered: neither ${first} nor ${second} has a height surface measured`,
    };
  if (first) return { offered: false, why: `Not offered: ${first} has no height surface measured` };
  return { offered: true, from: from.id, to: to.id };
}

/** `±X m`, X to two significant figures, rounded up: the floor is never understated. */
function plusMinus(m: number): string {
  if (!(m > 0)) return `±${String(m)} m`;
  const step = 10 ** (Math.floor(Math.log10(m)) - 1);
  // Through toPrecision, so 0.13 is not printed 0.13000000000000003.
  return `±${String(Number((Math.ceil(m / step - 1e-9) * step).toPrecision(2)))} m`;
}

/** A layer's one line on what it is, and the least change it can show. */
export function changeLabel(map: Pick<ChangeLayer, "minDetectableM">): string {
  const floor =
    map.minDetectableM === null
      ? "no cell could be compared between these dates"
      : `changes under ${plusMinus(map.minDetectableM)} cannot be seen between these dates`;
  return `Derived from height surfaces — not observed · ${floor}`;
}

/**
 * What is not measured behind the map (D7): the 3σ rule, and the surfaces' ground window when
 * the dates say what it was.
 */
export function calibrationNote(map: Pick<ChangeMap, "thresholdSigma">, epochs: readonly Epoch[]) {
  const windows = [
    ...new Set(
      epochs
        .map((e) => surfaceOf(e)?.groundWindowM)
        .filter((m): m is number => typeof m === "number"),
    ),
  ];
  const rule = `the ${String(map.thresholdSigma)}σ rule`;
  return windows.length
    ? `Uncalibrated: ${rule} and the ${windows.map((m) => `${String(m)} m`).join(" / ")} ground window are first principles until a site scanned twice measures them`
    : `Uncalibrated: ${rule} is first principles until a site scanned twice measures it`;
}

/** The vegetation layer's uncalibrated parts: the rule, and the wind sway allowed per date. */
export function vegetationNote(
  map: Pick<ChangeMap, "thresholdSigma">,
  layer: Pick<ChangeLayer, "motionM">,
): string {
  return `Uncalibrated: the ${String(map.thresholdSigma)}σ rule and ${plusMinus(layer.motionM)} of wind sway per date are first principles until a site scanned twice measures them`;
}

/**
 * When either date's placement is coarse (any phone, a drone without RTK evidence: ADR 0018
 * § API 4), the map says it is not survey-grade, with each date's verdict and the placement
 * uncertainty it was compared with; null when both are survey-grade.
 */
export function verdictNote(
  uncertainty: ChangeMap["uncertainty"],
  from: Pick<Epoch, "observedAt">,
  to: Pick<Epoch, "observedAt">,
): string | null {
  if (uncertainty.fromVerdict === "survey" && uncertainty.toVerdict === "survey") return null;
  const side = (epoch: Pick<Epoch, "observedAt">, verdict: string, m: number) =>
    `${formatEpochDate(epoch.observedAt)} ${verdict} (${plusMinus(m)})`;
  return `Not survey-grade: ${side(from, uncertainty.fromVerdict, uncertainty.fromM)} · ${side(to, uncertainty.toVerdict, uncertainty.toM)}`;
}

const REFUSALS: Record<string, string> = {
  "epoch-not-published": "One of these dates is no longer published (rejected or superseded).",
  "no-site-frame": "This site has no site frame, so its dates cannot be compared.",
  "surface-not-measured": "One of these dates has no height surface measured.",
  "confidence-not-measured":
    "One of these dates has no confidence map, so how far to trust its heights is unknown.",
  "frames-differ":
    "The two height surfaces are above different origins, so they cannot be compared.",
  "surface-unreadable": "A height surface could not be read.",
  "confidence-unreadable": "A confidence map could not be read.",
  "epoch-unaligned":
    "One of these dates is not aligned to the site (no alignment, or placed by its own GPS), so no change is claimed against it.",
  "not-metric":
    "One of these dates' heights are not in metres (its scale was never resolved), so they cannot be compared.",
  "scale-corrected":
    "One of these dates is drawn at a corrected scale its heights were not measured at, so they cannot be compared.",
};

/** A refused or failed request, in words (ADR 0018 § API 7). Status 0: no answer at all. */
export function changeError(status: number, code?: string | null): string {
  if (status === 401 || status === 403) return NEEDS_TOKEN;
  if (status === 409 && code && code in REFUSALS) return REFUSALS[code] ?? "";
  if (status === 404) return "This site, or one of these dates, was not found.";
  if (status === 422) return "The change map needs two different dates.";
  if (status === 429) return "The change map was asked for too often: try again in a minute.";
  if (status === 0) return "The change map could not be loaded: no answer from the API.";
  return `The change map could not be loaded (${String(status)}${code ? ` ${code}` : ""}).`;
}

/** One draw: the areas of one class at one confidence, every polygon of them. */
export interface ChangeBucket {
  cls: ChangeClass;
  lowConfidence: boolean;
  areas: number;
  cells: number;
  /** Polygons, each its rings (outer first), each ring of [lon, lat]. */
  polygons: Position[][][];
}

/**
 * The map's areas grouped for drawing: per class (raised, lowered, not comparable) and
 * confidence (confident first), in that order; an empty group is not drawn, and neither is
 * any class the drape does not know (unchanged is never outlined).
 */
export function changeBuckets(map: Pick<ChangeLayer, "areas">): ChangeBucket[] {
  const buckets: ChangeBucket[] = [];
  for (const cls of CHANGE_CLASSES) {
    for (const lowConfidence of [false, true]) {
      const areas = map.areas.filter((a) => a.class === cls && a.lowConfidence === lowConfidence);
      if (!areas.length) continue;
      buckets.push({
        cls,
        lowConfidence,
        areas: areas.length,
        cells: areas.reduce((sum, a) => sum + a.cells, 0),
        polygons: areas.flatMap((a) => polygonsOf(a.footprint as Parameters<typeof polygonsOf>[0])),
      });
    }
  }
  return buckets;
}

export interface LegendRow {
  key: ChangeClass | "unchanged";
  label: string;
  /** Cells of the class at the compared level. */
  cells: number;
  /** Of them, the cells either date's confidence does not vouch for. */
  lowConfidence: number;
  /** Outlined on the ground; unchanged cells are counted only. */
  drawn: boolean;
}

/** The legend's rows: each class's cells and low-confidence cells, unchanged last. */
export function legendRows(map: Pick<ChangeLayer, "counts" | "lowConfidence">): LegendRow[] {
  const low = map.lowConfidence as Partial<Record<ChangeClass | "unchanged", number>>;
  return [...CHANGE_CLASSES, "unchanged" as const].map((key) => ({
    key,
    label: CHANGE_LABELS[key],
    cells: map.counts[key],
    lowConfidence: low[key] ?? 0,
    drawn: key !== "unchanged",
  }));
}
