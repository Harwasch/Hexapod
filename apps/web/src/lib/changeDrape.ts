/**
 * What the change drape says and draws (ADR 0018; features/change, cesium/changeDrape.ts),
 * from a site's change map between two dates (`GET /sites/{id}/change`, `change-v1`). Pure and
 * Cesium-free: the legend, the Compare › Dates offer and the drawing all read it.
 *
 * The map is derived from the two dates' height surfaces, never observed and never from splat
 * pixels, and uncalibrated: the drape says so, with the least change it can show between those
 * dates, and names a date whose placement is not survey-grade, so a phone video never reads as
 * a survey.
 */
import type { Position } from "geojson";

import type { ChangeArea, ChangeMap, Epoch, EpochSurface } from "@twin/contracts";
import { polygonsOf } from "@twin/geo";

import { alignmentLabel, formatEpochDate } from "@/lib/epochs";

/** The classes a change map outlines; unchanged cells are counted, never outlined. */
export type ChangeClass = ChangeArea["class"];

export const CHANGE_CLASSES: readonly ChangeClass[] = ["raised", "lowered", "insufficient"];

/** Each class's colour, on the ground and in the legend: warm up, cool down, grey unknown. */
export const CHANGE_COLORS: Record<ChangeClass, string> = {
  raised: "#f08c3a",
  lowered: "#3a9cf0",
  insufficient: "#a3a7ad",
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

/** `±X m`, X to two significant figures. */
function plusMinus(m: number): string {
  return `±${String(Number(m.toPrecision(2)))} m`;
}

/** The drape's one line on what it is, and the least change it can show. */
export function changeLabel(map: Pick<ChangeMap, "minDetectableM">): string {
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

/**
 * The dates the map is only as good as: each of the two whose placement is not survey-grade
 * (aligned visually, not aligned, or not yet), in the words the date control uses. The date
 * that made the site frame, and a survey-tier one, need no note; a phone date never reads as
 * survey-grade, whatever its record claims (alignmentLabel, D4).
 */
export function placementNotes(
  epochs: readonly Epoch[],
  from: Epoch,
  to: Epoch,
  anchor: Epoch | null,
): string[] {
  return [from, to].flatMap((epoch) => {
    const label = alignmentLabel(epoch, anchor, epochs);
    if (label.short === "site frame" || label.short === "survey tier") return [];
    return [`${formatEpochDate(epoch.observedAt)}: ${label.long} — not survey-grade`];
  });
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
export function changeBuckets(map: Pick<ChangeMap, "areas">): ChangeBucket[] {
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
export function legendRows(map: Pick<ChangeMap, "counts" | "lowConfidence">): LegendRow[] {
  const low = map.lowConfidence as Partial<Record<ChangeClass | "unchanged", number>>;
  return [...CHANGE_CLASSES, "unchanged" as const].map((key) => ({
    key,
    label: CHANGE_LABELS[key],
    cells: map.counts[key],
    lowConfidence: low[key] ?? 0,
    drawn: key !== "unchanged",
  }));
}
