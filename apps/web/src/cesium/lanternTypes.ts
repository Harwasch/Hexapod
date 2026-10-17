import type { Footprint } from "@twin/contracts";

import type { LanternAvailability, LanternMode } from "@/state/lantern";

/** What a dated source shows: an aerial photograph, a satellite scene, an old map. */
export type LanternSourceKind = "aerial-photo" | "satellite" | "historical-map" | "aligned-map";

/** One record a source is made of, with its own date and stated accuracy. */
export interface LanternPart {
  footprint: Footprint;
  /** "15 June 2016", "Map dated 1959". */
  dateLabel: string;
  /** As LanternSource.accuracyLabel, for this record. */
  accuracyLabel: string;
  recordUrl: string;
}

/**
 * A dated source the Time Lantern can show (features/lantern/useDatedSources.ts maps the API's
 * catalog to these). Its labels are the rim's words: what it is, when, under what licence, and
 * how far off it may be -- never checked against today's 3D world, and saying so.
 */
export interface LanternSource {
  id: string;
  kind: LanternSourceKind;
  title: string;
  dateLabel: string;
  dateMeaning: string;
  /** ISO date; the rail is ordered by it, newest first. */
  sortDate: string;
  resolutionM: number;
  /** A map's printed scale (24000 for 1:24,000), else null. */
  mapScale: number | null;
  license: string;
  licenseUrl: string | null;
  attribution: string;
  /** "stated ±6 m by its record · not checked against today's 3D world". */
  accuracyLabel: string;
  /** West, south, east, north (degrees). */
  bounds: [number, number, number, number];
  parts: LanternPart[];
  tiles: {
    scheme: "web-mercator" | "geographic";
    /** Absolute: API-relative templates are prefixed before they get here. */
    urlTemplate: string;
    maximumLevel: number;
    headers?: Record<string, string>;
  };
  warnings: string[];
}

/** An area in degrees. */
export interface LanternArea {
  west: number;
  south: number;
  east: number;
  north: number;
}

/**
 * The lantern as the HUD needs it, published on every change and every frame the lens moves
 * (SceneEvents['lantern']): LanternBridge mirrors mode and index into the store, the rim
 * (LanternRim) follows the lens without a React render.
 */
export interface LanternFrame {
  mode: LanternMode;
  /** The source shown, in the order the bridge gave them. */
  index: number;
  availability: LanternAvailability;
  /** The lens on screen, CSS px from the canvas's top left; null while it is not drawn. */
  lens: { x: number; y: number; r: number } | null;
  /** The ground under the lens's centre; null over the sky. */
  ground: { longitude: number; latitude: number } | null;
  source: Pick<
    LanternSource,
    "id" | "kind" | "title" | "dateLabel" | "license" | "accuracyLabel" | "sortDate"
  > | null;
  /** The record under the lens's centre: its own date and accuracy, when it has one there. */
  part: { dateLabel: string; accuracyLabel: string } | null;
  /** Whether the shown source has a record under the lens's centre. */
  covered: boolean;
}
