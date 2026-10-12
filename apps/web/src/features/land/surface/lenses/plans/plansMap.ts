import type { Footprint } from "@twin/contracts";
import { footprintAreaM2, polygonsOf } from "@twin/geo";

import type { LandPoint } from "@/state/land";
import type { LandContextFeature, LandContextLayer } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";

import { anchorOf } from "../../geometryAnchor";
import { AVOID_COLOR, stepColor, WORK_AREA_COLOR, type Area, type Step } from "./planModel";
import { shortArea } from "./plansFormat";

/** Map layer ids; each starts with the lens id so the shell can tell whose they are. */
export const PLAN_LAYERS = {
  workArea: "plans-work-area",
  steps: "plans-steps",
  avoid: "plans-avoid",
  drawing: "plans-drawing",
} as const;

const DEG = Math.PI / 180;
const SQRT2 = Math.SQRT2;
const METRES_PER_DEGREE = 111_320;

/**
 * Diagonal lines clipped to an area, so an area to avoid reads as hatched on the map. Each
 * line runs at 45° in a local flat frame; crossings with every ring (holes included) pair up
 * into the segments that lie inside.
 */
export function hatchLines(area: Footprint, maxLines = 14): number[][][] {
  const polygons = polygonsOf(area);
  const points = polygons.flat(2);
  if (!points.length) return [];
  let west = Infinity;
  let east = -Infinity;
  let south = Infinity;
  let north = -Infinity;
  for (const [lon = 0, lat = 0] of points) {
    west = Math.min(west, lon);
    east = Math.max(east, lon);
    south = Math.min(south, lat);
    north = Math.max(north, lat);
  }
  const lon0 = (west + east) / 2;
  const lat0 = (south + north) / 2;
  const k = Math.cos(lat0 * DEG) || 1;
  const toUV = (point: number[]): [number, number] => {
    const x = ((point[0] ?? 0) - lon0) * k;
    const y = (point[1] ?? 0) - lat0;
    return [(x + y) / SQRT2, (y - x) / SQRT2];
  };
  const fromUV = (u: number, v: number): number[] => {
    const x = (u - v) / SQRT2;
    const y = (u + v) / SQRT2;
    return [x / k + lon0, y + lat0];
  };
  const lines: number[][][] = [];
  for (const polygon of polygons) {
    const rings = polygon.map((ring) => ring.map(toUV));
    const outer = rings[0];
    if (!outer?.length) continue;
    let low = Infinity;
    let high = -Infinity;
    for (const [, v] of outer) {
      low = Math.min(low, v);
      high = Math.max(high, v);
    }
    const spanMetres = (high - low) * METRES_PER_DEGREE;
    const count = Math.max(3, Math.min(maxLines, Math.round(spanMetres / 15)));
    for (let i = 1; i <= count; i++) {
      const v = low + ((high - low) * i) / (count + 1);
      const crossings: number[] = [];
      for (const ring of rings)
        for (let j = 0; j < ring.length - 1; j++) {
          const [au, av] = ring[j] ?? [0, 0];
          const [bu, bv] = ring[j + 1] ?? [0, 0];
          if (av > v === bv > v) continue;
          crossings.push(au + ((v - av) / (bv - av)) * (bu - au));
        }
      crossings.sort((a, b) => a - b);
      for (let j = 0; j + 1 < crossings.length; j += 2)
        lines.push([fromUV(crossings[j] ?? 0, v), fromUV(crossings[j + 1] ?? 0, v)]);
    }
  }
  return lines;
}

/** The plan's work area after exclusions: a faint outline in the lens colour. */
export function workAreaLayer(
  area: Area | null | undefined,
  title: string,
): LandContextLayer | null {
  if (!area) return null;
  return {
    id: PLAN_LAYERS.workArea,
    title,
    style: { color: WORK_AREA_COLOR, fill: 0.05, width: 2 },
    features: [{ id: "work-area", label: "Work area", geometry: area }],
  };
}

/** Each step's own area, in its own colour. Steps over the whole area draw nothing extra. */
export function stepsLayer(steps: Step[]): LandContextLayer | null {
  const features: LandContextFeature[] = steps.flatMap((step, index) =>
    step.footprint
      ? [
          {
            id: step.id,
            label: `${index + 1} · ${step.title || "Untitled step"}`,
            geometry: step.footprint,
            color: stepColor(index),
          },
        ]
      : [],
  );
  if (!features.length) return null;
  return {
    id: PLAN_LAYERS.steps,
    title: "Plan steps",
    style: { color: stepColor(0), fill: 0.3, width: 2.5 },
    features,
  };
}

/** Areas to avoid: red, hatched. */
export function avoidLayer(exclusions: Area[] | null | undefined): LandContextLayer | null {
  if (!exclusions?.length) return null;
  const features: LandContextFeature[] = [];
  exclusions.forEach((area, index) => {
    features.push({ id: `avoid-${index}`, label: `Area to avoid ${index + 1}`, geometry: area });
    hatchLines(area).forEach((coordinates, line) =>
      features.push({
        id: `avoid-${index}-hatch-${line}`,
        label: `Area to avoid ${index + 1}`,
        geometry: { type: "LineString", coordinates },
      }),
    );
  });
  return {
    id: PLAN_LAYERS.avoid,
    title: "Areas to avoid",
    style: { color: AVOID_COLOR, fill: 0.16, width: 2.5 },
    features,
  };
}

/** The outline being drawn: its corners and the path between them. */
export function drawingLayer(points: LandPoint[], color: string): LandContextLayer | null {
  if (!points.length) return null;
  const features: LandContextFeature[] = points.map((point, index) => ({
    id: `corner-${index}`,
    label: `Corner ${index + 1}`,
    geometry: { type: "Point", coordinates: [point[0], point[1]] },
  }));
  if (points.length > 1)
    features.push({
      id: "path",
      label: "Outline so far",
      geometry: {
        type: "LineString",
        coordinates: [...points, ...(points.length > 2 ? [points[0]] : [])].map((point) => [
          point?.[0] ?? 0,
          point?.[1] ?? 0,
        ]),
      },
    });
  return {
    id: PLAN_LAYERS.drawing,
    title: "Area being drawn",
    style: { color, fill: 0.2, width: 3 },
    features,
  };
}

function clip(text: string, length = 18): string {
  const value = text.trim() || "Untitled";
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

/**
 * A pin per step that has its own area ("1 · Mow · 4.2 ac"), and one for the steps that use
 * the whole work area. At most twelve.
 */
export function stepPins(
  steps: Step[],
  whole: Footprint | null | undefined,
  onSelect?: (stepId: string) => void,
): MapPin[] {
  const pins: MapPin[] = [];
  const wholeSteps: number[] = [];
  steps.forEach((step, index) => {
    if (!step.footprint) {
      wholeSteps.push(index);
      return;
    }
    if (pins.length >= 11) return;
    pins.push({
      id: `plans-step-${step.id}`,
      position: anchorOf(step.footprint),
      label: `${index + 1} · ${clip(step.title)} · ${shortArea(footprintAreaM2(step.footprint))}`,
      tone: "accent",
      ...(onSelect ? { onSelect: () => onSelect(step.id) } : {}),
    });
  });
  if (wholeSteps.length && whole) {
    const first = wholeSteps[0] ?? 0;
    const label =
      wholeSteps.length === 1
        ? `${first + 1} · ${clip(steps[first]?.title ?? "")} · whole area`
        : wholeSteps.length <= 4
          ? `Steps ${wholeSteps.map((index) => index + 1).join(", ")} · whole area`
          : `${wholeSteps.length} steps · whole area`;
    const firstStep = steps[first];
    pins.push({
      id: "plans-whole-area",
      position: anchorOf(whole),
      label,
      tone: "neutral",
      ...(onSelect && firstStep ? { onSelect: () => onSelect(firstStep.id) } : {}),
    });
  }
  return pins;
}

/** True when every corner of the outline lies inside the land (a quick check before saving). */
export function cornersInside(points: LandPoint[], land: Footprint): boolean {
  return points.every((point) => {
    for (const polygon of polygonsOf(land)) {
      let inside = false;
      for (const ring of polygon)
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
          const [ax = 0, ay = 0] = ring[i] ?? [];
          const [bx = 0, by = 0] = ring[j] ?? [];
          if (
            ay > point[1] !== by > point[1] &&
            point[0] < ((bx - ax) * (point[1] - ay)) / (by - ay) + ax
          )
            inside = !inside;
        }
      if (inside) return true;
    }
    return false;
  });
}

function crosses(a: LandPoint, b: LandPoint, c: LandPoint, d: LandPoint): boolean {
  const orient = (p: LandPoint, q: LandPoint, r: LandPoint) =>
    Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  return orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0;
}

/** A closed polygon from drawn corners, or why it cannot be one. */
export function outlineFromCorners(points: LandPoint[]): { area: Area } | { error: string } {
  const first = points[0];
  if (!first || points.length < 3) return { error: "Pick at least three corners." };
  const ring = [...points, first];
  for (let i = 0; i < ring.length - 1; i++)
    for (let j = i + 2; j < ring.length - 1; j++) {
      if (i === 0 && j === ring.length - 2) continue;
      const a = ring[i];
      const b = ring[i + 1];
      const c = ring[j];
      const d = ring[j + 1];
      if (a && b && c && d && crosses(a, b, c, d))
        return { error: "The outline crosses itself. Undo a corner and try again." };
    }
  return {
    area: { type: "Polygon", coordinates: [ring.map((point) => [point[0], point[1]])] },
  };
}
