/**
 * Framing ground in a perspective view: where to put the camera so a set of ground points
 * lands inside the part of the screen the HUD leaves open.
 *
 * Pure geometry (no Cesium), so a unit test can prove a framing holds for a viewport without a
 * renderer, and the same numbers serve a static bookmark (the demo ranch's arrival, written into
 * demo_ranch.json) and a live fit (`MissionManager.flyToProject`, the site pin). The model is
 * Cesium's: a perspective frustum whose field of view applies to the larger screen dimension,
 * a camera `range` back from its target along heading and pitch. The ground is taken as flat at
 * the target's height; hills on a ranch are tens of metres against a view kilometres deep, which
 * `margin` covers.
 */

import { DEG_TO_RAD, destination, EARTH_RADIUS_M, outerRings } from "@twin/geo";

import type { LonLat, Project } from "./types";

/** Fractions of the viewport the HUD covers on each side. */
export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** A window to frame into: its size in CSS pixels and its field of view (`CameraController.viewport`). */
export interface Viewport {
  width: number;
  height: number;
  fovDeg: number;
}

export interface FramingOptions {
  headingDeg: number;
  pitchDeg: number;
  /** Viewport width over height. */
  aspect: number;
  /** Cesium's `PerspectiveFrustum.fov`, degrees: the larger dimension's field of view. */
  fovDeg?: number;
  insets?: Insets;
  /** The fitted range is multiplied by this, for a little air around the outermost point. */
  margin?: number;
}

export interface Framing {
  target: LonLat;
  headingDeg: number;
  pitchDeg: number;
  rangeM: number;
}

/** A camera pose in degrees and metres, as `CameraController.flyTo` takes one. */
export interface EyePose {
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
}

/**
 * The desktop HUD: project card and search across the top, the tool rail on the left, the
 * status line along the bottom, and the mode tabs on the right.
 */
export const HUD_INSETS: Insets = { top: 0.12, right: 0.06, bottom: 0.1, left: 0.1 };
/**
 * A phone's (measured at 390×844): the project card across the top; the tab bar, the status
 * line and the map credits stacked along the bottom; the tool rail folds into "More".
 */
export const PHONE_HUD_INSETS: Insets = { top: 0.1, right: 0.05, bottom: 0.18, left: 0.05 };

/** The insets for a viewport: the phone's at 640 px wide and below (`mission.css`'s breakpoint). */
export function hudInsets(widthPx: number): Insets {
  return widthPx <= 640 ? PHONE_HUD_INSETS : HUD_INSETS;
}

/** The framing options for a live window, at a heading and pitch. */
export function viewportOptions(
  viewport: Viewport,
  headingDeg: number,
  pitchDeg: number,
): FramingOptions {
  return {
    headingDeg,
    pitchDeg,
    aspect: viewport.width / viewport.height,
    fovDeg: viewport.fovDeg,
    insets: hudInsets(viewport.width),
  };
}

const DEFAULT_FOV_DEG = 60;
const DEFAULT_MARGIN = 1.08;
const MIN_RANGE_M = 100;
const MAX_RANGE_M = 50_000;
const RECENTRE_ITERATIONS = 3;

type Vec = [number, number, number];

const dot = (a: Vec, b: Vec): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec, b: Vec): Vec => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

interface Frame {
  origin: LonLat;
  kx: number;
  ky: number;
}

/** Local east-north metres around `origin` (equirectangular; good to a few km). */
function localFrame(origin: LonLat): Frame {
  const ky = DEG_TO_RAD * EARTH_RADIUS_M;
  return { origin, kx: ky * Math.cos(origin.latitude * DEG_TO_RAD), ky };
}

function toLocal(frame: Frame, p: LonLat): Vec {
  return [
    (p.longitude - frame.origin.longitude) * frame.kx,
    (p.latitude - frame.origin.latitude) * frame.ky,
    0,
  ];
}

function toLonLat(frame: Frame, v: Vec): LonLat {
  return {
    longitude: frame.origin.longitude + v[0] / frame.kx,
    latitude: frame.origin.latitude + v[1] / frame.ky,
  };
}

interface Basis {
  forward: Vec;
  right: Vec;
  up: Vec;
  tanHalfH: number;
  tanHalfV: number;
}

function basis(o: FramingOptions): Basis {
  const h = o.headingDeg * DEG_TO_RAD;
  const p = o.pitchDeg * DEG_TO_RAD;
  const forward: Vec = [Math.sin(h) * Math.cos(p), Math.cos(h) * Math.cos(p), Math.sin(p)];
  const right: Vec = [Math.cos(h), -Math.sin(h), 0];
  const up = cross(right, forward);
  // Cesium's PerspectiveFrustum: `fov` spans the larger of the two dimensions.
  const tanHalf = Math.tan(((o.fovDeg ?? DEFAULT_FOV_DEG) * DEG_TO_RAD) / 2);
  const aspect = o.aspect > 0 && Number.isFinite(o.aspect) ? o.aspect : 1;
  return aspect >= 1
    ? { forward, right, up, tanHalfH: tanHalf, tanHalfV: tanHalf / aspect }
    : { forward, right, up, tanHalfH: tanHalf * aspect, tanHalfV: tanHalf };
}

interface Ndc {
  x: number;
  y: number;
  depth: number;
}

/** Normalised device coordinates (−1…1, y up) of local points seen from `range` back from `target`. */
function project(points: Vec[], target: Vec, range: number, b: Basis): Ndc[] {
  const eye: Vec = [
    target[0] - range * b.forward[0],
    target[1] - range * b.forward[1],
    target[2] - range * b.forward[2],
  ];
  return points.map((point) => {
    const d: Vec = [point[0] - eye[0], point[1] - eye[1], point[2] - eye[2]];
    const depth = dot(d, b.forward);
    if (depth <= 0) return { x: Infinity, y: Infinity, depth };
    return {
      x: dot(d, b.right) / (depth * b.tanHalfH),
      y: dot(d, b.up) / (depth * b.tanHalfV),
      depth,
    };
  });
}

interface Box {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function insetBox(insets: Insets): Box {
  return {
    left: -1 + 2 * insets.left,
    right: 1 - 2 * insets.right,
    top: 1 - 2 * insets.top,
    bottom: -1 + 2 * insets.bottom,
  };
}

/** How far the worst point is outside the box, in NDC; zero or less is inside. */
function overshoot(ndc: Ndc[], box: Box): { excess: number; index: number } {
  let excess = -Infinity;
  let index = -1;
  ndc.forEach((p, i) => {
    const e = Math.max(box.left - p.x, p.x - box.right, box.bottom - p.y, p.y - box.top);
    if (e > excess) {
      excess = e;
      index = i;
    }
  });
  return { excess, index };
}

function fitRange(points: Vec[], target: Vec, b: Basis, box: Box): number {
  let lo = MIN_RANGE_M;
  let hi = MAX_RANGE_M;
  if (overshoot(project(points, target, hi, b), box).excess > 0) return hi;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (overshoot(project(points, target, mid, b), box).excess > 0) lo = mid;
    else hi = mid;
  }
  return hi;
}

function mean(points: LonLat[]): LonLat {
  const sum = points.reduce(
    (acc, p) => ({ longitude: acc.longitude + p.longitude, latitude: acc.latitude + p.latitude }),
    { longitude: 0, latitude: 0 },
  );
  return { longitude: sum.longitude / points.length, latitude: sum.latitude / points.length };
}

/**
 * The camera that frames `points` on the ground inside the HUD-safe part of a viewport: the
 * closest range at which every point is inside the inset box, with the points' screen extent
 * centred in it, times `margin`.
 */
export function frameGround(points: LonLat[], o: FramingOptions): Framing {
  if (points.length === 0) throw new Error("frameGround needs at least one point");
  const frame = localFrame(mean(points));
  const local = points.map((p) => toLocal(frame, p));
  const b = basis(o);
  const box = insetBox(o.insets ?? HUD_INSETS);
  const boxX = (box.left + box.right) / 2;
  const boxY = (box.top + box.bottom) / 2;
  // Moving the target along the ground's forward direction moves the view's y by sin(-pitch).
  const along: Vec = [Math.sin(o.headingDeg * DEG_TO_RAD), Math.cos(o.headingDeg * DEG_TO_RAD), 0];
  const lift = Math.max(0.2, Math.sin(-o.pitchDeg * DEG_TO_RAD));
  let target: Vec = [0, 0, 0];
  let range = fitRange(local, target, b, box);
  for (let i = 0; i < RECENTRE_ITERATIONS; i++) {
    const ndc = project(local, target, range, b);
    const xs = ndc.map((p) => p.x);
    const ys = ndc.map((p) => p.y);
    const dx = (Math.min(...xs) + Math.max(...xs)) / 2 - boxX;
    const dy = (Math.min(...ys) + Math.max(...ys)) / 2 - boxY;
    const sideways = dx * range * b.tanHalfH;
    const forward = (dy * range * b.tanHalfV) / lift;
    target = [
      target[0] + sideways * b.right[0] + forward * along[0],
      target[1] + sideways * b.right[1] + forward * along[1],
      0,
    ];
    range = fitRange(local, target, b, box);
  }
  return {
    target: toLonLat(frame, target),
    headingDeg: o.headingDeg,
    pitchDeg: o.pitchDeg,
    rangeM: range * (o.margin ?? DEFAULT_MARGIN),
  };
}

/**
 * Whether every point is inside the inset box seen from `framing`, and the worst point: the
 * test that a stored framing still holds for the geometry and viewport it was made for.
 */
export function framingContains(
  framing: Framing,
  points: LonLat[],
  o: Omit<FramingOptions, "headingDeg" | "pitchDeg" | "margin">,
): { inside: boolean; worst: { index: number; excess: number; x: number; y: number } } {
  const frame = localFrame(framing.target);
  const b = basis({ ...o, headingDeg: framing.headingDeg, pitchDeg: framing.pitchDeg });
  const box = insetBox(o.insets ?? HUD_INSETS);
  const ndc = project(
    points.map((p) => toLocal(frame, p)),
    [0, 0, 0],
    framing.rangeM,
    b,
  );
  const { excess, index } = overshoot(ndc, box);
  const at = ndc[index] ?? { x: NaN, y: NaN };
  return { inside: excess <= 0, worst: { index, excess, x: at.x, y: at.y } };
}

/**
 * Where the camera stands for a framing whose target is on ground at `targetHeightM`
 * (ellipsoid): `range` back along the heading and up by the pitch. The same construction as
 * the API's `overview_bookmark` (apps/api/app/seed/data.py).
 */
export function eyeOf(framing: Framing, targetHeightM: number): EyePose {
  const pitch = framing.pitchDeg * DEG_TO_RAD;
  const horizontal = framing.rangeM * Math.cos(-pitch);
  const vertical = framing.rangeM * Math.sin(-pitch);
  const eye = destination(framing.target, framing.headingDeg + 180, horizontal);
  return {
    longitude: eye.longitude,
    latitude: eye.latitude,
    height: targetHeightM + vertical,
    heading: framing.headingDeg,
    pitch: framing.pitchDeg,
  };
}

/** What a project's arrival must show: every zone outline, chip and machine. */
export function projectGroundPoints(project: Pick<Project, "zones" | "machines">): LonLat[] {
  const points: LonLat[] = [];
  for (const zone of project.zones) {
    for (const ring of outerRings(zone.footprint).slice(0, 1)) {
      for (const [longitude, latitude] of ring) {
        if (longitude !== undefined && latitude !== undefined) points.push({ longitude, latitude });
      }
    }
    points.push(zone.anchor);
  }
  for (const machine of project.machines) points.push(machine.position);
  return points;
}
