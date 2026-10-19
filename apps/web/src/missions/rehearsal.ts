/**
 * The dress rehearsal: a draft plan driven by simulated machines before it is approved.
 *
 * Each machine the plan assigns (or, where it names none, one assumed generic machine per
 * machine class) drives the lanes the map draws for its zones (`MissionManager.previewLanes`,
 * from `footprintLanes`). A zone's lanes go round-robin among its machines; a lane takes its
 * length over the class's assumed speed, a turn its assumed turn time, and the moves between
 * lanes and zones are straight. Every 5 m along a lane the slope is read from an elevation
 * grid, split into the grade along the lane and the cross-slope across it; a lane's distance
 * to mapped creeks is measured exactly, piece against piece. Where a lane goes past a limit
 * its class is assumed to keep (`maxSlopePct` and `creekBufferM` in
 * apps/api/app/services/machine_classes.json, the one table the planner also sizes work from)
 * a hand is raised at the lane's worst point, one per lane and rule, naming the rule, the
 * value, and where both came from.
 *
 * It also counts what it could read: a lane outside the elevation grid, or beyond where the
 * creek lines were fetched, is not "inside the limits", and the summary says how many lanes a
 * rule was read on before it says that rule raised no hand. The lanes are the preview's, far
 * wider apart than a machine's swath, and the summary says that too.
 *
 * Pure: no Cesium and no fetching. `features/mission/rehearsalGround.ts` gathers the ground and
 * `cesium/RehearsalManager.ts` plays the result. Nothing here is measured on a machine: the
 * limits are assumptions, the lanes a preview, and the lines it writes say so.
 */

import type { MachineClassId } from "@twin/contracts";
import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";

import { MACHINE_CLASSES } from "./machineClasses";
import { lerp } from "./rehearsalClock";

export { positionAt, REHEARSAL_SPEEDS, type RehearsalSpeed } from "./rehearsalClock";

/** How often the ground is read along a lane, metres. */
export const SAMPLE_M = 5;

const CLASS_IDS: readonly MachineClassId[] = ["mower", "seeder", "survey", "drone"];

/** What a machine is assumed to do in the rehearsal, all from its class. */
export interface RehearsalCapability {
  classId: MachineClassId;
  /** The class's label, which says "(assumed)". */
  label: string;
  kind: "ground" | "air";
  speedMps: number;
  /** Seconds per turn between lanes; 0 when the class does not time turns (the drone). */
  turnS: number;
  /** Steepest grade or cross-slope the class is assumed to work, %; null for the air. */
  maxSlopePct: number | null;
  /** How close to a mapped creek a lane may come, metres; null for the air. */
  creekBufferM: number | null;
  /** Why the machine has this class, e.g. "its model, Ranger 2 · mower". */
  basis: string;
  /** True when nothing said which class it is, and the default was taken. */
  defaulted: boolean;
}

export function capabilityOf(
  classId: MachineClassId,
  basis: string,
  defaulted = false,
): RehearsalCapability {
  if (classId === "drone") {
    const drone = MACHINE_CLASSES.drone;
    return {
      classId,
      label: drone.label,
      kind: "air",
      speedMps: drone.speedMps,
      turnS: 0,
      maxSlopePct: null,
      creekBufferM: null,
      basis,
      defaulted,
    };
  }
  const ground = MACHINE_CLASSES[classId];
  return {
    classId,
    label: ground.label,
    kind: "ground",
    speedMps: ground.speedMps,
    turnS: ground.turnS,
    maxSlopePct: ground.maxSlopePct,
    creekBufferM: ground.creekBufferM,
    basis,
    defaulted,
  };
}

/** The class a machine model names after its "·", e.g. "Ranger 2 · mower" → "mower". */
export function classOfModel(model: string): MachineClassId | null {
  const word = model.split("·").pop()?.trim().toLowerCase() ?? "";
  return CLASS_IDS.find((id) => id === word) ?? null;
}

/** A ghost: one of the project's machines, or an assumed generic one. */
export interface RehearsalMachine {
  id: string;
  /** The label beside the ghost on the map, e.g. "TR-04 · SIM". */
  label: string;
  /** e.g. "TR-04 Kestrel", or "Assumed generic ground machine (simulated)". */
  name: string;
  generic: boolean;
  capability: RehearsalCapability;
}

/** A lane to drive: two points for a coverage lane, the whole line for a mapped one. */
export interface RehearsalLaneInput {
  points: LonLat[];
  pass: number;
  /** The lane's number in the copy (default: its place in the list, from 1). */
  number?: number;
  /**
   * True for the mapped line a zone follows (a fence): it is the zone's whole work, not one of
   * its drawn lanes, so it cannot be left out of the preview.
   */
  followsLine?: boolean;
}

export interface RehearsalZoneInput {
  zoneId: string;
  /** How lists name the zone: its id ("Z-14") or, for a Land's zones, its name. */
  label: string;
  /** In driving order, as drawn; lanes left out of the preview are not here. */
  lanes: RehearsalLaneInput[];
  /** The machines the plan assigns to the zone, in plan order. */
  machineIds: string[];
  /** The plan's machine class for the zone (its PATH row), when it has one. */
  classId: MachineClassId | null;
}

/** Heights on a lon/lat grid, row 0 at the north (a decoded `LonLatGrid`). */
export interface ElevationGrid {
  west: number;
  south: number;
  east: number;
  north: number;
  columns: number;
  rows: number;
  /** Metres, row-major from the north-west cell; NaN where there is no value. */
  values: Float32Array;
}

export interface ElevationSource {
  grid: ElevationGrid;
  /** What the heights are, e.g. "USGS 3DEP 1/3″ bare-earth DEM, 10 m". */
  label: string;
  /** Which record they came from, e.g. "raster 8c5becbb"; null for live samples. */
  ref: string | null;
  /** The Land raster's id, when the heights are one (workable ground reads the same one). */
  rasterId?: string;
  /** Where the grid has heights, said where a lane has none, e.g. "it covers the Land only". */
  extent?: string;
}

export interface CreekLine {
  id: string;
  name: string | null;
  /** NHD permanence: perennial, intermittent, ephemeral, unclassified, artificial-path, other. */
  kind: string;
  coordinates: LonLat[];
}

/** Where creek lines were fetched: beyond it a creek may run that the rehearsal never saw. */
export interface CreekCoverage {
  /** The outline the lines were read for: polygons, each its outer ring then its holes. */
  polygons: LonLat[][][];
  /** How far past the outline the lines still reach, metres (0: clipped to it). */
  marginM: number;
  /** The area in words, e.g. "within 100 m of Oak Bench Ranch". */
  label: string;
}

export interface CreekSource {
  lines: CreekLine[];
  /** e.g. "USGS NHD flowlines, 1:24,000". */
  label: string;
  /** Where the lines were read; absent when they cover everywhere the lanes go. */
  coverage?: CreekCoverage | null;
  /** The source stopped at its record limit: more lines may exist than were read. */
  truncated?: boolean;
  /** Anything else that limits the lines, in a sentence (the demo's single supplied creek). */
  note?: string;
}

export interface ProjectMachineInput {
  id: string;
  name: string;
  model: string;
}

export interface RehearsalInput {
  zones: RehearsalZoneInput[];
  machines: readonly ProjectMachineInput[];
  elevation: ElevationSource | null;
  /** Why slope is not checked, when `elevation` is null. */
  elevationMissing?: string;
  creeks: CreekSource | null;
  /** Why the creek buffer is not checked, when `creeks` is null. */
  creeksMissing?: string;
  /** Ephemeral drainages count as creeks too (off by default: they run only after rain). */
  includeEphemeral?: boolean;
  sampleM?: number;
}

export interface LaneRef {
  zoneId: string;
  /** Index into the zone's lanes as given (0-based). */
  index: number;
}

export interface RehearsalSegment {
  t0: number;
  t1: number;
  from: LonLat;
  to: LonLat;
  /** The lane being worked; null while turning or moving between lanes. */
  lane: LaneRef | null;
}

export interface RehearsalTrack {
  machine: RehearsalMachine;
  segments: RehearsalSegment[];
  endS: number;
}

/**
 * How much of a lane a rule could read: every sample, some, or none (outside the elevation
 * grid, or beyond where creeks were fetched). Null where the rule does not apply: an air
 * machine, or no source at all.
 */
export type LaneRead = "all" | "part" | "none" | null;

export interface RehearsedLane extends LaneRef {
  /** How the copy numbers the lane. */
  number: number;
  machineId: string;
  points: LonLat[];
  lengthM: number;
  pass: number;
  /** When the lane is started and finished, simulated seconds from the start. */
  t0: number;
  t1: number;
  /** A drawn lane can be left out of the preview; a followed line (a fence) cannot. */
  removable: boolean;
  slopeRead: LaneRead;
  creekRead: LaneRead;
}

export type HandKind = "cross-slope" | "grade" | "creek-buffer";

export interface RaisedHand extends LaneRef {
  id: string;
  kind: HandKind;
  /** How the copy numbers the lane. */
  number: number;
  machineId: string;
  zoneLabel: string;
  /** The lane's worst point and when the machine reaches it. */
  at: LonLat;
  t: number;
  /** % for the slopes, metres from the creek for the buffer. */
  value: number;
  limit: number;
  /** The short line on the map, e.g. "31 % cross-slope · limit 25 % (assumed)". */
  label: string;
  /** The rule in a sentence, with the class it is assumed for. */
  rule: string;
  /** Where the value came from, e.g. "Slope: USGS 3DEP … raster 8c5becbb". */
  source: string;
  /** The creek's NHD id and name, for a buffer hand. */
  creekId?: string;
  creekName?: string | null;
}

/**
 * One line of the end summary. info: what was driven; hand: lanes over a limit; clear: a rule
 * that raised no hand on the lanes it read; gap: lanes or rules it could not read.
 */
export interface SummaryLine {
  tone: "info" | "hand" | "clear" | "gap";
  text: string;
}

export interface Rehearsal {
  machines: RehearsalMachine[];
  tracks: RehearsalTrack[];
  lanes: RehearsedLane[];
  hands: RaisedHand[];
  durationS: number;
  lengthM: number;
  summary: SummaryLine[];
  /** What was not checked, and why, in sentences. */
  skipped: string[];
  /** Where the checks came from, for the source chips. */
  sources: string[];
  /** The creek lines the buffer was checked against (drawn while rehearsing). */
  creeks: CreekLine[];
}

interface Frame {
  origin: LonLat;
  kx: number;
  ky: number;
}

interface XY {
  x: number;
  y: number;
}

function frameAt(origin: LonLat): Frame {
  return {
    origin,
    kx: Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M,
    ky: DEG_TO_RAD * EARTH_RADIUS_M,
  };
}

function toXY(frame: Frame, p: LonLat): XY {
  return {
    x: (p.longitude - frame.origin.longitude) * frame.kx,
    y: (p.latitude - frame.origin.latitude) * frame.ky,
  };
}

function distanceM(frame: Frame, a: LonLat, b: LonLat): number {
  const p = toXY(frame, a);
  const q = toXY(frame, b);
  return Math.hypot(q.x - p.x, q.y - p.y);
}

function cellValue(grid: ElevationGrid, col: number, row: number): number {
  const c = Math.min(grid.columns - 1, Math.max(0, col));
  const r = Math.min(grid.rows - 1, Math.max(0, row));
  return grid.values[r * grid.columns + c] ?? Number.NaN;
}

/** Height at a point, bilinear between cell centres; NaN outside the grid or on no data. */
export function elevationAt(grid: ElevationGrid, longitude: number, latitude: number): number {
  if (
    longitude < grid.west ||
    longitude > grid.east ||
    latitude < grid.south ||
    latitude > grid.north
  )
    return Number.NaN;
  const dlon = (grid.east - grid.west) / grid.columns;
  const dlat = (grid.north - grid.south) / grid.rows;
  const x = (longitude - grid.west) / dlon - 0.5;
  const y = (grid.north - latitude) / dlat - 0.5;
  const c0 = Math.floor(x);
  const r0 = Math.floor(y);
  const fx = x - c0;
  const fy = y - r0;
  const z00 = cellValue(grid, c0, r0);
  const z10 = cellValue(grid, c0 + 1, r0);
  const z01 = cellValue(grid, c0, r0 + 1);
  const z11 = cellValue(grid, c0 + 1, r0 + 1);
  if ([z00, z10, z01, z11].every(Number.isFinite)) {
    return (z00 * (1 - fx) + z10 * fx) * (1 - fy) + (z01 * (1 - fx) + z11 * fx) * fy;
  }
  // At the edge of the data: the cell the point is in, if it has a value.
  return cellValue(grid, Math.round(x), Math.round(y));
}

/**
 * The ground's gradient at a point, metres per metre east and north: central differences one
 * cell either side (one-sided at the edge of the data), as the API derives its slope band.
 * Null where there is no height.
 */
export function gradientAt(
  grid: ElevationGrid,
  longitude: number,
  latitude: number,
): { east: number; north: number } | null {
  const z = elevationAt(grid, longitude, latitude);
  if (!Number.isFinite(z)) return null;
  const dlon = (grid.east - grid.west) / grid.columns;
  const dlat = (grid.north - grid.south) / grid.rows;
  const dx = dlon * Math.cos(latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
  const dy = dlat * DEG_TO_RAD * EARTH_RADIUS_M;
  const along = (plus: number, minus: number, step: number): number | null => {
    if (Number.isFinite(plus) && Number.isFinite(minus)) return (plus - minus) / (2 * step);
    if (Number.isFinite(plus)) return (plus - z) / step;
    if (Number.isFinite(minus)) return (z - minus) / step;
    return null;
  };
  const east = along(
    elevationAt(grid, longitude + dlon, latitude),
    elevationAt(grid, longitude - dlon, latitude),
    dx,
  );
  const north = along(
    elevationAt(grid, longitude, latitude + dlat),
    elevationAt(grid, longitude, latitude - dlat),
    dy,
  );
  if (east === null || north === null) return null;
  return { east, north };
}

/** The point of segment a→b nearest to p: its distance, and where it is along a→b (0 to 1). */
function nearestOn(p: XY, a: XY, b: XY): { d: number; f: number } {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const length2 = vx * vx + vy * vy;
  const f =
    length2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / length2)) : 0;
  return { d: Math.hypot(p.x - (a.x + f * vx), p.y - (a.y + f * vy)), f };
}

/**
 * How close segment p→q comes to segment a→b, and where along p→q (0 to 1): 0 at the crossing
 * when they cross; otherwise the nearest pair of points, one of which is an end of a segment.
 */
function closestApproach(p: XY, q: XY, a: XY, b: XY): { d: number; f: number } {
  const rx = q.x - p.x;
  const ry = q.y - p.y;
  const sx = b.x - a.x;
  const sy = b.y - a.y;
  const denominator = rx * sy - ry * sx;
  if (denominator !== 0) {
    const wx = a.x - p.x;
    const wy = a.y - p.y;
    const f = (wx * sy - wy * sx) / denominator;
    const g = (wx * ry - wy * rx) / denominator;
    if (f >= 0 && f <= 1 && g >= 0 && g <= 1) return { d: 0, f };
  }
  const toA = nearestOn(a, p, q);
  const toB = nearestOn(b, p, q);
  let best = { d: nearestOn(p, a, b).d, f: 0 };
  const fromQ = nearestOn(q, a, b).d;
  if (fromQ < best.d) best = { d: fromQ, f: 1 };
  if (toA.d < best.d) best = toA;
  if (toB.d < best.d) best = toB;
  return best;
}

interface Edge {
  a: XY;
  b: XY;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function edgeOf(a: XY, b: XY): Edge {
  return {
    a,
    b,
    minX: Math.min(a.x, b.x),
    maxX: Math.max(a.x, b.x),
    minY: Math.min(a.y, b.y),
    maxY: Math.max(a.y, b.y),
  };
}

/** Even-odd: whether p is inside a ring (closed or not). */
function ringContains(ring: readonly XY[], p: XY): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i];
    const b = ring[j];
    if (!a || !b || a.y > p.y === b.y > p.y) continue;
    if (p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/**
 * Whether the whole creek buffer around a lane point lies where creek lines were fetched: the
 * point's signed distance to the outline (negative inside) is at most the margin the lines
 * reach past it, less the buffer. Beyond that a creek could be inside the buffer unseen.
 */
function coverageTest(frame: Frame, coverage: CreekCoverage): (p: XY, bufferM: number) => boolean {
  const polygons = coverage.polygons.map((rings) =>
    rings.map((ring) => ring.map((q) => toXY(frame, q))),
  );
  const edges: Edge[] = [];
  for (const rings of polygons)
    for (const ring of rings)
      ring.forEach((a, i) => {
        const b = ring[(i + 1) % ring.length];
        if (b && (a.x !== b.x || a.y !== b.y)) edges.push(edgeOf(a, b));
      });
  const inside = (p: XY) =>
    polygons.some(
      ([outer, ...holes]) =>
        outer !== undefined && ringContains(outer, p) && !holes.some((h) => ringContains(h, p)),
    );
  return (p, bufferM) => {
    const reach = coverage.marginM - bufferM;
    const within = inside(p);
    if (within && reach >= 0) return true;
    if (!within && reach < 0) return false;
    const r = Math.abs(reach);
    const near = edges.some(
      (e) =>
        p.x >= e.minX - r &&
        p.x <= e.maxX + r &&
        p.y >= e.minY - r &&
        p.y <= e.maxY + r &&
        nearestOn(p, e.a, e.b).d <= r,
    );
    // Inside: covered unless the outline is nearer than the buffer reaches past the margin.
    // Outside: covered only within the margin less the buffer.
    return within ? !near : near;
  };
}

/**
 * How far apart a zone's drawn lanes are, metres: the median distance across from a lane to
 * the next pass's, so a zone in several parts (each swept from its own edge) is not measured
 * from one part to the next. Null for a followed line or a single pass.
 */
function laneSpacingM(frame: Frame, lanes: readonly RehearsalLaneInput[]): number | null {
  const drawn = lanes.filter((lane) => !lane.followsLine && lane.points.length === 2);
  const offsets: number[] = [];
  for (let i = 0; i < drawn.length; i++) {
    const lane = drawn[i];
    const next = drawn.slice(i + 1).find((other) => other.pass === (lane?.pass ?? 0) + 1);
    const [a0, b0] = lane?.points ?? [];
    const p0 = next?.points[0];
    if (!a0 || !b0 || !p0) continue;
    const a = toXY(frame, a0);
    const b = toXY(frame, b0);
    const p = toXY(frame, p0);
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    if (length > 0)
      offsets.push(Math.abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / length);
  }
  if (offsets.length === 0) return null;
  offsets.sort((x, y) => x - y);
  return offsets[Math.floor(offsets.length / 2)] ?? null;
}

/** "~46 m apart", or "~20 to 46 m apart" over zones; null when no zone has drawn passes. */
function spacingText(spacings: readonly number[]): string | null {
  const rounded = spacings.filter(Number.isFinite).map((s) => Math.max(1, Math.round(s)));
  if (rounded.length === 0) return null;
  const low = Math.min(...rounded);
  const high = Math.max(...rounded);
  return low === high ? `~${low} m apart` : `~${low} to ${high} m apart`;
}

interface CreekPiece extends Edge {
  line: CreekLine;
}

/** NHD kinds that count as creeks: every natural channel but the ephemeral, unless asked. */
export function countsAsCreek(kind: string, includeEphemeral: boolean): boolean {
  if (kind === "ephemeral") return includeEphemeral;
  return (
    kind === "perennial" ||
    kind === "intermittent" ||
    kind === "unclassified" ||
    kind === "artificial-path"
  );
}

function creekPieces(frame: Frame, creeks: CreekSource, includeEphemeral: boolean): CreekPiece[] {
  const pieces: CreekPiece[] = [];
  for (const line of creeks.lines) {
    if (!countsAsCreek(line.kind, includeEphemeral)) continue;
    for (let i = 0; i + 1 < line.coordinates.length; i++) {
      const p = line.coordinates[i];
      const q = line.coordinates[i + 1];
      if (!p || !q) continue;
      pieces.push({ line, ...edgeOf(toXY(frame, p), toXY(frame, q)) });
    }
  }
  return pieces;
}

function pct(value: number): string {
  return `${Math.round(value)} %`;
}

function lanesText(refs: readonly { label: string; number: number }[]): string {
  const byZone = new Map<string, number[]>();
  for (const ref of refs) byZone.set(ref.label, [...(byZone.get(ref.label) ?? []), ref.number]);
  return [...byZone.entries()]
    .map(([label, numbers]) => {
      const sorted = [...new Set(numbers)].sort((a, b) => a - b);
      const shown = sorted.slice(0, 6).join(", ");
      const more = sorted.length > 6 ? ` and ${sorted.length - 6} more` : "";
      return `${label} lane${sorted.length === 1 ? "" : "s"} ${shown}${more}`;
    })
    .join("; ");
}

/** "2 h 14 min", "45 min", "under a minute". */
export function formatSimDuration(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return "under a minute";
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h === 0) return `${m} min`;
  return m === 0 ? `${h} h` : `${h} h ${m} min`;
}

function km(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`;
}

/** The ghosts and which zone's lanes each drives. */
function castMachines(
  zones: readonly RehearsalZoneInput[],
  projectMachines: readonly ProjectMachineInput[],
): { machines: RehearsalMachine[]; crews: Map<string, string[]>; notes: string[] } {
  const machines = new Map<string, RehearsalMachine>();
  const crews = new Map<string, string[]>();
  const notes: string[] = [];
  const generic = new Map<MachineClassId, RehearsalMachine>();
  for (const zone of zones) {
    const crew: string[] = [];
    for (const id of zone.machineIds) {
      const existing = machines.get(id);
      if (existing) {
        crew.push(id);
        continue;
      }
      const known = projectMachines.find((m) => m.id === id);
      const fromModel = known ? classOfModel(known.model) : null;
      const classId = fromModel ?? zone.classId ?? "mower";
      const capability = capabilityOf(
        classId,
        fromModel && known
          ? `its model, ${known.model}`
          : zone.classId
            ? `the plan's machine class for ${zone.label}`
            : "a default assumption: nothing says which class it is",
        !fromModel && !zone.classId,
      );
      if (!fromModel) {
        notes.push(
          `${id}${known ? ` (model “${known.model}”)` : ""}: no machine class in its model, so it is rehearsed as a ${capability.label}${capability.defaulted ? ", a default assumption" : ", the plan's class for its zone"}.`,
        );
      }
      machines.set(id, {
        id,
        label: `${id} · SIM`,
        name: known?.name ?? id,
        generic: false,
        capability,
      });
      crew.push(id);
    }
    if (crew.length === 0) {
      const classId = zone.classId ?? "mower";
      let ghost = generic.get(classId);
      if (!ghost) {
        const id = `SIM-${generic.size + 1}`;
        const capability = capabilityOf(
          classId,
          zone.classId
            ? `the plan's machine class for ${zone.label}`
            : "a default assumption: the plan names no machine class",
          !zone.classId,
        );
        ghost = {
          id,
          label: `${id} · ${classId}`,
          name:
            capability.kind === "air"
              ? "Assumed generic mapping drone (simulated)"
              : "Assumed generic ground machine (simulated)",
          generic: true,
          capability,
        };
        generic.set(classId, ghost);
        machines.set(id, ghost);
      }
      crew.push(ghost.id);
    }
    crews.set(zone.zoneId, crew);
  }
  if (generic.size > 0) {
    notes.unshift(
      `No machines are assigned${machines.size > generic.size ? " to some zones" : ""}, so one assumed generic machine (simulated) per machine class drives their lanes: ${[...generic.values()].map((g) => `${g.id}, ${g.capability.label}`).join("; ")}.`,
    );
  }
  return { machines: [...machines.values()], crews, notes };
}

interface Worst {
  value: number;
  at: LonLat;
  t: number;
  creek?: CreekLine;
  /** The lane runs over the creek's line. */
  crosses?: boolean;
}

/**
 * How many of the ground lanes a rule applies to it read in full, in part, or not at all, and
 * how many metres of those lanes it could not read.
 */
interface ReadTally {
  total: number;
  all: number;
  part: number;
  none: number;
  laneM: number;
  unreadM: number;
}

/** Metres of a lane each rule could not read. */
type Unread = Map<RehearsedLane, { slope: number; creek: number }>;

function tally(
  lanes: readonly RehearsedLane[],
  rule: "slope" | "creek",
  unread: Unread,
): ReadTally {
  const counts: ReadTally = { total: 0, all: 0, part: 0, none: 0, laneM: 0, unreadM: 0 };
  for (const lane of lanes) {
    const read = rule === "slope" ? lane.slopeRead : lane.creekRead;
    if (read === null) continue;
    counts.total++;
    counts[read]++;
    counts.laneM += lane.lengthM;
    counts.unreadM += unread.get(lane)?.[rule] ?? 0;
  }
  return counts;
}

function readOf(samples: number, read: number): Exclude<LaneRead, null> {
  if (read === 0) return "none";
  return read === samples ? "all" : "part";
}

function lanesWord(n: number): string {
  return `lane${n === 1 ? "" : "s"}`;
}

/** "140 m of 20.2 km of lanes (3 of 28 lanes wholly, 2 in part)": how much went unread. */
function unreadText({ total, part, none, laneM, unreadM: unread }: ReadTally): string {
  const which =
    none && part
      ? `${none} of ${total} lanes wholly, ${part} in part`
      : none
        ? `${none} of ${total} ${lanesWord(total)}`
        : `in part on ${part} of ${total} ${lanesWord(total)}`;
  return `${km(unread)} of ${km(laneM)} of lanes (${which})`;
}

/**
 * Plays the plan through on paper: who drives which lane when, and where a hand goes up.
 * Deterministic for the same input.
 */
export function planRehearsal(input: RehearsalInput): Rehearsal {
  const sampleM = input.sampleM ?? SAMPLE_M;
  const includeEphemeral = input.includeEphemeral ?? false;
  const { machines, crews, notes } = castMachines(input.zones, input.machines);
  const allPoints = input.zones.flatMap((z) => z.lanes.flatMap((l) => l.points));
  const origin = allPoints.length
    ? {
        longitude: allPoints.reduce((s, p) => s + p.longitude, 0) / allPoints.length,
        latitude: allPoints.reduce((s, p) => s + p.latitude, 0) / allPoints.length,
      }
    : { longitude: 0, latitude: 0 };
  const frame = frameAt(origin);
  const creeks = input.creeks;
  const pieces = creeks ? creekPieces(frame, creeks, includeEphemeral) : [];
  const covered = creeks?.coverage ? coverageTest(frame, creeks.coverage) : null;
  const maxBuffer = Math.max(0, ...machines.map((m) => m.capability.creekBufferM ?? 0));

  const clocks = new Map<string, { t: number; at: LonLat | null; zoneId: string | null }>();
  const segments = new Map<string, RehearsalSegment[]>();
  for (const machine of machines) {
    clocks.set(machine.id, { t: 0, at: null, zoneId: null });
    segments.set(machine.id, []);
  }
  const lanes: RehearsedLane[] = [];
  const unread: Unread = new Map();
  const hands: RaisedHand[] = [];
  const skipped = new Set<string>();
  const airChecked = new Set<string>();

  for (const zone of input.zones) {
    const crew = crews.get(zone.zoneId) ?? [];
    zone.lanes.forEach((lane, index) => {
      const machineId = crew[index % crew.length];
      const machine = machines.find((m) => m.id === machineId);
      if (!machineId || !machine || lane.points.length < 2) return;
      const clock = clocks.get(machineId);
      const track = segments.get(machineId);
      const start = lane.points[0];
      if (!clock || !track || !start) return;
      const capability = machine.capability;
      // The move to the lane: a turn and a straight run within a zone, a straight run between.
      if (clock.at) {
        const move = distanceM(frame, clock.at, start);
        const turn = clock.zoneId === zone.zoneId ? capability.turnS : 0;
        const t1 = clock.t + move / capability.speedMps + turn;
        if (t1 > clock.t) track.push({ t0: clock.t, t1, from: clock.at, to: start, lane: null });
        clock.t = t1;
      }
      const laneT0 = clock.t;
      let lengthM = 0;
      const worst = new Map<HandKind, Worst>();
      const ground = capability.kind === "ground";
      if (!ground && !airChecked.has(machine.id)) {
        airChecked.add(machine.id);
        skipped.add(
          `${machine.id} is a ${capability.label}: it flies, so the slope and creek rules for ground machines were not checked on its lanes.`,
        );
      }
      const maxSlope = ground ? capability.maxSlopePct : null;
      const elevation = maxSlope !== null ? input.elevation : null;
      const buffer = ground ? (capability.creekBufferM ?? 0) : 0;
      const creekRule = creeks !== null && buffer > 0;
      // Samples each rule could read on this lane, of those it tried.
      const slopeRead = { samples: 0, read: 0 };
      const creekRead = { samples: 0, read: 0 };
      for (let i = 0; i + 1 < lane.points.length; i++) {
        const a = lane.points[i];
        const b = lane.points[i + 1];
        if (!a || !b) continue;
        const pieceM = distanceM(frame, a, b);
        if (!(pieceM > 0)) continue;
        const t0 = laneT0 + lengthM / capability.speedMps;
        const t1 = t0 + pieceM / capability.speedMps;
        track.push({ t0, t1, from: a, to: b, lane: { zoneId: zone.zoneId, index } });
        lengthM += pieceM;
        if (!ground) continue;
        const pa = toXY(frame, a);
        const pb = toXY(frame, b);
        if (creekRule) {
          // Piece against piece, exactly: a lane that crosses a creek between two samples
          // still crosses it, and its nearest point is where the hand goes.
          for (const creek of pieces) {
            if (
              creek.maxX < Math.min(pa.x, pb.x) - buffer ||
              creek.minX > Math.max(pa.x, pb.x) + buffer ||
              creek.maxY < Math.min(pa.y, pb.y) - buffer ||
              creek.minY > Math.max(pa.y, pb.y) + buffer
            )
              continue;
            const near = closestApproach(pa, pb, creek.a, creek.b);
            const current = worst.get("creek-buffer");
            if (near.d >= buffer || (current && near.d >= current.value)) continue;
            worst.set("creek-buffer", {
              value: near.d,
              at: lerp(a, b, near.f),
              t: t0 + (t1 - t0) * near.f,
              creek: creek.line,
              crosses: near.d === 0,
            });
          }
        }
        if (!elevation && !creekRule) continue;
        const ux = (pb.x - pa.x) / pieceM;
        const uy = (pb.y - pa.y) / pieceM;
        const steps = Math.max(1, Math.ceil(pieceM / sampleM));
        for (let k = 0; k <= steps; k++) {
          // A shared corner is read once, at the start of the next piece.
          if (k === steps && i + 2 < lane.points.length) continue;
          const f = k / steps;
          if (elevation && maxSlope !== null) {
            const at = lerp(a, b, f);
            slopeRead.samples++;
            const g = gradientAt(elevation.grid, at.longitude, at.latitude);
            if (g) {
              slopeRead.read++;
              const grade = Math.abs(g.east * ux + g.north * uy) * 100;
              const cross = Math.abs(-g.east * uy + g.north * ux) * 100;
              const t = t0 + (t1 - t0) * f;
              for (const [kind, value] of [
                ["grade", grade],
                ["cross-slope", cross],
              ] as const) {
                if (value <= maxSlope) continue;
                const current = worst.get(kind);
                if (!current || value > current.value) worst.set(kind, { value, at, t });
              }
            }
          }
          if (creekRule) {
            creekRead.samples++;
            const p = { x: pa.x + (pb.x - pa.x) * f, y: pa.y + (pb.y - pa.y) * f };
            if (!covered || covered(p, buffer)) creekRead.read++;
          }
        }
      }
      const laneT1 = laneT0 + lengthM / capability.speedMps;
      const end = lane.points[lane.points.length - 1] ?? start;
      clock.t = laneT1;
      clock.at = end;
      clock.zoneId = zone.zoneId;
      const number = lane.number ?? index + 1;
      let creekReading: LaneRead = creekRule ? readOf(creekRead.samples, creekRead.read) : null;
      const unreadPart = (read: { samples: number; read: number }) =>
        read.samples > 0 ? lengthM * (1 - read.read / read.samples) : lengthM;
      // A creek line found past the fetched area is still a creek: the lane was read there.
      if (creekReading === "none" && worst.has("creek-buffer")) creekReading = "part";
      const rehearsed: RehearsedLane = {
        zoneId: zone.zoneId,
        index,
        number,
        machineId,
        points: lane.points,
        lengthM,
        pass: lane.pass,
        t0: laneT0,
        t1: laneT1,
        removable: !lane.followsLine,
        slopeRead: elevation ? readOf(slopeRead.samples, slopeRead.read) : null,
        creekRead: creekReading,
      };
      lanes.push(rehearsed);
      unread.set(rehearsed, {
        slope: elevation ? unreadPart(slopeRead) : 0,
        creek: creekRule ? unreadPart(creekRead) : 0,
      });
      for (const kind of ["cross-slope", "grade", "creek-buffer"] as const) {
        const found = worst.get(kind);
        if (!found) continue;
        hands.push(
          raiseHand(kind, found, {
            zone,
            index,
            number,
            machine,
            elevation: input.elevation,
            creeks,
          }),
        );
      }
    });
  }

  const tracks: RehearsalTrack[] = machines
    .map((machine) => {
      const list = segments.get(machine.id) ?? [];
      return { machine, segments: list, endS: list[list.length - 1]?.t1 ?? 0 };
    })
    .filter((track) => track.segments.length > 0);
  // A lane's end and its last segment's are the same sum in a different order: take the later.
  const durationS = [...tracks.map((t) => t.endS), ...lanes.map((l) => l.t1)].reduce(
    (a, b) => Math.max(a, b),
    0,
  );
  const lengthM = lanes.reduce((s, l) => s + l.lengthM, 0);
  hands.sort((a, b) => a.t - b.t || a.id.localeCompare(b.id));

  const groundLanes = lanes.filter(
    (lane) => machines.find((m) => m.id === lane.machineId)?.capability.kind === "ground",
  );
  const slope = tally(groundLanes, "slope", unread);
  const water = tally(groundLanes, "creek", unread);
  const slopeChecked = slope.all + slope.part;
  const creekChecked = water.all + water.part;
  const spacing = spacingText(
    input.zones.map((z) => laneSpacingM(frame, z.lanes)).filter((s): s is number => s !== null),
  );
  const drawn = spacing ? `preview spacing, ${spacing}` : null;

  // What could not be read, said in the summary as well as the list: a rule read on no lane,
  // or on only some, has not shown those lanes to be inside the limits.
  const gaps: string[] = [];
  if (groundLanes.length > 0) {
    if (!input.elevation)
      gaps.push(input.elevationMissing ?? "Slope not checked: no terrain for this ground.");
    else if (slope.none || slope.part)
      gaps.push(
        `Slope not read on ${unreadText(slope)}: ${input.elevation.label} has no heights there${input.elevation.extent ? `; ${input.elevation.extent}` : ""}.`,
      );
    if (!creeks)
      gaps.push(
        input.creeksMissing ?? "Creek buffer not checked: no mapped creeks for this ground.",
      );
    else if (water.total > 0) {
      if (creeks.coverage && (water.none || water.part))
        gaps.push(
          `Creeks checked only ${creeks.coverage.label}: ${unreadText(water)} lie outside it, so a creek there would raise no hand.`,
        );
      if (creeks.truncated)
        gaps.push(
          `${creeks.label} stopped at its record limit here: more flowlines may exist than the ${creeks.lines.length} read.`,
        );
    }
  }
  for (const gap of gaps) skipped.add(gap);
  if (groundLanes.length > 0 && creeks) {
    if (!includeEphemeral && creeks.lines.some((l) => l.kind === "ephemeral"))
      skipped.add(
        "Ephemeral drainages (NHD channels that run only after rain) are not counted as creeks.",
      );
    if (creeks.note) skipped.add(creeks.note);
  }
  skipped.add(
    `The ghosts drive the lanes the map draws${drawn ? ` (${drawn})` : ""}, not every swath, so the ground between drawn lanes is not checked; turns and moves between lanes are straight and not checked.`,
  );

  const summary: SummaryLine[] = [
    {
      tone: "info",
      text: `${tracks.length} simulated machine${tracks.length === 1 ? "" : "s"} drove ${lanes.length} drawn lane${lanes.length === 1 ? "" : "s"} (${km(lengthM)}${drawn ? `, ${drawn}` : ""}) in ${formatSimDuration(durationS)} at assumed speeds.`,
    },
  ];
  for (const text of handLines(hands, { slope: slopeChecked, creek: creekChecked }))
    summary.push({ tone: "hand", text });
  // A rule that raised no hand is clear only on the lanes it read, and only on those lanes.
  const fully: string[] = [];
  const partly: string[] = [];
  if (input.elevation && slopeChecked > 0 && !hands.some((h) => h.kind !== "creek-buffer")) {
    if (slope.none || slope.part)
      partly.push(
        `No slope hands on the ${slopeChecked} of ${slope.total} lanes where slope was read.`,
      );
    else fully.push("slope");
  }
  if (creeks && creekChecked > 0 && !hands.some((h) => h.kind === "creek-buffer")) {
    if (water.none || water.part || creeks.truncated)
      partly.push(
        `No creek-buffer hands on the ${creekChecked} of ${water.total} lanes checked for creeks.`,
      );
    else fully.push("the creek buffer");
  }
  if (fully.length)
    summary.push({
      tone: "clear",
      text: `No hands for ${fully.join(" or ")} on the drawn lanes${drawn ? ` (${drawn}); the ground between them is not checked` : ""}.`,
    });
  for (const text of partly) summary.push({ tone: "clear", text });
  for (const text of gaps) summary.push({ tone: "gap", text });

  const sources: string[] = [];
  if (input.elevation && slopeChecked > 0)
    sources.push(
      `Slope: ${input.elevation.label}${input.elevation.ref ? `, ${input.elevation.ref}` : ""}`,
    );
  const checkedCreeks = creeks !== null && groundLanes.length > 0 && maxBuffer > 0;
  if (creeks && creekChecked > 0) sources.push(`Creek: ${creeks.label}`);
  sources.push("Limits: assumed per machine class (machine_classes.json)");

  return {
    machines: tracks.map((t) => t.machine),
    tracks,
    lanes,
    hands,
    durationS,
    lengthM,
    summary,
    skipped: [...notes, ...skipped],
    sources,
    creeks:
      checkedCreeks && creeks
        ? creeks.lines.filter((line) => countsAsCreek(line.kind, includeEphemeral))
        : [],
  };
}

function raiseHand(
  kind: HandKind,
  found: Worst,
  context: {
    zone: RehearsalZoneInput;
    index: number;
    number: number;
    machine: RehearsalMachine;
    elevation: ElevationSource | null;
    creeks: CreekSource | null;
  },
): RaisedHand {
  const { zone, index, number, machine } = context;
  const capability = machine.capability;
  const base = {
    id: `${zone.zoneId}:${index}:${kind}`,
    kind,
    zoneId: zone.zoneId,
    index,
    number,
    machineId: machine.id,
    zoneLabel: zone.label,
    at: found.at,
    t: found.t,
  };
  const where = `${zone.label} lane ${number}`;
  // "a robotic brush mower": the class in a sentence that already says the limit is assumed.
  const plain = capability.label.replace(/\s*\(assumed\)$/, "");
  const machineClass = `${plain.slice(0, 1).toLowerCase()}${plain.slice(1)}`;
  if (kind === "creek-buffer") {
    const limit = capability.creekBufferM ?? 0;
    const creek = found.creek;
    const name = creek?.name ?? "a mapped creek";
    const crosses = found.crosses === true;
    return {
      ...base,
      value: found.value,
      limit,
      label: `${crosses ? `Crosses ${creek?.name ?? "a creek"}` : `${Math.round(found.value)} m from ${creek?.name ?? "a creek"}`} · buffer ${limit} m (assumed)`,
      rule: `${where} ${crosses ? `crosses ${name}` : `comes within ${Math.round(found.value)} m of ${name}`}. Assumed buffer for a ${machineClass}: ${limit} m from creeks.`,
      source: `Creek: ${context.creeks?.label ?? "mapped creeks"}${creek ? `, flowline ${creek.id} (${creek.kind})` : ""}`,
      creekId: creek?.id,
      creekName: creek?.name ?? null,
    };
  }
  const limit = capability.maxSlopePct ?? 0;
  const what = kind === "grade" ? "grade" : "cross-slope";
  return {
    ...base,
    value: found.value,
    limit,
    label: `${pct(found.value)} ${what} · limit ${pct(limit)} (assumed)`,
    rule: `${where} reaches ${pct(found.value)} ${what === "grade" ? "grade along the lane" : "cross-slope across the lane"} at its steepest. Assumed limit for a ${machineClass}: ${pct(limit)}.`,
    source: `Slope: ${context.elevation?.label ?? "elevation"}${context.elevation?.ref ? `, ${context.elevation.ref}` : ""}`,
  };
}

/**
 * One summary line per rule that raised hands: how many of the lanes that rule read, and
 * which. `checked` counts the ground lanes each rule could read at all.
 */
function handLines(
  hands: readonly RaisedHand[],
  checked: { slope: number; creek: number },
): string[] {
  const lines: string[] = [];
  const of = (n: number, k: number) => `${n} of ${k} checked ${lanesWord(k)}`;
  const refs = (list: readonly RaisedHand[]) =>
    lanesText(list.map((h) => ({ label: h.zoneLabel, number: h.number })));
  const count = (list: readonly RaisedHand[]) =>
    new Set(list.map((h) => `${h.zoneId}:${h.index}`)).size;
  for (const kind of ["cross-slope", "grade"] as const) {
    const list = hands.filter((h) => h.kind === kind);
    if (list.length === 0) continue;
    const limits = [...new Set(list.map((h) => h.limit))];
    const over =
      limits.length === 1
        ? `over ${pct(limits[0] ?? 0)} ${kind}`
        : `over their class's ${kind} limit`;
    lines.push(`${of(count(list), checked.slope)} ${over} (${refs(list)})`);
  }
  const creek = hands.filter((h) => h.kind === "creek-buffer");
  if (creek.length) {
    const limits = [...new Set(creek.map((h) => h.limit))];
    const named = [
      ...new Set(creek.map((h) => h.creekName).filter((n): n is string => Boolean(n))),
    ];
    const unnamed = new Set(creek.filter((h) => !h.creekName).map((h) => h.creekId)).size;
    const creeks =
      named.length === 0
        ? unnamed === 1
          ? "a mapped creek"
          : "mapped creeks"
        : `${named.join(", ")}${unnamed ? ` and ${unnamed} unnamed channel${unnamed === 1 ? "" : "s"}` : ""}`;
    const buffer = limits.length === 1 ? `the ${limits[0]} m buffer` : "their class's buffer";
    lines.push(`${of(count(creek), checked.creek)} inside ${buffer} of ${creeks} (${refs(creek)})`);
  }
  return lines;
}
