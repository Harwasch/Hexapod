/**
 * The dress rehearsal: a draft plan driven by simulated machines before it is approved.
 *
 * Each machine the plan assigns (or, where it names none, one assumed generic machine per
 * machine class) drives the lanes the map draws for its zones (`MissionManager.previewLanes`,
 * from `footprintLanes`). A zone's lanes go round-robin among its machines; a lane takes its
 * length over the class's assumed speed, a turn its assumed turn time, and the moves between
 * lanes and zones are straight. Every 5 m along a lane the ground is read: the slope from an
 * elevation grid, split into the grade along the lane and the cross-slope across it, and the
 * distance to mapped creeks. Where a lane goes past a limit its class is assumed to keep
 * (`maxSlopePct` and `creekBufferM` in apps/api/app/services/machine_classes.json, the one
 * table the planner also sizes work from) a hand is raised at the lane's worst point, one per
 * lane and rule, naming the rule, the value, and where both came from.
 *
 * Pure: no Cesium and no fetching. `features/mission/rehearsalGround.ts` gathers the ground and
 * `cesium/RehearsalManager.ts` plays the result. Nothing here is measured on a machine: the
 * limits are assumptions, the lanes a preview, and the lines it writes say so.
 */

import type { MachineClassId } from "@twin/contracts";
import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";

import { MACHINE_CLASSES } from "./machineClasses";

/** How often the ground is read along a lane, metres. */
export const SAMPLE_M = 5;
/** The two speeds a rehearsal plays at: simulated seconds per real second. */
export const REHEARSAL_SPEEDS = [300, 600] as const;
export type RehearsalSpeed = (typeof REHEARSAL_SPEEDS)[number];

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
}

export interface CreekLine {
  id: string;
  name: string | null;
  /** NHD permanence: perennial, intermittent, ephemeral, unclassified, artificial-path, other. */
  kind: string;
  coordinates: LonLat[];
}

export interface CreekSource {
  lines: CreekLine[];
  /** e.g. "USGS NHD flowlines, 1:24,000". */
  label: string;
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

export interface Rehearsal {
  machines: RehearsalMachine[];
  tracks: RehearsalTrack[];
  lanes: RehearsedLane[];
  hands: RaisedHand[];
  durationS: number;
  lengthM: number;
  summary: string[];
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

function lerp(a: LonLat, b: LonLat, f: number): LonLat {
  return {
    longitude: a.longitude + (b.longitude - a.longitude) * f,
    latitude: a.latitude + (b.latitude - a.latitude) * f,
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

function segmentDistance(p: XY, a: XY, b: XY): number {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const length2 = vx * vx + vy * vy;
  const t =
    length2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / length2)) : 0;
  return Math.hypot(p.x - (a.x + t * vx), p.y - (a.y + t * vy));
}

interface CreekPiece {
  line: CreekLine;
  a: XY;
  b: XY;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
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
      const a = toXY(frame, p);
      const b = toXY(frame, q);
      pieces.push({
        line,
        a,
        b,
        minX: Math.min(a.x, b.x),
        maxX: Math.max(a.x, b.x),
        minY: Math.min(a.y, b.y),
        maxY: Math.max(a.y, b.y),
      });
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
}

/**
 * Plays the plan through on paper: who drives which lane when, and where a hand goes up.
 * Deterministic for the same input.
 */
export function planRehearsal(input: RehearsalInput): Rehearsal {
  const sampleM = input.sampleM ?? SAMPLE_M;
  const { machines, crews, notes } = castMachines(input.zones, input.machines);
  const allPoints = input.zones.flatMap((z) => z.lanes.flatMap((l) => l.points));
  const origin = allPoints.length
    ? {
        longitude: allPoints.reduce((s, p) => s + p.longitude, 0) / allPoints.length,
        latitude: allPoints.reduce((s, p) => s + p.latitude, 0) / allPoints.length,
      }
    : { longitude: 0, latitude: 0 };
  const frame = frameAt(origin);
  const pieces = input.creeks
    ? creekPieces(frame, input.creeks, input.includeEphemeral ?? false)
    : [];
  const maxBuffer = Math.max(0, ...machines.map((m) => m.capability.creekBufferM ?? 0));

  const clocks = new Map<string, { t: number; at: LonLat | null; zoneId: string | null }>();
  const segments = new Map<string, RehearsalSegment[]>();
  for (const machine of machines) {
    clocks.set(machine.id, { t: 0, at: null, zoneId: null });
    segments.set(machine.id, []);
  }
  const lanes: RehearsedLane[] = [];
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
      for (let i = 0; i + 1 < lane.points.length; i++) {
        const a = lane.points[i];
        const b = lane.points[i + 1];
        if (!a || !b) continue;
        const pieceM = distanceM(frame, a, b);
        if (!(pieceM > 0)) continue;
        const t0 = laneT0 + lengthM / capability.speedMps;
        const t1 = t0 + pieceM / capability.speedMps;
        track.push({ t0, t1, from: a, to: b, lane: { zoneId: zone.zoneId, index } });
        if (ground) {
          const pa = toXY(frame, a);
          const pb = toXY(frame, b);
          const ux = (pb.x - pa.x) / pieceM;
          const uy = (pb.y - pa.y) / pieceM;
          const buffer = capability.creekBufferM ?? 0;
          const near =
            buffer > 0
              ? pieces.filter(
                  (c) =>
                    c.maxX >= Math.min(pa.x, pb.x) - buffer &&
                    c.minX <= Math.max(pa.x, pb.x) + buffer &&
                    c.maxY >= Math.min(pa.y, pb.y) - buffer &&
                    c.minY <= Math.max(pa.y, pb.y) + buffer,
                )
              : [];
          const steps = Math.max(1, Math.ceil(pieceM / sampleM));
          for (let k = 0; k <= steps; k++) {
            // A shared corner is read once, at the start of the next piece.
            if (k === steps && i + 2 < lane.points.length) continue;
            const f = k / steps;
            const at = lerp(a, b, f);
            const t = t0 + (t1 - t0) * f;
            if (input.elevation && capability.maxSlopePct !== null) {
              const g = gradientAt(input.elevation.grid, at.longitude, at.latitude);
              if (g) {
                const grade = Math.abs(g.east * ux + g.north * uy) * 100;
                const cross = Math.abs(-g.east * uy + g.north * ux) * 100;
                for (const [kind, value] of [
                  ["grade", grade],
                  ["cross-slope", cross],
                ] as const) {
                  if (value <= capability.maxSlopePct) continue;
                  const current = worst.get(kind);
                  if (!current || value > current.value) worst.set(kind, { value, at, t });
                }
              }
            }
            if (near.length && capability.creekBufferM !== null) {
              const p = { x: pa.x + (pb.x - pa.x) * f, y: pa.y + (pb.y - pa.y) * f };
              for (const piece of near) {
                const d = segmentDistance(p, piece.a, piece.b);
                if (d >= capability.creekBufferM) continue;
                const current = worst.get("creek-buffer");
                if (!current || d < current.value)
                  worst.set("creek-buffer", { value: d, at, t, creek: piece.line });
              }
            }
          }
        }
        lengthM += pieceM;
      }
      const laneT1 = laneT0 + lengthM / capability.speedMps;
      const end = lane.points[lane.points.length - 1] ?? start;
      clock.t = laneT1;
      clock.at = end;
      clock.zoneId = zone.zoneId;
      const number = lane.number ?? index + 1;
      lanes.push({
        zoneId: zone.zoneId,
        index,
        number,
        machineId,
        points: lane.points,
        lengthM,
        pass: lane.pass,
        t0: laneT0,
        t1: laneT1,
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
            creeks: input.creeks,
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

  const groundMachines = tracks.filter((t) => t.machine.capability.kind === "ground");
  if (groundMachines.length > 0) {
    if (!input.elevation)
      skipped.add(input.elevationMissing ?? "Slope not checked: no terrain for this ground.");
    if (!input.creeks)
      skipped.add(
        input.creeksMissing ?? "Creek buffer not checked: no mapped creeks for this ground.",
      );
    else if (!input.includeEphemeral && input.creeks.lines.some((l) => l.kind === "ephemeral"))
      skipped.add(
        "Ephemeral drainages (NHD channels that run only after rain) are not counted as creeks.",
      );
  }
  skipped.add(
    "The ghosts drive the lanes the map draws (preview spacing), not every swath, so the ground between drawn lanes is not checked; turns and moves between lanes are straight and not checked.",
  );

  const summary: string[] = [];
  summary.push(
    `${tracks.length} simulated machine${tracks.length === 1 ? "" : "s"} drove ${lanes.length} drawn lane${lanes.length === 1 ? "" : "s"} (${km(lengthM)}) in ${formatSimDuration(durationS)} at assumed speeds.`,
  );
  const checked = lanes.filter(
    (lane) => machines.find((m) => m.id === lane.machineId)?.capability.kind === "ground",
  ).length;
  summary.push(...handLines(hands, checked));
  if (hands.length === 0 && groundMachines.length > 0 && (input.elevation || input.creeks))
    summary.push("No hands raised: every lane stayed inside the assumed limits.");

  const sources: string[] = [];
  if (input.elevation && groundMachines.length > 0)
    sources.push(
      `Slope: ${input.elevation.label}${input.elevation.ref ? `, ${input.elevation.ref}` : ""}`,
    );
  const checkedCreeks = input.creeks && groundMachines.length > 0 && maxBuffer > 0;
  if (checkedCreeks && input.creeks) sources.push(`Creek: ${input.creeks.label}`);
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
      checkedCreeks && input.creeks
        ? input.creeks.lines.filter((line) =>
            countsAsCreek(line.kind, input.includeEphemeral ?? false),
          )
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
  if (kind === "creek-buffer") {
    const limit = capability.creekBufferM ?? 0;
    const creek = found.creek;
    const name = creek?.name ?? "a mapped creek";
    return {
      ...base,
      value: found.value,
      limit,
      label: `${Math.round(found.value)} m from ${creek?.name ?? "a creek"} · buffer ${limit} m (assumed)`,
      rule: `${where} comes within ${Math.round(found.value)} m of ${name}. A ${capability.label} is assumed to keep ${limit} m from creeks.`,
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
    rule: `${where} reaches ${pct(found.value)} ${what === "grade" ? "grade along the lane" : "cross-slope across the lane"} at its steepest. A ${capability.label} is assumed to work up to ${pct(limit)}.`,
    source: `Slope: ${context.elevation?.label ?? "elevation"}${context.elevation?.ref ? `, ${context.elevation.ref}` : ""}`,
  };
}

/** One summary line per rule that raised hands: how many of the checked lanes, and which. */
function handLines(hands: readonly RaisedHand[], checked: number): string[] {
  const lines: string[] = [];
  const of = (n: number) =>
    checked > n ? `${n} of ${checked} lanes` : `${n} lane${n === 1 ? "" : "s"}`;
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
    lines.push(`${of(count(list))} ${over} (${refs(list)})`);
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
    lines.push(`${of(count(creek))} inside ${buffer} of ${creeks} (${refs(creek)})`);
  }
  return lines;
}

/** Where a ghost is at `t`, and the lane it is on (null between lanes or when done). */
export function positionAt(
  track: RehearsalTrack,
  t: number,
): { at: LonLat; lane: LaneRef | null } | null {
  const segments = track.segments;
  const first = segments[0];
  const last = segments[segments.length - 1];
  if (!first || !last) return null;
  if (t <= first.t0) return { at: first.from, lane: null };
  if (t >= last.t1) return { at: last.to, lane: null };
  let lo = 0;
  let hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((segments[mid]?.t1 ?? 0) < t) lo = mid + 1;
    else hi = mid;
  }
  const segment = segments[lo] ?? last;
  const span = segment.t1 - segment.t0;
  const f = span > 0 ? Math.min(1, Math.max(0, (t - segment.t0) / span)) : 1;
  return { at: lerp(segment.from, segment.to, f), lane: segment.lane };
}
