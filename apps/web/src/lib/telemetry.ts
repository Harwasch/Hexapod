/**
 * Telemetry bindings (hexapod.telemetry v1, docs/SCENE_OBJECTS.md §4 "Telemetry on
 * instances"): which live pose stream moves which instance of a scan, written beside the tiles
 * as `telemetry.json` and declared on their root as `extras.telemetry = { uri, count }` -- the
 * pattern of `instances`, `skin` and `materials`.
 *
 * A separate file, not part of `instances.json` or `skin.json`: those are rewritten by the
 * segmentation and skinning steps, while a binding is deployment data (which robot is which
 * object) that must survive a refit, as `materials.json` does. It is keyed by instance id, so a
 * re-segmentation that renumbers instances needs the bindings re-pointed.
 *
 * Also the wire format of a reading (`parsePoseMessages`): what an SSE or WebSocket source
 * sends, and what a robot or fleet bridge publishes.
 */

import {
  DEFAULT_TRACK_OPTIONS,
  ecefToGeodetic,
  enuToEcefRotation,
  POSE_FRAMES,
  QUAT_IDENTITY,
  quatFromHeadingPitchRoll,
  quatMultiply,
  quatNormalize,
  type PoseFrame,
  type PoseSample,
  type Quat,
  type StalePolicy,
  type SyntheticPath,
  type TrackOptions,
  type Vec3,
} from "@twin/world";

import { resolveBeside } from "./instances";

export const TELEMETRY_FORMAT = "hexapod.telemetry";
export const TELEMETRY_VERSION = 1;

/** `root.extras.telemetry`. */
export interface TelemetryRef {
  uri: string;
  count: number;
}

/** A deterministic path sampled on a clock (dev, e2e, demos). */
export interface SyntheticSourceConfig {
  readonly kind: "synthetic";
  readonly frame: PoseFrame;
  readonly path: SyntheticPath;
  readonly rateHz: number;
  readonly delayMs: number;
  readonly jitterMs: number;
  readonly dropouts: readonly (readonly [number, number])[];
  /** The path's t = 0 in source time (ms); omitted, the clock's 0. */
  readonly startMs?: number;
}

/** Readings pushed by a server: Server-Sent Events or a WebSocket, JSON messages. */
export interface StreamSourceConfig {
  readonly kind: "sse" | "websocket";
  readonly frame: PoseFrame;
  readonly url: string;
}

export type TelemetrySourceConfig = SyntheticSourceConfig | StreamSourceConfig;

/** The body frame's pose at capture, in `frame`. */
export interface RestPose {
  readonly frame: PoseFrame;
  readonly position: Vec3;
  readonly orientation: Quat;
}

export interface TelemetryBinding extends TrackOptions {
  /** The instances.json id it moves (with everything below it). */
  readonly instance: number;
  /** A key of `sources`. */
  readonly source: string;
  /** Only readings whose `stream` is this (a source carrying several bodies). */
  readonly stream?: string;
  /** Omitted: the instance's base centre (bounds), level, in the scan frame. */
  readonly rest?: RestPose;
}

export interface TelemetryDoc {
  readonly sources: ReadonlyMap<string, TelemetrySourceConfig>;
  readonly bindings: readonly TelemetryBinding[];
  /** Problems found while reading; what was well formed is kept. */
  readonly issues: string[];
}

/** `extras.telemetry` off a tileset's root tile; null when absent or malformed. */
export function telemetryRefOf(extras: unknown): TelemetryRef | null {
  const ref = (extras as { telemetry?: Record<string, unknown> } | null | undefined)?.telemetry;
  if (typeof ref?.uri !== "string" || ref.uri.length === 0) return null;
  const count = typeof ref.count === "number" && Number.isFinite(ref.count) ? ref.count : 0;
  return { uri: ref.uri, count: Math.max(0, Math.round(count)) };
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function vec3(value: unknown): Vec3 | null {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const [x, y, z] = value as unknown[];
  return finite(x) && finite(y) && finite(z) ? [x, y, z] : null;
}

function quat(value: unknown): Quat | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const [x, y, z, w] = value as unknown[];
  if (!(finite(x) && finite(y) && finite(z) && finite(w))) return null;
  if (!(Math.hypot(x, y, z, w) > 1e-9)) return null;
  return quatNormalize([x, y, z, w]);
}

function frameOf(value: unknown, fallback: PoseFrame): PoseFrame | null {
  if (value === undefined) return fallback;
  return POSE_FRAMES.includes(value as PoseFrame) ? (value as PoseFrame) : null;
}

/**
 * An orientation from `orientation` (`[x, y, z, w]`, body → frame) or, failing that,
 * `headingDeg` / `pitchDeg` / `rollDeg` in the local level frame (the scan frame for `scan`,
 * east-north-up at the position otherwise; see `poseToScan`). Neither: level, facing east.
 */
function orientationOf(r: Record<string, unknown>): Quat | null {
  if (r.orientation !== undefined) return quat(r.orientation);
  if (r.headingDeg !== undefined) {
    if (!finite(r.headingDeg)) return null;
    const pitch = finite(r.pitchDeg) ? r.pitchDeg : 0;
    const roll = finite(r.rollDeg) ? r.rollDeg : 0;
    return quatFromHeadingPitchRoll(r.headingDeg, pitch, roll);
  }
  return QUAT_IDENTITY;
}

/**
 * A heading-given orientation in the ECEF frame: the heading is in the level frame at the
 * position (east-north-up), turned into ECEF. Everything else is already in its frame.
 */
function levelToFrame(
  frame: PoseFrame,
  position: Vec3,
  orientation: Quat,
  r: Record<string, unknown>,
): Quat {
  if (frame !== "ecef" || r.orientation !== undefined) return orientation;
  const [lon, lat] = ecefToGeodetic(position);
  return quatMultiply(enuToEcefRotation(lon, lat), orientation);
}

function pathOf(raw: unknown): SyntheticPath | null {
  const r = raw as Record<string, unknown> | null | undefined;
  if (r?.kind === "circle") {
    const centre = vec3(r.centre);
    if (!centre || !finite(r.radius) || !finite(r.periodS) || r.periodS <= 0) return null;
    return {
      kind: "circle",
      centre,
      radius: r.radius,
      periodS: r.periodS,
      ...(finite(r.phaseDeg) ? { phaseDeg: r.phaseDeg } : {}),
      ...(typeof r.clockwise === "boolean" ? { clockwise: r.clockwise } : {}),
    };
  }
  if (r?.kind === "polyline") {
    if (!Array.isArray(r.points) || !finite(r.speedMps) || r.speedMps <= 0) return null;
    const points = (r.points as unknown[]).map(vec3);
    if (points.length < 2 || points.some((p) => p === null)) return null;
    return {
      kind: "polyline",
      points: points as Vec3[],
      speedMps: r.speedMps,
      ...(typeof r.closed === "boolean" ? { closed: r.closed } : {}),
    };
  }
  return null;
}

function sourceOf(raw: unknown): TelemetrySourceConfig | string {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return "not an object";
  const frame = frameOf(r.frame, "scan");
  if (frame === null) return `unknown frame ${JSON.stringify(r.frame)}`;
  if (r.kind === "synthetic") {
    const path = pathOf(r.path);
    if (!path) return "synthetic: bad path";
    const rateHz = finite(r.rateHz) && r.rateHz > 0 ? r.rateHz : 2;
    const dropouts: [number, number][] = [];
    if (Array.isArray(r.dropouts)) {
      for (const d of r.dropouts as unknown[]) {
        if (Array.isArray(d) && d.length === 2 && finite(d[0]) && finite(d[1]) && d[1] > d[0])
          dropouts.push([d[0], d[1]]);
      }
    }
    return {
      kind: "synthetic",
      frame,
      path,
      rateHz,
      delayMs: finite(r.delayMs) && r.delayMs >= 0 ? r.delayMs : 0,
      jitterMs: finite(r.jitterMs) && r.jitterMs >= 0 ? r.jitterMs : 0,
      dropouts,
      ...(finite(r.startMs) ? { startMs: r.startMs } : {}),
    };
  }
  if (r.kind === "sse" || r.kind === "websocket") {
    if (typeof r.url !== "string" || r.url.length === 0) return `${r.kind}: no url`;
    return { kind: r.kind, frame, url: r.url };
  }
  return `unknown source kind ${JSON.stringify(r.kind)}`;
}

function nonNegative(value: unknown, fallback: number): number {
  return finite(value) && value >= 0 ? value : fallback;
}

function bindingOf(raw: unknown, sources: ReadonlyMap<string, unknown>): TelemetryBinding | string {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return "not an object";
  const instance = r.instance;
  if (!Number.isInteger(instance) || (instance as number) < 1) return "no instance id";
  if (typeof r.source !== "string" || !sources.has(r.source))
    return `instance ${String(instance)}: unknown source ${JSON.stringify(r.source)}`;
  let rest: RestPose | undefined;
  if (r.rest !== undefined) {
    const rr = r.rest as Record<string, unknown> | null;
    const frame = frameOf(rr?.frame, "scan");
    const position = vec3(rr?.position);
    const orientation = rr ? orientationOf(rr) : null;
    if (!rr || frame === null || !position || !orientation)
      return `instance ${String(instance)}: bad rest pose`;
    rest = { frame, position, orientation: levelToFrame(frame, position, orientation, rr) };
  }
  const stale: StalePolicy = r.stale === "freeze" ? "freeze" : DEFAULT_TRACK_OPTIONS.stale;
  const d = DEFAULT_TRACK_OPTIONS;
  return {
    instance: instance as number,
    source: r.source,
    ...(typeof r.stream === "string" ? { stream: r.stream } : {}),
    ...(rest ? { rest } : {}),
    latencyMs: nonNegative(r.latencyMs, d.latencyMs),
    extrapolateMs: nonNegative(r.extrapolateMs, d.extrapolateMs),
    staleMs: nonNegative(r.staleMs, d.staleMs),
    stale,
    fadeMs: nonNegative(r.fadeMs, d.fadeMs),
  };
}

/**
 * Reads a `telemetry.json` document: null when it is not one, otherwise every well-formed
 * source and binding (one binding per instance: later ones are dropped), with the rest noted
 * in `issues`.
 */
export function parseTelemetry(raw: unknown): TelemetryDoc | null {
  const doc = raw as Record<string, unknown> | null | undefined;
  if (doc?.format !== TELEMETRY_FORMAT || doc.version !== TELEMETRY_VERSION) return null;
  const issues: string[] = [];
  const sources = new Map<string, TelemetrySourceConfig>();
  const rawSources = doc.sources;
  if (typeof rawSources === "object" && rawSources !== null && !Array.isArray(rawSources)) {
    for (const [id, value] of Object.entries(rawSources as Record<string, unknown>)) {
      const source = sourceOf(value);
      if (typeof source === "string") issues.push(`source ${id}: ${source}`);
      else sources.set(id, source);
    }
  }
  const bindings: TelemetryBinding[] = [];
  const seen = new Set<number>();
  for (const value of Array.isArray(doc.bindings) ? (doc.bindings as unknown[]) : []) {
    const binding = bindingOf(value, sources);
    if (typeof binding === "string") {
      issues.push(`binding: ${binding}`);
      continue;
    }
    if (seen.has(binding.instance)) {
      issues.push(`binding: instance ${String(binding.instance)} is bound twice; the first wins`);
      continue;
    }
    seen.add(binding.instance);
    bindings.push(binding);
  }
  return { sources, bindings, issues };
}

/** Fetches and reads a scan's `telemetry.json`. Throws when it is missing or not one. */
export async function loadTelemetry(tilesetUrl: string, ref: TelemetryRef): Promise<TelemetryDoc> {
  const response = await fetch(resolveBeside(tilesetUrl, ref.uri));
  if (!response.ok) throw new Error(`telemetry answered ${String(response.status)}`);
  const doc = parseTelemetry(await response.json());
  if (!doc) throw new Error("telemetry: not a hexapod.telemetry v1 document");
  return doc;
}

// ---- The wire format ---------------------------------------------------------------------

/** One reading as a source hands it over, with the stream it belongs to (if any). */
export interface StreamSample extends PoseSample {
  readonly stream?: string;
}

/**
 * Readings from one message: a reading, an array of them, or `{ "samples": [...] }`. A reading
 * is `{ t, position, orientation?, headingDeg?, pitchDeg?, rollDeg?, frame?, stream? }`: `t` in
 * milliseconds (epoch, the source's clock) or an ISO 8601 string; `position` in `frame`
 * (`defaultFrame` when absent). Malformed readings are skipped.
 */
export function parsePoseMessages(raw: unknown, defaultFrame: PoseFrame): StreamSample[] {
  const list: unknown[] = Array.isArray(raw)
    ? raw
    : Array.isArray((raw as { samples?: unknown } | null)?.samples)
      ? (raw as { samples: unknown[] }).samples
      : [raw];
  const out: StreamSample[] = [];
  for (const item of list) {
    const r = item as Record<string, unknown> | null;
    if (typeof r !== "object" || r === null) continue;
    const t = finite(r.t) ? r.t : typeof r.t === "string" ? Date.parse(r.t) : NaN;
    if (!Number.isFinite(t)) continue;
    const frame = frameOf(r.frame, defaultFrame);
    const position = vec3(r.position);
    const orientation = orientationOf(r);
    if (frame === null || !position || !orientation) continue;
    out.push({
      t,
      frame,
      position,
      orientation: levelToFrame(frame, position, orientation, r),
      ...(typeof r.stream === "string" ? { stream: r.stream } : {}),
    });
  }
  return out;
}
