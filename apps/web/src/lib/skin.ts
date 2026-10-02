/**
 * Scene-object skins (hexapod.skin v1, docs/SCENE_OBJECTS.md §4): per splat, signed weights on
 * a few handles, written by `tools/captures/skin_scene.py` beside the measured tiles and
 * declared on their root as `extras.skin = { uri, count }`.
 *
 * A skinned splat moves by
 *
 *     x' = x + Σ_j w_j(x) · Z_j · [x − origin; 1]
 *
 * with `Z_j` a 3×4 affine per handle (row-major, in the skin's rest frame: the tileset's local
 * ENU axes, origin at `origin`). Handle 0 is the constant field (`w_0 ≡ 1`, not stored): with
 * `Z_0 = [R − I | t]` the object moves rigidly, exactly. Handles `1..m−1` are the object's
 * smallest elastic eigenmodes (Simplicits/FreeForm), each scaled to `max |w| = 1`. Weights are
 * signed and never normalised.
 *
 * `skin.json`'s `tiles` maps each tile checksum to run-length `[skin, count, ...]` pairs in the
 * tile's own gaussian order (0: no skin; `tileRuns.ts`) and `row`, where the tile's skinned
 * splats' rows start in `skin.bin`: 16 bytes a skinned splat, byte `k` the int8 weight of
 * handle `k + 1` (times `weights.scale`), in the tile's order.
 */

import { decodeRuns, runsLength, tileRunsIssue } from "@twin/world";

import { resolveBeside, type Vec3 } from "./instances";

export const SKIN_FORMAT = "hexapod.skin";
export const SKIN_VERSION = 1;
/** Bytes of one splat's weight row: one RGBA32UI texel. */
export const SKIN_ROW_BYTES = 16;
/** Handles a skin may have, the constant one included. */
export const MAX_SKIN_HANDLES = 16;

/** `root.extras.skin`. */
export interface SkinRef {
  uri: string;
  count: number;
}

export interface SkinSupport {
  /** Where the handle acts: the |w|-weighted centre, in the rest frame (from `origin`). */
  centre: Vec3;
  /** And how far around it, metres (|w|-weighted rms). */
  radius: number;
}

export interface SkinEntry {
  /** 1-based; 0 means none. */
  id: number;
  /** The instance (`instances.json`) this skin moves, with everything below it. */
  instance: number;
  /** Handles, the constant one included (2..16). */
  handles: number;
  /** The rest frame's origin, tileset local ENU metres: the object's base. */
  origin: Vec3;
  /** Half the object's largest extent, metres. */
  scale: number;
  /** Per learned handle (1..m−1): its eigenvalue (stiffness, unit box, uniform material). */
  eigenvalues: number[];
  /** Per learned handle (1..m−1). */
  support: SkinSupport[];
}

export interface SkinTile {
  runs: Int32Array;
  /** The first row of `skin.bin` the tile's skinned splats take. */
  row: number;
}

export interface SkinDoc {
  skins: SkinEntry[];
  byId: ReadonlyMap<number, SkinEntry>;
  byInstance: ReadonlyMap<number, SkinEntry>;
  maxId: number;
  tiles: ReadonlyMap<string, SkinTile>;
  /** `skin.bin` as words: four a row. */
  words: Uint32Array;
  rows: number;
  /** A stored byte times this is the weight. */
  scale: number;
  issues: string[];
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function vec3(value: unknown): Vec3 | null {
  if (!Array.isArray(value) || value.length !== 3) return null;
  const [x, y, z] = value as unknown[];
  if (![x, y, z].every((v) => typeof v === "number" && Number.isFinite(v))) return null;
  return [x as number, y as number, z as number];
}

/** `extras.skin` off a tileset's root tile; null when absent or malformed. */
export function skinRefOf(extras: unknown): SkinRef | null {
  const ref = (extras as { skin?: Record<string, unknown> } | null | undefined)?.skin;
  if (typeof ref?.uri !== "string" || ref.uri.length === 0) return null;
  return { uri: ref.uri, count: Math.max(0, Math.round(finite(ref.count))) };
}

function skinOf(raw: unknown): SkinEntry | null {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return null;
  const { id, instance, handles } = r;
  if (!Number.isInteger(id) || (id as number) < 1) return null;
  if (!Number.isInteger(instance) || (instance as number) < 1) return null;
  if (!Number.isInteger(handles) || (handles as number) < 1) return null;
  if ((handles as number) > MAX_SKIN_HANDLES) return null;
  const origin = vec3(r.origin);
  if (!origin) return null;
  const eigenvalues = Array.isArray(r.eigenvalues)
    ? (r.eigenvalues as unknown[]).map((v) => finite(v))
    : [];
  const support: SkinSupport[] = Array.isArray(r.support)
    ? (r.support as { centre?: unknown; radius?: unknown }[]).map((s) => ({
        centre: vec3(s?.centre) ?? [0, 0, 0],
        radius: finite(s?.radius),
      }))
    : [];
  return {
    id: id as number,
    instance: instance as number,
    handles: handles as number,
    origin,
    scale: finite(r.scale, 1),
    eigenvalues,
    support,
  };
}

/**
 * Reads a `skin.json` document and its `skin.bin`. Null when it is not one; otherwise every
 * well-formed skin and tile, with what was skipped in `issues`. A tile whose rows run past the
 * blob is dropped (its splats stay still).
 */
export function parseSkin(raw: unknown, blob: ArrayBuffer): SkinDoc | null {
  const doc = raw as Record<string, unknown> | null | undefined;
  if (doc?.format !== SKIN_FORMAT || doc.version !== SKIN_VERSION) return null;
  if (!Array.isArray(doc.skins)) return null;
  const weights = doc.weights as Record<string, unknown> | undefined;
  if (weights?.dtype !== "int8" || weights.rowBytes !== SKIN_ROW_BYTES) return null;
  const issues: string[] = [];
  const skins: SkinEntry[] = [];
  const byId = new Map<number, SkinEntry>();
  const byInstance = new Map<number, SkinEntry>();
  for (const entry of doc.skins as unknown[]) {
    const skin = skinOf(entry);
    if (!skin) {
      issues.push("a skin without an id, instance, handles (1..16) and origin was skipped");
      continue;
    }
    if (byId.has(skin.id)) {
      issues.push(`skin ${String(skin.id)} is listed twice; the first is kept`);
      continue;
    }
    byId.set(skin.id, skin);
    if (!byInstance.has(skin.instance)) byInstance.set(skin.instance, skin);
    skins.push(skin);
  }
  let maxId = 0;
  for (const skin of skins) maxId = Math.max(maxId, skin.id);
  const rows = Math.floor(blob.byteLength / SKIN_ROW_BYTES);
  const words = new Uint32Array(blob, 0, rows * 4);
  const tiles = new Map<string, SkinTile>();
  const entries = doc.tiles;
  if (typeof entries === "object" && entries !== null && !Array.isArray(entries)) {
    for (const [key, value] of Object.entries(entries as Record<string, unknown>)) {
      const v = value as { skins?: unknown; row?: unknown } | null;
      const issue = tileRunsIssue(key, v?.skins, maxId, "a listed skin");
      if (issue !== undefined) {
        issues.push(issue);
        continue;
      }
      const runs = Int32Array.from(v?.skins as number[]);
      const row = v?.row;
      let skinned = 0;
      for (let i = 0; i < runs.length; i += 2) if ((runs[i] ?? 0) > 0) skinned += runs[i + 1] ?? 0;
      if (!Number.isInteger(row) || (row as number) < 0 || (row as number) + skinned > rows) {
        issues.push(`${key}: its rows are not in skin.bin`);
        continue;
      }
      tiles.set(key, { runs, row: row as number });
    }
  } else {
    issues.push("tiles must be an object keyed by tile checksum");
  }
  return {
    skins,
    byId,
    byInstance,
    maxId,
    tiles,
    words,
    rows,
    scale: finite(weights.scale, 1 / 127),
    issues,
  };
}

/** One tile's skins and weight rows, per gaussian in its own order. */
export interface TileSkin {
  /** The skin id of every gaussian (0: none). */
  skins: Uint32Array;
  /** Four words a gaussian: its row of `skin.bin`, zeros where it has no skin. */
  words: Uint32Array;
}

/** The skins and rows of the tile whose checksum is `checksum`, or undefined. */
export function tileSkin(doc: SkinDoc, checksum: string): TileSkin | undefined {
  const tile = doc.tiles.get(checksum);
  if (tile === undefined) return undefined;
  const count = runsLength(tile.runs);
  const skins = decodeRuns(tile.runs, new Uint32Array(count));
  const words = new Uint32Array(count * 4);
  let row = tile.row;
  for (let i = 0; i < count; i += 1) {
    if (skins[i] === 0) continue;
    words.set(doc.words.subarray(row * 4, row * 4 + 4), i * 4);
    row += 1;
  }
  return { skins, words };
}

/** Weight `k` (handle `k + 1`) of a splat's row, as the shader decodes it. */
export function rowWeight(words: ArrayLike<number>, at: number, k: number, scale: number): number {
  const word = words[at * 4 + (k >> 2)] ?? 0;
  const byte = (word >>> (8 * (k & 3))) & 0xff;
  return (byte >= 128 ? byte - 256 : byte) * scale;
}

/** Fetches and reads a scan's `skin.json` and its weights. Throws when either is missing. */
export async function loadSkin(tilesetUrl: string, ref: SkinRef): Promise<SkinDoc> {
  const url = resolveBeside(tilesetUrl, ref.uri);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`skin answered ${String(response.status)}`);
  const raw = (await response.json()) as { weights?: { file?: unknown } };
  const file = typeof raw.weights?.file === "string" ? raw.weights.file : "skin.bin";
  const binary = await fetch(resolveBeside(url, file));
  if (!binary.ok) throw new Error(`skin weights answered ${String(binary.status)}`);
  const doc = parseSkin(raw, await binary.arrayBuffer());
  if (!doc) throw new Error("skin: not a hexapod.skin v1 document");
  return doc;
}

// ---- Handles -----------------------------------------------------------------------------

/** Twelve numbers a handle: `Z_j` row-major, 3×4. */
export const HANDLE_FLOATS = 12;

/** Handles all at rest, for a skin of `handles`. */
export function restHandles(handles: number): Float64Array {
  return new Float64Array(handles * HANDLE_FLOATS);
}

/**
 * `Z = [R − I | t]` for the unit quaternion `(x, y, z, w)` and translation `t`: a rigid motion
 * of the whole object when it is the constant handle's. `R − I` is written from the quaternion
 * so a small rotation keeps its precision.
 */
export function rigidHandle(
  rotation: readonly [number, number, number, number],
  translation: Vec3,
  out: Float64Array = new Float64Array(HANDLE_FLOATS),
  at = 0,
): Float64Array {
  const [x, y, z, w] = rotation;
  const m = [
    -2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    -2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    -2 * (x * x + y * y),
  ];
  for (let r = 0; r < 3; r += 1) {
    out[at + r * 4] = m[r * 3] ?? 0;
    out[at + r * 4 + 1] = m[r * 3 + 1] ?? 0;
    out[at + r * 4 + 2] = m[r * 3 + 2] ?? 0;
    out[at + r * 4 + 3] = translation[r] ?? 0;
  }
  return out;
}

/**
 * The displacement `Σ_j w_j Z_j [x − origin; 1]` of one point in the rest frame: the format's
 * arithmetic, in float64 (`learned[k]` is handle `k + 1`'s weight).
 */
export function skinDisplacement(
  handles: ArrayLike<number>,
  count: number,
  origin: Vec3,
  learned: ArrayLike<number>,
  position: Vec3,
): [number, number, number] {
  const lx = position[0] - origin[0];
  const ly = position[1] - origin[1];
  const lz = position[2] - origin[2];
  const out: [number, number, number] = [0, 0, 0];
  for (let j = 0; j < count; j += 1) {
    const w = j === 0 ? 1 : (learned[j - 1] ?? 0);
    if (w === 0) continue;
    const o = j * HANDLE_FLOATS;
    for (let r = 0; r < 3; r += 1) {
      out[r] =
        (out[r] ?? 0) +
        w *
          ((handles[o + r * 4] ?? 0) * lx +
            (handles[o + r * 4 + 1] ?? 0) * ly +
            (handles[o + r * 4 + 2] ?? 0) * lz +
            (handles[o + r * 4 + 3] ?? 0));
    }
  }
  return out;
}
