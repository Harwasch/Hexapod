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
 * handle `k + 1` (times `weights.scale`), in the tile's order. A file whose `weights.rowBytes`
 * is 32 (a skin of more than 16 handles: a big tree, docs/SCENE_OBJECTS.md §9) takes two
 * texels a splat. A skin of one handle (a rigid object: only the constant handle) takes no
 * rows at all.
 *
 * A **limbs skin** (`method.name` `limbs`, docs/SCENE_OBJECTS.md §9) is a plant's rig carried
 * in this format: handle `j` is limb `j`, each skin entry carries a `limbs` block (the plant's
 * wind and one record per limb, `@twin/world`'s `LimbSkinSource`) for its driver
 * (`limbWind.ts`), and the last byte of every row -- which no weight uses, a row of `b` bytes
 * holding at most `b − 1` -- is the splat's leaf flutter share. Its driven handles carry the
 * frame's flutter after the `12·m` numbers (`skinFloats`).
 */

import {
  decodeRuns,
  LIMB_FLUTTER_FLOATS,
  runsLength,
  tileRunsIssue,
  type LimbHandle,
  type LimbSkinSource,
} from "@twin/world";

import { resolveBeside, type Vec3 } from "./instances";
import { jsonBytes, scanPayloads } from "./payloadCache";

export const SKIN_FORMAT = "hexapod.skin";
export const SKIN_VERSION = 1;
/** Bytes of one splat's weight row: one RGBA32UI texel. */
export const SKIN_ROW_BYTES = 16;
/** ... or two texels, in a file with a skin of more than 16 handles. */
export const SKIN_WIDE_ROW_BYTES = 32;
/** Handles a skin may have, the constant one included (16 in a file of one-texel rows). */
export const MAX_SKIN_HANDLES = 32;

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
  /** What a modal driver needs (`dynamics`), when the file carries it. */
  dynamics?: SkinDynamics;
  /**
   * What the skin carries of the object it moves (a variant's skin, skin_variants.py): its
   * name, behaviour and property scores, for a viewer whose `instances.json` does not list it
   * (a scan with none: the wind's prior and the poke read these).
   */
  traits?: SkinTraits;
  /** The handle policy's stiffness class (`rigid`, `firm`, `plant`, `tree`), when recorded. */
  stiffnessClass?: string;
  /**
   * A limbs skin's own driver data (`skin.json`'s `limbs`, docs/SCENE_OBJECTS.md §9): its
   * handles are a plant's limbs, swayed by `@twin/world`'s `limbWind.ts`, not eigenmodes. Only
   * in a document whose `method.name` is `limbs`, and only when the block is whole.
   */
  limbs?: LimbSkinSource;
}

/** `method.name` of a limbs skin's document. */
export const LIMBS_METHOD = "limbs";

export interface SkinTraits {
  label?: string;
  category?: string;
  behaviour?: string;
  properties?: Record<string, number>;
}

/**
 * `dynamics` of a skin: the weights' Gram over the object's splats (`mass`) and over its anchor
 * splats (`anchorGram`), each `m × m` as its upper triangle row by row, `w_0 = 1` included.
 */
export interface SkinDynamics {
  mass: number[];
  anchorGram: number[];
  /** How many splats anchor it, and the band above its lowest splat they lie in (metres). */
  anchorSplats: number;
  anchorBand: number;
}

export interface SkinTile {
  runs: Int32Array;
  /** The first row of `skin.bin` the tile's skinned splats take. */
  row: number;
}

export interface SkinDoc {
  /** `method.name`: how the weights were made (`simplicits-rkpm`, `limbs`, ...), or null. */
  method: string | null;
  skins: SkinEntry[];
  byId: ReadonlyMap<number, SkinEntry>;
  byInstance: ReadonlyMap<number, SkinEntry>;
  maxId: number;
  tiles: ReadonlyMap<string, SkinTile>;
  /** `skin.bin` as words: `rowWords` a row. */
  words: Uint32Array;
  /** 4 (one texel a splat) or 8 (two: a file with a skin of more than 16 handles). */
  rowWords: 4 | 8;
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
  const dynamics = dynamicsOf(r.dynamics, handles as number);
  const traits = traitsOf(r.traits);
  return {
    id: id as number,
    instance: instance as number,
    handles: handles as number,
    origin,
    scale: finite(r.scale, 1),
    eigenvalues,
    support,
    ...(dynamics ? { dynamics } : {}),
    ...(traits ? { traits } : {}),
    ...(typeof r.class === "string" ? { stiffnessClass: r.class } : {}),
  };
}

function traitsOf(raw: unknown): SkinTraits | undefined {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return undefined;
  const out: SkinTraits = {};
  if (typeof r.label === "string") out.label = r.label;
  if (typeof r.category === "string") out.category = r.category;
  if (typeof r.behaviour === "string") out.behaviour = r.behaviour;
  if (typeof r.properties === "object" && r.properties !== null) {
    const properties: Record<string, number> = {};
    for (const [k, v] of Object.entries(r.properties as Record<string, unknown>))
      if (typeof v === "number" && Number.isFinite(v)) properties[k] = v;
    out.properties = properties;
  }
  return out;
}

function positive(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

/** One limb of a limbs skin (`limbs.handles[j − 1]`), or null when any field is unusable. */
function limbHandleOf(raw: unknown, handles: number): LimbHandle | null {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return null;
  const pivot = vec3(r.pivot);
  const direction = vec3(r.direction);
  const samplePoint = vec3(r.samplePoint);
  const frequencyHz = positive(r.frequencyHz);
  const gain = typeof r.gain === "number" && Number.isFinite(r.gain) ? r.gain : null;
  const parent = r.parent;
  if (!pivot || !direction || !samplePoint || frequencyHz === null || gain === null) return null;
  if (!Number.isInteger(r.key) || (r.key as number) < 0) return null;
  if (!Number.isInteger(parent) || (parent as number) < 0 || (parent as number) >= handles)
    return null;
  const damping = finite(r.damping, Number.NaN);
  if (!(damping >= 0 && damping < 1)) return null;
  return {
    key: r.key as number,
    pivot,
    parent: parent as number,
    level: Math.max(0, Math.round(finite(r.level))),
    spanM: Math.max(0, finite(r.spanM)),
    frequencyHz,
    damping,
    tree: r.tree === true,
    gain,
    limitRad: positive(r.limitRad) ?? Number.POSITIVE_INFINITY,
    direction,
    samplePoint,
    widthM: Math.max(0, finite(r.widthM)),
    heightM: Math.max(0, finite(r.heightM)),
    staticTipM: Math.max(0, finite(r.staticTipM)),
    flutterM: Math.max(0, finite(r.flutterM)),
  };
}

/**
 * A limbs skin's `limbs` block (written by `skin_methods.fit_limbs_from_rig`): the plant's wind
 * (seed, reference speed, turbulence, gusts, seasons, flutter) and one record per learned
 * handle, trunk first. Undefined when anything the driver needs is missing or malformed: such
 * a skin is not swayed (and never as eigenmodes).
 */
function limbsOf(
  raw: unknown,
  handles: number,
  origin: Vec3,
  rowBytes: number,
): LimbSkinSource | undefined {
  const r = raw as Record<string, unknown> | null | undefined;
  if (typeof r !== "object" || r === null) return undefined;
  const wind = r.wind as Record<string, unknown> | undefined;
  const turbulence = wind?.turbulence as Record<string, unknown> | undefined;
  const gust = wind?.gust as Record<string, unknown> | undefined;
  const winter = (r.seasons as { winter?: Record<string, unknown> } | undefined)?.winter;
  const list = r.handles;
  if (!Array.isArray(list) || list.length !== handles - 1) return undefined;
  const limbs = list.map((h) => limbHandleOf(h, handles));
  if (limbs.some((h) => h === null)) return undefined;
  const seed = r.seed;
  const reference = positive(r.referenceSpeedMps);
  const leaf = positive(r.leafSizeM);
  const lengthScale = positive(wind?.lengthScaleM);
  if (!Number.isInteger(seed) || reference === null || leaf === null || lengthScale === null)
    return undefined;
  const along = finite(turbulence?.along, Number.NaN);
  const across = finite(turbulence?.across, Number.NaN);
  if (!(along >= 0) || !(across >= 0)) return undefined;
  return {
    origin,
    seed: seed as number,
    referenceSpeedMps: reference,
    leafSizeM: leaf,
    turbulence: { along, across },
    lengthScaleM: lengthScale,
    gust: {
      strength: Math.max(0, finite(gust?.strength)),
      variance: Math.max(0, finite(gust?.variance)),
      frequencyPerMin: positive(gust?.frequencyPerMin) ?? 1,
      durationS: positive(gust?.durationS) ?? 1,
    },
    canopyAdvection: Math.max(0, finite(wind?.canopyAdvection, 0.3)),
    winter: {
      dampingScale: positive(winter?.dampingScale) ?? 1,
      branchFrequencyScale: positive(winter?.branchFrequencyScale) ?? 1,
      flutterScale: Math.max(0, finite(winter?.flutterScale)),
    },
    flutterReferenceM: Math.max(
      0,
      finite((r.flutter as Record<string, unknown> | undefined)?.referenceM),
    ),
    // The row's last byte: no weight uses it (a row of b bytes holds at most b − 1).
    flutterByte: rowBytes - 1,
    handles: limbs as LimbHandle[],
  };
}

/**
 * Numbers a skin's driven handles take: `12·m`, and a limbs skin's flutter after them
 * (`LIMB_FLUTTER_FLOATS`, `limbWind.ts`).
 */
export function skinFloats(skin: Pick<SkinEntry, "handles" | "limbs">): number {
  return skin.handles * HANDLE_FLOATS + (skin.limbs ? LIMB_FLUTTER_FLOATS : 0);
}

/** Whether a skin's splats take rows of `skin.bin`: one of a single handle has no weights. */
export function takesRows(skin: Pick<SkinEntry, "handles"> | undefined): boolean {
  return (skin?.handles ?? 0) > 1;
}

function upperOf(value: unknown, m: number): number[] | null {
  if (!Array.isArray(value) || value.length !== (m * (m + 1)) / 2) return null;
  const out = (value as unknown[]).map((v) => finite(v, Number.NaN));
  return out.every(Number.isFinite) ? out : null;
}

/** A skin's `dynamics`, or undefined when absent or not `m × m`. */
function dynamicsOf(raw: unknown, m: number): SkinDynamics | undefined {
  const r = raw as { mass?: unknown; anchor?: Record<string, unknown> } | null | undefined;
  if (typeof r !== "object" || r === null) return undefined;
  const mass = upperOf(r.mass, m);
  const anchorGram = upperOf(r.anchor?.gram, m);
  if (!mass || !anchorGram) return undefined;
  return {
    mass,
    anchorGram,
    anchorSplats: Math.max(0, Math.round(finite(r.anchor?.splats))),
    anchorBand: finite(r.anchor?.band),
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
  if (weights?.dtype !== "int8") return null;
  if (weights.rowBytes !== SKIN_ROW_BYTES && weights.rowBytes !== SKIN_WIDE_ROW_BYTES) return null;
  const rowBytes = weights.rowBytes;
  const rowWords = rowBytes === SKIN_WIDE_ROW_BYTES ? 8 : 4;
  const methodName = (doc.method as { name?: unknown } | undefined)?.name;
  const method = typeof methodName === "string" ? methodName : null;
  const issues: string[] = [];
  const skins: SkinEntry[] = [];
  const byId = new Map<number, SkinEntry>();
  const byInstance = new Map<number, SkinEntry>();
  for (const entry of doc.skins as unknown[]) {
    const skin = skinOf(entry);
    if (!skin || skin.handles > rowBytes) {
      issues.push(
        `a skin without an id, instance, handles (1..${String(rowBytes)}) and origin was skipped`,
      );
      continue;
    }
    if (method === LIMBS_METHOD && skin.handles > 1) {
      const limbs = limbsOf(
        (entry as { limbs?: unknown }).limbs,
        skin.handles,
        skin.origin,
        rowBytes,
      );
      if (limbs) skin.limbs = limbs;
      else issues.push(`skin ${String(skin.id)}'s limbs are incomplete; it is not swayed`);
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
  const rows = Math.floor(blob.byteLength / rowBytes);
  const words = new Uint32Array(blob, 0, rows * rowWords);
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
      for (let i = 0; i < runs.length; i += 2)
        if (takesRows(byId.get(runs[i] ?? 0))) skinned += runs[i + 1] ?? 0;
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
    method,
    skins,
    byId,
    byInstance,
    maxId,
    tiles,
    words,
    rowWords,
    rows,
    scale: finite(weights.scale, 1 / 127),
    issues,
  };
}

/** One tile's skins and weight rows, per gaussian in its own order. */
export interface TileSkin {
  /** The skin id of every gaussian (0: none). */
  skins: Uint32Array;
  /** Four words a gaussian: its row of `skin.bin` (weights 1..16), zeros where it has none. */
  words: Uint32Array;
  /** In a file of wide rows, the second texel (weights 17..32), four words a gaussian. */
  words2?: Uint32Array;
}

/** The skins and rows of the tile whose checksum is `checksum`, or undefined. */
export function tileSkin(doc: SkinDoc, checksum: string): TileSkin | undefined {
  const tile = doc.tiles.get(checksum);
  if (tile === undefined) return undefined;
  const count = runsLength(tile.runs);
  const skins = decodeRuns(tile.runs, new Uint32Array(count));
  const words = new Uint32Array(count * 4);
  const wide = doc.rowWords === 8;
  const words2 = wide ? new Uint32Array(count * 4) : undefined;
  const stride = doc.rowWords;
  let row = tile.row;
  let last = -1;
  let rows = false;
  for (let i = 0; i < count; i += 1) {
    const id = skins[i] ?? 0;
    if (id === 0) continue;
    if (id !== last) {
      last = id;
      rows = takesRows(doc.byId.get(id));
    }
    if (!rows) continue;
    words.set(doc.words.subarray(row * stride, row * stride + 4), i * 4);
    if (words2) words2.set(doc.words.subarray(row * stride + 4, row * stride + 8), i * 4);
    row += 1;
  }
  return words2 ? { skins, words, words2 } : { skins, words };
}

/**
 * Weight `k` (handle `k + 1`) of a splat's row, as the shader decodes it: from `words` (four a
 * splat) for `k < 16`, from `words2` (a wide row's second texel) beyond.
 */
export function rowWeight(
  words: ArrayLike<number>,
  at: number,
  k: number,
  scale: number,
  words2?: ArrayLike<number>,
): number {
  const source = k < 16 ? words : words2;
  if (!source) return 0;
  const kk = k & 15;
  const word = source[at * 4 + (kk >> 2)] ?? 0;
  const byte = (word >>> (8 * (kk & 3))) & 0xff;
  return (byte >= 128 ? byte - 256 : byte) * scale;
}

/**
 * Fetches and reads a scan's `skin.json` and its weights. Throws when either is missing. Kept
 * in memory once read (lib/payloadCache.ts): picking a motion method again is instant.
 */
export function loadSkin(tilesetUrl: string, ref: SkinRef): Promise<SkinDoc> {
  const url = resolveBeside(tilesetUrl, ref.uri);
  return scanPayloads.get(url, async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`skin answered ${String(response.status)}`);
    const text = await response.text();
    const raw = JSON.parse(text) as { weights?: { file?: unknown } };
    const file = typeof raw.weights?.file === "string" ? raw.weights.file : "skin.bin";
    const binary = await fetch(resolveBeside(url, file));
    if (!binary.ok) throw new Error(`skin weights answered ${String(binary.status)}`);
    const weights = await binary.arrayBuffer();
    const doc = parseSkin(raw, weights);
    if (!doc) throw new Error("skin: not a hexapod.skin v1 document");
    return { value: doc, bytes: weights.byteLength + jsonBytes(text) };
  });
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
