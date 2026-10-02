/**
 * Scene objects moving under a dedicated splat renderer (PlayCanvas, Spark), as the motion
 * chain moves them under CesiumJS's own (docs/SCENE_OBJECTS.md §4): a skin's handles (the
 * wind, C1; telemetry through a skin's constant handle, C3) and a rigid motion per instance
 * (telemetry on an unskinned object, C3).
 *
 * **The drivers are shared.** The wind (`skinWind.ts`) and telemetry (`telemetry.ts`) hand
 * their motions to the scan's skin part (`splatSkin.ts`, `setInstanceHandles`) and rigid part
 * (`splatRigid.ts`, `setInstanceMotion`) whatever renderer draws the scan -- CesiumJS keeps the
 * scan's tileset loaded, hidden, under every renderer -- and those parts keep what was set
 * (`drivenSkins`, `instanceMotions`). `ScanMotionLink` reads them each frame and hands the
 * back-end one `ScanMotion` whenever anything changed: renderers apply motions, they never
 * drive them.
 *
 * **The data is the same.** Per splat, the instance id (from `instances.json`) and the skin id
 * and weight row (from `skin.json` + `skin.bin`), each keyed by the checksum of the tile's own
 * positions, which every back-end digests as it decodes a tile (`scanInstances.ts`). Per skin
 * and per driven instance, small tables in the layout CesiumJS's parts upload:
 *
 * - **handles** (RGBA32F, 1024 texels a row, `TEXELS_PER_SKIN` a skin): `(moving, m)`, then
 *   each handle's `[A | t − A·o]` -- `Z_j` folded about the skin's origin `o` into the scan's
 *   frame, the frame both back-ends draw in -- so a splat moves by `Σ_j w_j (A_j x + t_j)`;
 * - **slots** (RGBA32UI, four instance ids a texel): each id's slot, written at a driven
 *   instance and every instance below it (a deeper driven instance keeps its own);
 * - **poses** (RGBA32F, three texels a slot): `[R − I | t]`.
 *
 * `SCAN_MOTION_GLSL` is the shader both back-ends share: the displacement and its linear part
 * `A`, and the covariance drawn through `J = I + A` (`hexapodCovariance`). Neither renderer
 * takes a covariance from a modifier -- PlayCanvas's work buffer and Spark's splats hold a
 * rotation and three scales -- so `J·Σ·Jᵀ` is decomposed again into a rotation and scales
 * (Jacobi), which is exact: any symmetric positive matrix is `R·S²·Rᵀ`. A turn (`J` a
 * rotation, telemetry's rigid motion) takes the short way: the splat's rotation turned.
 *
 * A scan PlayCanvas streams from its own package (`sog/lod-meta.json`) carries no tile
 * checksums, so neither ids nor skins can be bound there: the link reports a gap, which the
 * objects panel and the wind control show with a switch to CesiumJS (`state/instances.ts`).
 */

import type { RigidMotion } from "@twin/world";

import { withDescendants, type InstancesDoc } from "@/lib/instances";
import { splitObjectsOf } from "@/lib/sceneObjects";
import { HANDLE_FLOATS, rigidHandle, skinRefOf, type SkinDoc } from "@/lib/skin";
import { telemetryRefOf } from "@/lib/telemetry";
import { useInstances } from "@/state/instances";

import { instancesDocOf } from "../splatInstances";
import { foldHandle, handleTextureRows, skinningOf, TEXELS_PER_SKIN } from "../splatSkin";
import { telemetryOf } from "../telemetry";
import type { ScanBackend } from "./types";

/** Texels a row of every motion table holds. */
export const MOTION_TEXTURE_WIDTH = 1024;
/** Instance ids a slot texel holds. */
export const SLOTS_PER_TEXEL = 4;
/** Pose texels a slot takes. */
export const TEXELS_PER_SLOT = 3;
const FLOATS_PER_TEXEL = 4;
const IDENTITY: readonly number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Everything a back-end needs to draw a scan's objects moving. */
export interface ScanMotion {
  /** Whose instance ids a tile carries (rigid motion), by checksum. */
  readonly instances: InstancesDoc | null;
  /** Whose skin ids and weight rows a tile carries, by checksum. */
  readonly skin: SkinDoc | null;
  /** RGBA32F texels, `MOTION_TEXTURE_WIDTH` a row: per skin a header, then `[A | t]` rows. */
  readonly handles: Float32Array;
  readonly handleRows: number;
  /** RGBA32UI texels, four ids a texel: the slot of every instance id, 0 for none. */
  readonly slots: Uint32Array;
  readonly slotRows: number;
  /** RGBA32F texels, three a slot: `[R − I | t]` (scan frame). */
  readonly poses: Float32Array;
  readonly poseRows: number;
  /** x: any skin moving; y: the largest skin id; z: any instance moving; w: the largest id. */
  readonly params: readonly [number, number, number, number];
  /** x: what a stored weight byte is worth; y: covariances follow the motion (1) or not. */
  readonly extra: readonly [number, number, number, number];
  /** Skins whose handles changed since the last motion handed: tiles holding them redraw. */
  readonly changedSkins: ReadonlySet<number>;
  /** Leaf instance ids whose rigid motion changed since the last motion handed. */
  readonly changedIds: ReadonlySet<number>;
  /** Each moving leaf instance id's motion, `[R | t]` row-major (scan frame): for sorting. */
  readonly rigidByLeaf: ReadonlyMap<number, Float64Array>;
}

/** What the link reads: the skin part's and the rigid part's state, and the instances. */
export interface MotionSources {
  skin():
    | {
        doc: SkinDoc;
        drivenSkins: ReadonlyMap<number, Float64Array>;
        covariance: boolean;
        motionVersion: number;
      }
    | undefined;
  rigid(): { instanceMotions: ReadonlyMap<number, RigidMotion>; motionVersion: number } | undefined;
  instances(): InstancesDoc | undefined;
}

/** The shared parts of `assetId`'s scan (`splatSkin.ts`, `telemetry.ts`, `splatInstances.ts`). */
export function sharedMotionSources(assetId: string): MotionSources {
  return {
    skin: () => skinningOf(assetId),
    rigid: () => telemetryOf(assetId)?.rigid,
    instances: () => instancesDocOf(assetId),
  };
}

function rowsFor(count: number, perRow: number): number {
  return Math.max(1, Math.ceil(count / perRow));
}

/** The skin handle table: every driven skin's handles folded about its origin (scan frame). */
export function packSkinHandles(
  doc: SkinDoc,
  driven: ReadonlyMap<number, ArrayLike<number>>,
): { data: Float32Array; rows: number } {
  const rows = handleTextureRows(doc.maxId);
  const data = new Float32Array(rows * MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL);
  for (const [id, handles] of driven) {
    const skin = doc.byId.get(id);
    if (!skin) continue;
    const base = id * TEXELS_PER_SKIN * FLOATS_PER_TEXEL;
    data[base] = 1;
    data[base + 1] = skin.handles;
    for (let j = 0; j < skin.handles; j += 1) {
      foldHandle(
        handles,
        j * HANDLE_FLOATS,
        skin.origin,
        IDENTITY,
        IDENTITY,
        data,
        base + (1 + 3 * j) * FLOATS_PER_TEXEL,
      );
    }
  }
  return { data, rows };
}

/**
 * The slot of every instance id and each slot's pose, for the driven instances: a driven
 * instance's slot is written at it and at everything below it, shallow ones first, so a driven
 * instance inside another keeps its own (as `splatRigid.ts`).
 */
export function packRigid(
  doc: Pick<InstancesDoc, "instances" | "maxId">,
  motions: ReadonlyMap<number, RigidMotion>,
): {
  slots: Uint32Array;
  slotRows: number;
  poses: Float32Array;
  poseRows: number;
  byLeaf: Map<number, Float64Array>;
} {
  const slotRows = rowsFor(doc.maxId + 1, MOTION_TEXTURE_WIDTH * SLOTS_PER_TEXEL);
  const slots = new Uint32Array(slotRows * MOTION_TEXTURE_WIDTH * SLOTS_PER_TEXEL);
  const levels = new Map(doc.instances.map((i) => [i.id, i.level] as const));
  const driven = [...motions.keys()]
    .filter((id) => id >= 1 && id <= doc.maxId)
    .sort((a, b) => (levels.get(a) ?? 0) - (levels.get(b) ?? 0) || a - b);
  const poseRows = rowsFor((driven.length + 1) * TEXELS_PER_SLOT, MOTION_TEXTURE_WIDTH);
  const poses = new Float32Array(poseRows * MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL);
  const byLeaf = new Map<number, Float64Array>();
  driven.forEach((id, k) => {
    const slot = k + 1;
    const motion = motions.get(id);
    if (!motion) return;
    const z = rigidHandle(motion.rotation, motion.translation);
    poses.set(
      Array.from(z, (v) => v),
      slot * TEXELS_PER_SLOT * FLOATS_PER_TEXEL,
    );
    const full = Float64Array.from(z);
    full[0] = (full[0] ?? 0) + 1;
    full[5] = (full[5] ?? 0) + 1;
    full[10] = (full[10] ?? 0) + 1;
    for (const leaf of withDescendants(doc, [id])) {
      if (leaf < 1 || leaf > doc.maxId) continue;
      slots[leaf] = slot;
      byLeaf.set(leaf, full);
    }
  });
  return { slots, slotRows, poses, poseRows, byLeaf };
}

/** Builds the motion back-ends draw from the shared parts' state. */
export function buildScanMotion(
  instances: InstancesDoc | null,
  skin: {
    doc: SkinDoc;
    drivenSkins: ReadonlyMap<number, ArrayLike<number>>;
    covariance: boolean;
  } | null,
  rigid: ReadonlyMap<number, RigidMotion> | null,
  changedSkins: ReadonlySet<number>,
  changedIds: ReadonlySet<number>,
): ScanMotion {
  const handles = skin
    ? packSkinHandles(skin.doc, skin.drivenSkins)
    : { data: new Float32Array(MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL), rows: 1 };
  const packed =
    instances && rigid
      ? packRigid(instances, rigid)
      : {
          slots: new Uint32Array(MOTION_TEXTURE_WIDTH * SLOTS_PER_TEXEL),
          slotRows: 1,
          poses: new Float32Array(MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL),
          poseRows: 1,
          byLeaf: new Map<number, Float64Array>(),
        };
  const skinMoving = skin !== null && skin.drivenSkins.size > 0;
  const rigidMoving = packed.byLeaf.size > 0;
  return {
    instances,
    skin: skin?.doc ?? null,
    handles: handles.data,
    handleRows: handles.rows,
    slots: packed.slots,
    slotRows: packed.slotRows,
    poses: packed.poses,
    poseRows: packed.poseRows,
    params: [skinMoving ? 1 : 0, skin?.doc.maxId ?? 0, rigidMoving ? 1 : 0, instances?.maxId ?? 0],
    extra: [skin?.doc.scale ?? 0, skin?.covariance === false ? 0 : 1, 0, 0],
    changedSkins,
    changedIds,
    rigidByLeaf: packed.byLeaf,
  };
}

/**
 * The GLSL both back-ends share. Samplers are parameters (Spark's dynos name their uniforms),
 * positions are the scan frame's, rotations are quaternions `x, y, z, w`.
 */
export const SCAN_MOTION_GLSL = `
vec4 hexapodMotionTexel(highp sampler2D table, int i) {
    return texelFetch(table, ivec2(i & ${String(MOTION_TEXTURE_WIDTH - 1)}, i >> ${String(Math.log2(MOTION_TEXTURE_WIDTH))}), 0);
}

uint hexapodMotionQuad(highp usampler2D table, uint i) {
    uint q = i >> 2u;
    uvec4 v = texelFetch(table, ivec2(int(q & ${String(MOTION_TEXTURE_WIDTH - 1)}u), int(q >> ${String(Math.log2(MOTION_TEXTURE_WIDTH))}u)), 0);
    uint k = i & 3u;
    return k == 0u ? v.r : (k == 1u ? v.g : (k == 2u ? v.b : v.a));
}

float hexapodSkinWeight(uvec4 words, int k, float scale) {
    uint word = k < 4 ? words.x : (k < 8 ? words.y : (k < 12 ? words.z : words.w));
    // Byte k & 3 of the word, sign-extended: an int8 weight.
    return float(int(word << uint(24 - 8 * (k & 3))) >> 24) * scale;
}

// Adds skin \`skin\`'s displacement of x to delta and its linear part to linear.
void hexapodSkinMotion(highp sampler2D handles, uint skin, uvec4 words, vec4 params, float scale,
                       vec3 x, inout vec3 delta, inout mat3 linear) {
    if (params.x < 0.5 || skin == 0u || float(skin) > params.y) {
        return;
    }
    int base = int(skin) * ${String(TEXELS_PER_SKIN)};
    vec4 head = hexapodMotionTexel(handles, base);
    if (head.x < 0.5) {
        return;
    }
    int count = int(head.y);
    vec4 xh = vec4(x, 1.0);
    for (int j = 0; j < 16; j++) {
        if (j >= count) {
            break;
        }
        float w = j == 0 ? 1.0 : hexapodSkinWeight(words, j - 1, scale);
        if (w == 0.0) {
            continue;
        }
        int at = base + 1 + 3 * j;
        vec4 r0 = hexapodMotionTexel(handles, at);
        vec4 r1 = hexapodMotionTexel(handles, at + 1);
        vec4 r2 = hexapodMotionTexel(handles, at + 2);
        delta += w * vec3(dot(r0, xh), dot(r1, xh), dot(r2, xh));
        // mat3 is column-major: column c holds row entries (r0[c], r1[c], r2[c]).
        linear += w * mat3(r0.x, r1.x, r2.x, r0.y, r1.y, r2.y, r0.z, r1.z, r2.z);
    }
}

// Adds instance \`id\`'s rigid displacement of x to delta and its linear part to linear.
void hexapodRigidMotion(highp usampler2D slots, highp sampler2D poses, uint id, vec4 params,
                        vec3 x, inout vec3 delta, inout mat3 linear) {
    if (params.z < 0.5 || id == 0u || float(id) > params.w) {
        return;
    }
    uint slot = hexapodMotionQuad(slots, id);
    if (slot == 0u) {
        return;
    }
    int at = int(slot) * ${String(TEXELS_PER_SLOT)};
    vec4 r0 = hexapodMotionTexel(poses, at);
    vec4 r1 = hexapodMotionTexel(poses, at + 1);
    vec4 r2 = hexapodMotionTexel(poses, at + 2);
    vec4 xh = vec4(x, 1.0);
    delta += vec3(dot(r0, xh), dot(r1, xh), dot(r2, xh));
    linear += mat3(r0.x, r1.x, r2.x, r0.y, r1.y, r2.y, r0.z, r1.z, r2.z);
}

vec4 hexapodQuatOfMatrix(mat3 m) {
    // m[c][r] is row r, column c.
    float m00 = m[0][0];
    float m11 = m[1][1];
    float m22 = m[2][2];
    float m01 = m[1][0];
    float m10 = m[0][1];
    float m02 = m[2][0];
    float m20 = m[0][2];
    float m12 = m[2][1];
    float m21 = m[1][2];
    float trace = m00 + m11 + m22;
    vec4 q;
    if (trace > 0.0) {
        float s = sqrt(trace + 1.0) * 2.0;
        q = vec4((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s);
    } else if (m00 > m11 && m00 > m22) {
        float s = sqrt(1.0 + m00 - m11 - m22) * 2.0;
        q = vec4(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
    } else if (m11 > m22) {
        float s = sqrt(1.0 + m11 - m00 - m22) * 2.0;
        q = vec4((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s);
    } else {
        float s = sqrt(1.0 + m22 - m00 - m11) * 2.0;
        q = vec4((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s);
    }
    return normalize(q);
}

mat3 hexapodMatrixOfQuat(vec4 q) {
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;
    return mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
        2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
        2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)
    );
}

vec4 hexapodQuatMul(vec4 a, vec4 b) {
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

// One Jacobi rotation of symmetric a, zeroing a[p][q]; v gathers the eigenvectors (columns).
void hexapodJacobi(inout mat3 a, inout mat3 v, int p, int q) {
    float apq = a[q][p];
    if (abs(apq) < 1e-30) {
        return;
    }
    float theta = (a[q][q] - a[p][p]) / (2.0 * apq);
    float t = (theta >= 0.0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1.0));
    float c = inversesqrt(t * t + 1.0);
    float s = t * c;
    mat3 r = mat3(1.0);
    r[p][p] = c;
    r[q][q] = c;
    r[q][p] = s;
    r[p][q] = -s;
    a = transpose(r) * a * r;
    v = v * r;
}

// The splat's rotation and scales once its covariance is drawn through j: J·Σ·Jᵀ = R'·S'²·R'ᵀ.
void hexapodCovariance(mat3 j, inout vec4 rotation, inout vec3 scale) {
    if (j == mat3(1.0)) {
        return;
    }
    mat3 jtj = transpose(j) * j - mat3(1.0);
    float off = max(max(max(abs(jtj[0][0]), abs(jtj[1][1])), max(abs(jtj[2][2]), abs(jtj[1][0]))),
                    max(abs(jtj[2][0]), abs(jtj[2][1])));
    if (off < 1e-5 && determinant(j) > 0.0) {
        // A turn: the splat turns with it, its scales stay.
        rotation = normalize(hexapodQuatMul(hexapodQuatOfMatrix(j), rotation));
        return;
    }
    mat3 m = j * hexapodMatrixOfQuat(normalize(rotation)) * mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    mat3 a = m * transpose(m);
    mat3 v = mat3(1.0);
    for (int sweep = 0; sweep < 6; sweep++) {
        hexapodJacobi(a, v, 0, 1);
        hexapodJacobi(a, v, 0, 2);
        hexapodJacobi(a, v, 1, 2);
    }
    if (determinant(v) < 0.0) {
        v[2] = -v[2];
    }
    scale = sqrt(max(vec3(a[0][0], a[1][1], a[2][2]), vec3(0.0)));
    rotation = hexapodQuatOfMatrix(v);
}
`;

// ---- CPU mirrors (tests) ----------------------------------------------------------------

/** What `hexapodSkinMotion` + `hexapodRigidMotion` compute for one splat, in float64. */
export function evaluateScanMotion(
  motion: Pick<ScanMotion, "handles" | "slots" | "poses" | "params" | "extra">,
  splat: { skin: number; words: ArrayLike<number>; id: number },
  x: readonly [number, number, number],
): { delta: [number, number, number]; linear: number[] } {
  const delta: [number, number, number] = [0, 0, 0];
  const linear = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const texel = (table: Float32Array, i: number): number[] =>
    [0, 1, 2, 3].map((k) => table[i * FLOATS_PER_TEXEL + k] ?? 0);
  const add = (rows: number[][], w: number): void => {
    for (let r = 0; r < 3; r += 1) {
      const row = rows[r] ?? [];
      delta[r] =
        (delta[r] ?? 0) +
        w * ((row[0] ?? 0) * x[0] + (row[1] ?? 0) * x[1] + (row[2] ?? 0) * x[2] + (row[3] ?? 0));
      for (let c = 0; c < 3; c += 1)
        linear[r * 3 + c] = (linear[r * 3 + c] ?? 0) + w * (row[c] ?? 0);
    }
  };
  const [skinOn, maxSkin, rigidOn, maxId] = motion.params;
  if (skinOn >= 0.5 && splat.skin !== 0 && splat.skin <= maxSkin) {
    const base = splat.skin * TEXELS_PER_SKIN;
    const head = texel(motion.handles, base);
    if ((head[0] ?? 0) >= 0.5) {
      for (let j = 0; j < Math.min(head[1] ?? 0, 16); j += 1) {
        let w = 1;
        if (j > 0) {
          const word = splat.words[(j - 1) >> 2] ?? 0;
          const byte = (word >>> (8 * ((j - 1) & 3))) & 0xff;
          w = (byte >= 128 ? byte - 256 : byte) * motion.extra[0];
        }
        if (w === 0) continue;
        const at = base + 1 + 3 * j;
        add(
          [texel(motion.handles, at), texel(motion.handles, at + 1), texel(motion.handles, at + 2)],
          w,
        );
      }
    }
  }
  if (rigidOn >= 0.5 && splat.id !== 0 && splat.id <= maxId) {
    const slot = motion.slots[splat.id] ?? 0;
    if (slot !== 0) {
      const at = slot * TEXELS_PER_SLOT;
      add([texel(motion.poses, at), texel(motion.poses, at + 1), texel(motion.poses, at + 2)], 1);
    }
  }
  return { delta, linear };
}

type Mat3 = number[][]; // [row][col]

function mul3(a: Mat3, b: Mat3): Mat3 {
  return [0, 1, 2].map((r) =>
    [0, 1, 2].map((c) =>
      [0, 1, 2].reduce((sum, k) => sum + (a[r]?.[k] ?? 0) * (b[k]?.[c] ?? 0), 0),
    ),
  );
}

function transpose3(a: Mat3): Mat3 {
  return [0, 1, 2].map((r) => [0, 1, 2].map((c) => a[c]?.[r] ?? 0));
}

function det3(m: Mat3): number {
  const g = (r: number, c: number): number => m[r]?.[c] ?? 0;
  return (
    g(0, 0) * (g(1, 1) * g(2, 2) - g(1, 2) * g(2, 1)) -
    g(0, 1) * (g(1, 0) * g(2, 2) - g(1, 2) * g(2, 0)) +
    g(0, 2) * (g(1, 0) * g(2, 1) - g(1, 1) * g(2, 0))
  );
}

/** A unit quaternion `x, y, z, w` as a rotation matrix, [row][col]. */
export function matrixOfQuat(q: readonly number[]): Mat3 {
  const [x = 0, y = 0, z = 0, w = 1] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

function quatOfMatrix(m: Mat3): [number, number, number, number] {
  const g = (r: number, c: number): number => m[r]?.[c] ?? 0;
  const trace = g(0, 0) + g(1, 1) + g(2, 2);
  let q: [number, number, number, number];
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(g(2, 1) - g(1, 2)) / s, (g(0, 2) - g(2, 0)) / s, (g(1, 0) - g(0, 1)) / s, 0.25 * s];
  } else if (g(0, 0) > g(1, 1) && g(0, 0) > g(2, 2)) {
    const s = Math.sqrt(1 + g(0, 0) - g(1, 1) - g(2, 2)) * 2;
    q = [0.25 * s, (g(0, 1) + g(1, 0)) / s, (g(0, 2) + g(2, 0)) / s, (g(2, 1) - g(1, 2)) / s];
  } else if (g(1, 1) > g(2, 2)) {
    const s = Math.sqrt(1 + g(1, 1) - g(0, 0) - g(2, 2)) * 2;
    q = [(g(0, 1) + g(1, 0)) / s, 0.25 * s, (g(1, 2) + g(2, 1)) / s, (g(0, 2) - g(2, 0)) / s];
  } else {
    const s = Math.sqrt(1 + g(2, 2) - g(0, 0) - g(1, 1)) * 2;
    q = [(g(0, 2) + g(2, 0)) / s, (g(1, 2) + g(2, 1)) / s, 0.25 * s, (g(1, 0) - g(0, 1)) / s];
  }
  const n = Math.hypot(...q);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function quatMul(a: readonly number[], b: readonly number[]): [number, number, number, number] {
  const [ax = 0, ay = 0, az = 0, aw = 1] = a;
  const [bx = 0, by = 0, bz = 0, bw = 1] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/**
 * What `hexapodCovariance` does, in float64: the rotation and scales of a splat whose
 * covariance `R·S²·Rᵀ` is drawn through `j` ([row][col]).
 */
export function covarianceThrough(
  j: Mat3,
  rotation: readonly number[],
  scale: readonly number[],
): { rotation: [number, number, number, number]; scale: [number, number, number] } {
  const identity = j.every((row, r) => row.every((v, c) => v === (r === c ? 1 : 0)));
  const q0: [number, number, number, number] = [
    rotation[0] ?? 0,
    rotation[1] ?? 0,
    rotation[2] ?? 0,
    rotation[3] ?? 1,
  ];
  const s0: [number, number, number] = [scale[0] ?? 0, scale[1] ?? 0, scale[2] ?? 0];
  if (identity) return { rotation: q0, scale: s0 };
  const jtj = mul3(transpose3(j), j);
  let off = 0;
  for (let r = 0; r < 3; r += 1)
    for (let c = 0; c < 3; c += 1)
      off = Math.max(off, Math.abs((jtj[r]?.[c] ?? 0) - (r === c ? 1 : 0)));
  if (off < 1e-5 && det3(j) > 0) {
    const turned = quatMul(quatOfMatrix(j), q0);
    const n = Math.hypot(...turned);
    return { rotation: [turned[0] / n, turned[1] / n, turned[2] / n, turned[3] / n], scale: s0 };
  }
  const n0 = Math.hypot(...q0);
  const m = mul3(mul3(j, matrixOfQuat(q0.map((v) => v / n0))), [
    [s0[0], 0, 0],
    [0, s0[1], 0],
    [0, 0, s0[2]],
  ]);
  let a = mul3(m, transpose3(m));
  let v: Mat3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const rotate = (p: number, q: number): void => {
    const apq = a[p]?.[q] ?? 0;
    if (Math.abs(apq) < 1e-30) return;
    const theta = ((a[q]?.[q] ?? 0) - (a[p]?.[p] ?? 0)) / (2 * apq);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    // Row p, column q is s; row q, column p is -s (Numerical Recipes' P_pq).
    const entry = (row: number, col: number): number => {
      if (row === col) return row === p || row === q ? c : 1;
      if (row === p && col === q) return s;
      if (row === q && col === p) return -s;
      return 0;
    };
    const r: Mat3 = [0, 1, 2].map((row) => [0, 1, 2].map((col) => entry(row, col)));
    a = mul3(mul3(transpose3(r), a), r);
    v = mul3(v, r);
  };
  for (let sweep = 0; sweep < 6; sweep += 1) {
    rotate(0, 1);
    rotate(0, 2);
    rotate(1, 2);
  }
  if (det3(v) < 0) v = v.map((row) => [row[0] ?? 0, row[1] ?? 0, -(row[2] ?? 0)]);
  return {
    rotation: quatOfMatrix(v),
    scale: [0, 1, 2].map((k) => Math.sqrt(Math.max(0, a[k]?.[k] ?? 0))) as [number, number, number],
  };
}

/**
 * A tile's centres moved by the rigid motions of its splats' instances, for a sorter that
 * reads centres on the CPU (PlayCanvas): `rest` relative to the tile's `origin` (scan frame),
 * `ids` per splat in the same order. A moving splat goes to `R (c + o) + t − o`; the others
 * keep their rest centre. Returns whether any moved.
 */
export function movedCenters(
  rest: Float32Array,
  origin: readonly [number, number, number],
  ids: ArrayLike<number>,
  byLeaf: ReadonlyMap<number, ArrayLike<number>>,
  out: Float32Array,
): boolean {
  out.set(rest);
  if (byLeaf.size === 0) return false;
  let moved = false;
  const count = Math.min(ids.length, Math.floor(rest.length / 3));
  for (let i = 0; i < count; i += 1) {
    const m = byLeaf.get(ids[i] ?? 0);
    if (!m) continue;
    moved = true;
    const x = (rest[i * 3] ?? 0) + origin[0];
    const y = (rest[i * 3 + 1] ?? 0) + origin[1];
    const z = (rest[i * 3 + 2] ?? 0) + origin[2];
    for (let r = 0; r < 3; r += 1) {
      out[i * 3 + r] =
        (m[r * 4] ?? 0) * x +
        (m[r * 4 + 1] ?? 0) * y +
        (m[r * 4 + 2] ?? 0) * z +
        (m[r * 4 + 3] ?? 0) -
        (origin[r] ?? 0);
    }
  }
  return moved;
}

// ---- The link ----------------------------------------------------------------------------

/** Whether a scan's root declares anything that moves its objects (skin, telemetry). */
export function declaresMotion(extras: unknown): boolean {
  return skinRefOf(extras) !== null || telemetryRefOf(extras) !== null;
}

/** Whether a scan's root declares split objects. */
export function declaresObjects(extras: unknown): boolean {
  return splitObjectsOf(extras).length > 0;
}

/** Why a back-end cannot move the scan's objects, or null when it can. */
export function motionGap(
  backend: Pick<ScanBackend<unknown>, "setMotion" | "name">,
  native: boolean,
): string | null {
  if (!backend.setMotion) return `The ${backend.name} renderer cannot move them.`;
  if (native) {
    return "This scan streams in PlayCanvas's own format, which carries no tile checksums to bind skins and object ids to.";
  }
  return null;
}

/**
 * Keeps `backend` drawing asset `assetId`'s objects as the drivers move them: `update` once a
 * frame (before the render) hands the back-end a new `ScanMotion` when anything changed, and
 * says so. A back-end that cannot (no `setMotion`, or a scan streamed without checksums) is
 * reported to the store as a motion gap -- only when the scan declares motion at all.
 */
export class ScanMotionLink {
  readonly #backend: Pick<ScanBackend<unknown>, "setMotion" | "name">;
  readonly #sources: MotionSources;
  readonly #assetId: string;
  readonly #gap: string | null;
  /** The parts' state and versions last handed over (their documents and maps by identity). */
  #seen: {
    skin: unknown;
    skinVersion: number | undefined;
    covariance: boolean | undefined;
    rigid: unknown;
    rigidVersion: number | undefined;
  } = {
    skin: undefined,
    skinVersion: undefined,
    covariance: undefined,
    rigid: undefined,
    rigidVersion: undefined,
  };
  #skins = new Map<number, ArrayLike<number>>();
  #rigid = new Map<number, RigidMotion>();
  #rigidDoc: InstancesDoc | null = null;
  #handed = false;
  #updates = 0;

  constructor(
    assetId: string,
    backend: Pick<ScanBackend<unknown>, "setMotion" | "name">,
    options: { native: boolean; extras: unknown; sources?: MotionSources },
  ) {
    this.#assetId = assetId;
    this.#backend = backend;
    this.#sources = options.sources ?? sharedMotionSources(assetId);
    this.#gap = declaresMotion(options.extras) ? motionGap(backend, options.native) : null;
    if (this.#gap !== null) {
      useInstances.getState().setMotionGap(assetId, { renderer: backend.name, reason: this.#gap });
    }
  }

  /** Motions handed to the back-end so far. */
  get updates(): number {
    return this.#updates;
  }

  /** Brings the back-end up to the drivers' state; true when it was handed a new motion. */
  update(): boolean {
    if (this.#gap !== null || !this.#backend.setMotion) return false;
    const skin = this.#sources.skin();
    const rigid = this.#sources.rigid();
    const instances = this.#sources.instances() ?? null;
    const seen = this.#seen;
    if (
      this.#handed &&
      seen.skin === skin?.doc &&
      seen.skinVersion === skin?.motionVersion &&
      seen.covariance === skin?.covariance &&
      seen.rigid === rigid?.instanceMotions &&
      seen.rigidVersion === rigid?.motionVersion &&
      instances === this.#rigidDoc
    ) {
      return false;
    }
    this.#seen = {
      skin: skin?.doc,
      skinVersion: skin?.motionVersion,
      covariance: skin?.covariance,
      rigid: rigid?.instanceMotions,
      rigidVersion: rigid?.motionVersion,
    };
    // What changed since the last hand-over: tiles holding it redraw.
    const changedSkins = new Set<number>();
    const nowSkins = skin?.drivenSkins ?? new Map<number, Float64Array>();
    for (const [id, handles] of nowSkins) if (this.#skins.get(id) !== handles) changedSkins.add(id);
    for (const id of this.#skins.keys()) if (!nowSkins.has(id)) changedSkins.add(id);
    this.#skins = new Map(nowSkins);
    const nowRigid = rigid?.instanceMotions ?? new Map<number, RigidMotion>();
    const changedRoots = new Set<number>();
    for (const [id, motion] of nowRigid) if (this.#rigid.get(id) !== motion) changedRoots.add(id);
    for (const id of this.#rigid.keys()) if (!nowRigid.has(id)) changedRoots.add(id);
    this.#rigid = new Map(nowRigid);
    if (instances !== this.#rigidDoc) for (const id of nowRigid.keys()) changedRoots.add(id);
    this.#rigidDoc = instances;
    const changedIds = instances ? withDescendants(instances, changedRoots) : new Set<number>();
    this.#backend.setMotion(
      buildScanMotion(
        instances,
        skin ? { doc: skin.doc, drivenSkins: nowSkins, covariance: skin.covariance } : null,
        instances ? nowRigid : null,
        changedSkins,
        changedIds,
      ),
    );
    this.#handed = true;
    this.#updates += 1;
    return true;
  }

  dispose(): void {
    if (this.#gap !== null) useInstances.getState().setMotionGap(this.#assetId, null);
    else if (this.#handed) this.#backend.setMotion?.(null);
  }
}
