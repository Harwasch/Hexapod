/**
 * The frame arithmetic between the rig's local ENU positions and the baked positions the engine
 * renders. Pure: no Cesium, no GPU.
 *
 * `GaussianSplatPrimitive.transformTile` (CesiumJS 1.145) writes
 *
 *     content.positions[i] = B · gltfPositions[i],
 *     B = inverse(rootTransform) · tile.computedTransform · axisCorrection · content.worldTransform
 *
 * and caches `B` on the content. Everything downstream — the snapshot, `primitive._positions`,
 * the packed texture — is in that baked frame, while the motion rig is in the local ENU frame
 * the capture was authored in. So the deformer has to move both ways: un-bake once at attach to
 * recover the canonical local positions, and re-bake every frame to produce what the texture
 * expects.
 *
 * The un-bake is what closes the checksum gap. `checksumPositions` is bit-exact and was taken
 * over the *local* positions, which `B` has since destroyed; but the capture pipeline snaps
 * every coordinate to SPZ's 1/4096 m grid before writing, so `snap(B⁻¹ · B · p)` recovers the
 * original float32 exactly as long as the round trip's error stays well inside half a grid step
 * — it is around 2e-6 m for a tree-sized capture, against a 1.2e-4 m half-step. That makes the
 * bit-exact digest usable at runtime, which is the difference between "this is the tree the rig
 * was built for" and "this is approximately tree-shaped".
 */

/** A 4×4 matrix in CesiumJS's column-major order: element `(row r, column c)` is `m[c * 4 + r]`. */
export type Mat4 = ArrayLike<number>;

/** SPZ's position quantisation step, metres. Every coordinate in a tiled capture is a multiple. */
export const SPZ_POSITION_QUANTUM_M = 1 / 4096;

/** WGS84 semi-major axis, metres. */
const WGS84_A = 6378137.0;
/** WGS84 semi-minor axis, metres. */
const WGS84_B = 6356752.314245179;

/** Mutable xyz scratch, so the hot loops allocate nothing. */
export type MutableVec3 = [number, number, number];

/**
 * `matrix · (x, y, z, 1)`, in the same order of operations as `Matrix4.multiplyByPoint`.
 *
 * The order matters: `transformTile` takes the fast path for a rigid placement and computes the
 * baked positions with exactly this expression, so reproducing it term for term makes our
 * re-bake of an undisplaced splat bit-identical to the engine's — which is how wind → 0 returns
 * the tree to its measured pose exactly rather than nearly.
 */
export function transformPoint(
  matrix: Mat4,
  x: number,
  y: number,
  z: number,
  out: MutableVec3,
): MutableVec3 {
  const m = matrix;
  out[0] = (m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z + (m[12] ?? 0);
  out[1] = (m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z + (m[13] ?? 0);
  out[2] = (m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z + (m[14] ?? 0);
  return out;
}

/**
 * The inverse of an affine matrix (bottom row `[0, 0, 0, 1]`), or `undefined` when the upper
 * 3×3 is singular or non-finite. Returning `undefined` rather than NaNs keeps a degenerate
 * placement a refusal instead of a tree that renders at the centre of the Earth.
 */
export function invertAffine(matrix: Mat4): number[] | undefined {
  const m = matrix;
  const a = m[0] ?? 0;
  const b = m[4] ?? 0;
  const c = m[8] ?? 0;
  const d = m[1] ?? 0;
  const e = m[5] ?? 0;
  const f = m[9] ?? 0;
  const g = m[2] ?? 0;
  const h = m[6] ?? 0;
  const i = m[10] ?? 0;
  const cofactor0 = e * i - f * h;
  const cofactor1 = f * g - d * i;
  const cofactor2 = d * h - e * g;
  const det = a * cofactor0 + b * cofactor1 + c * cofactor2;
  if (!Number.isFinite(det) || det === 0) return undefined;
  const invDet = 1 / det;
  const r00 = cofactor0 * invDet;
  const r01 = (c * h - b * i) * invDet;
  const r02 = (b * f - c * e) * invDet;
  const r10 = cofactor1 * invDet;
  const r11 = (a * i - c * g) * invDet;
  const r12 = (c * d - a * f) * invDet;
  const r20 = cofactor2 * invDet;
  const r21 = (b * g - a * h) * invDet;
  const r22 = (a * e - b * d) * invDet;
  const tx = m[12] ?? 0;
  const ty = m[13] ?? 0;
  const tz = m[14] ?? 0;
  const out = [
    r00,
    r10,
    r20,
    0,
    r01,
    r11,
    r21,
    0,
    r02,
    r12,
    r22,
    0,
    -(r00 * tx + r01 * ty + r02 * tz),
    -(r10 * tx + r11 * ty + r12 * tz),
    -(r20 * tx + r21 * ty + r22 * tz),
    1,
  ];
  return out.every((value) => Number.isFinite(value)) ? out : undefined;
}

/** Applies `matrix` to a flat xyz array, writing into `out`. `out` may not alias `positions`. */
export function transformPositions(
  positions: Float32Array,
  matrix: Mat4,
  out: Float32Array,
): Float32Array {
  if (out === positions) throw new Error("transformPositions: out must not alias positions");
  const count = Math.min(Math.floor(positions.length / 3), Math.floor(out.length / 3));
  const scratch: MutableVec3 = [0, 0, 0];
  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    transformPoint(
      matrix,
      positions[base] ?? 0,
      positions[base + 1] ?? 0,
      positions[base + 2] ?? 0,
      scratch,
    );
    out[base] = scratch[0];
    out[base + 1] = scratch[1];
    out[base + 2] = scratch[2];
  }
  return out;
}

/**
 * Rounds to the nearest multiple of `quantum`, normalising `-0` to `+0`.
 *
 * The signed zero is not pedantry: `checksumPositions` digests bytes, `-0` and `+0` differ in
 * bytes, and SPZ's integer round trip normalises `-0` to `+0`. The capture tool normalises for
 * the same reason (S2's third finding); un-baking has to agree or the digest misses.
 */
export function snapToGrid(value: number, quantum: number): number {
  const snapped = Math.round(value / quantum) * quantum;
  return snapped === 0 ? 0 : snapped;
}

/**
 * Recovers the canonical local positions from baked ones: `snap(inverse · p)`.
 *
 * The snap is the point of the exercise — without it the result is only numerically close to
 * the positions the rig was built from, and a bit-exact digest over it would never match.
 */
export function unbakePositions(
  baked: Float32Array,
  inverse: Mat4,
  quantum: number = SPZ_POSITION_QUANTUM_M,
  out?: Float32Array,
): Float32Array {
  const count = Math.floor(baked.length / 3);
  const target = out ?? new Float32Array(count * 3);
  if (target === baked) throw new Error("unbakePositions: out must not alias baked");
  const scratch: MutableVec3 = [0, 0, 0];
  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    transformPoint(inverse, baked[base] ?? 0, baked[base + 1] ?? 0, baked[base + 2] ?? 0, scratch);
    target[base] = snapToGrid(scratch[0], quantum);
    target[base + 1] = snapToGrid(scratch[1], quantum);
    target[base + 2] = snapToGrid(scratch[2], quantum);
  }
  return target;
}

/** The largest absolute difference between two flat arrays, or `Infinity` if lengths differ. */
export function maxAbsDifference(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let worst = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = Math.abs((a[i] ?? 0) - (b[i] ?? 0));
    if (d > worst) worst = d;
  }
  return worst;
}

/** The WGS84 geodetic surface normal at an ECEF point, as a unit vector. */
export function wgs84SurfaceNormal(x: number, y: number, z: number): MutableVec3 {
  const nx = x / (WGS84_A * WGS84_A);
  const ny = y / (WGS84_A * WGS84_A);
  const nz = z / (WGS84_B * WGS84_B);
  const length = Math.hypot(nx, ny, nz);
  if (!(length > 0) || !Number.isFinite(length)) return [0, 0, 1];
  return [nx / length, ny / length, nz / length];
}

/**
 * How well a root transform's Z column agrees with the geodetic normal at its own origin: the
 * dot product, `1` for a perfect east-north-up frame.
 *
 * This is the frame assertion, worded the way S0's measurements forced. "The principal axis of
 * the capture points up" is *false* for a correct site-sized capture — mygla's is
 * `[0.316, 0.948, 0.026]`, nearly horizontal — and false for the synthetic tree too, whose
 * crown makes its point cloud almost isotropic (principal axis `[0.86, -0.21, 0.46]`). The
 * root transform, by contrast, is `Transforms.eastNorthUpToFixedFrame` of the tileset centre and
 * measured `1.0` to fifteen significant figures. Check the thing that is actually true.
 */
export function geodeticFrameAlignment(rootTransform: Mat4): number {
  const m = rootTransform;
  const normal = wgs84SurfaceNormal(m[12] ?? 0, m[13] ?? 0, m[14] ?? 0);
  const zx = m[8] ?? 0;
  const zy = m[9] ?? 0;
  const zz = m[10] ?? 0;
  const length = Math.hypot(zx, zy, zz);
  if (!(length > 0) || !Number.isFinite(length)) return 0;
  return (zx * normal[0] + zy * normal[1] + zz * normal[2]) / length;
}

/** Below this the local frame's up axis is not the geodetic up and the deformer refuses. */
export const GEODETIC_ALIGNMENT_MIN = 1 - 1e-6;

/** What the tree's own point cloud says about which way is up. */
export interface TreeUprightness {
  /** Extent along local +Z, metres. */
  readonly verticalExtentM: number;
  /** Largest horizontal extent, metres. */
  readonly horizontalExtentM: number;
  /** Horizontal spread of the lowest 15 % of the tree — the trunk, if +Z is up. */
  readonly baseSpreadM: number;
  /** Horizontal spread of the upper half — the crown, if +Z is up. */
  readonly crownSpreadM: number;
  /** `crownSpread / baseSpread`. Large for a tree standing up, near 1 for one lying down. */
  readonly crownToBase: number;
  readonly upright: boolean;
}

/** A tree is judged upright when its crown is at least this many times wider than its base. */
export const TREE_CROWN_TO_BASE_MIN = 3;

/** 90th-percentile radius of a set of horizontal offsets about their own mean, metres. */
function horizontalSpread(xs: readonly number[], ys: readonly number[]): number {
  const n = xs.length;
  if (n < 8) return Number.NaN;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i += 1) {
    mx += xs[i] ?? 0;
    my += ys[i] ?? 0;
  }
  mx /= n;
  my /= n;
  const radii = new Float64Array(n);
  for (let i = 0; i < n; i += 1) radii[i] = Math.hypot((xs[i] ?? 0) - mx, (ys[i] ?? 0) - my);
  radii.sort();
  return radii[Math.min(n - 1, Math.floor(n * 0.9))] ?? Number.NaN;
}

/**
 * Whether an isolated tree's local positions really stand along +Z, by comparing the horizontal
 * spread of its base against its crown.
 *
 * Belt and braces behind the checksum, which already pins the positions exactly; this is the
 * check that still means something for a capture whose rig was derived some other way. On the
 * committed synthetic fixture the ratio is 9.9 with +Z up, and 1.4 and 1.7 for the two axis
 * swaps — a wide enough margin that the threshold is not a tuned number.
 */
export function treeUprightness(local: Float32Array): TreeUprightness {
  const count = Math.floor(local.length / 3);
  const empty: TreeUprightness = {
    verticalExtentM: 0,
    horizontalExtentM: 0,
    baseSpreadM: Number.NaN,
    crownSpreadM: Number.NaN,
    crownToBase: Number.NaN,
    upright: false,
  };
  if (count < 16) return empty;
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i += 1) {
    const x = local[i * 3] ?? 0;
    const y = local[i * 3 + 1] ?? 0;
    const z = local[i * 3 + 2] ?? 0;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const verticalExtentM = maxZ - minZ;
  const horizontalExtentM = Math.max(maxX - minX, maxY - minY);
  if (!(verticalExtentM > 0)) return { ...empty, horizontalExtentM };
  const baseTop = minZ + 0.15 * verticalExtentM;
  const crownFloor = minZ + 0.5 * verticalExtentM;
  const baseX: number[] = [];
  const baseY: number[] = [];
  const crownX: number[] = [];
  const crownY: number[] = [];
  for (let i = 0; i < count; i += 1) {
    const x = local[i * 3] ?? 0;
    const y = local[i * 3 + 1] ?? 0;
    const z = local[i * 3 + 2] ?? 0;
    if (z <= baseTop) {
      baseX.push(x);
      baseY.push(y);
    }
    if (z >= crownFloor) {
      crownX.push(x);
      crownY.push(y);
    }
  }
  const baseSpreadM = horizontalSpread(baseX, baseY);
  const crownSpreadM = horizontalSpread(crownX, crownY);
  const crownToBase = baseSpreadM > 0 ? crownSpreadM / baseSpreadM : Number.POSITIVE_INFINITY;
  const upright = Number.isFinite(crownSpreadM) && crownToBase >= TREE_CROWN_TO_BASE_MIN;
  return {
    verticalExtentM,
    horizontalExtentM,
    baseSpreadM,
    crownSpreadM,
    crownToBase,
    upright,
  };
}

/**
 * Produces the baked positions a frame should upload.
 *
 * A splat whose node is at rest is copied straight from `canonicalBaked` — the engine's own
 * bytes — rather than re-baked. That is what makes zero wind *exactly* the measured pose: the
 * re-bake is numerically faithful but not guaranteed bit-identical, and "returns to where it was
 * measured" is a claim this feature should be able to make without an epsilon.
 *
 * `out` must not alias either input.
 */
export function resolveBakedPositions(
  displacedLocal: Float32Array,
  canonicalBaked: Float32Array,
  assignment: Uint16Array,
  nodeMoves: Uint8Array,
  matrix: Mat4,
  out: Float32Array,
): Float32Array {
  if (out === displacedLocal || out === canonicalBaked) {
    throw new Error("resolveBakedPositions: out must not alias its inputs");
  }
  const count = Math.min(
    Math.floor(displacedLocal.length / 3),
    Math.floor(canonicalBaked.length / 3),
    Math.floor(out.length / 3),
    assignment.length,
  );
  const scratch: MutableVec3 = [0, 0, 0];
  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    if ((nodeMoves[assignment[i] ?? 0] ?? 0) === 0) {
      out[base] = canonicalBaked[base] ?? 0;
      out[base + 1] = canonicalBaked[base + 1] ?? 0;
      out[base + 2] = canonicalBaked[base + 2] ?? 0;
      continue;
    }
    transformPoint(
      matrix,
      displacedLocal[base] ?? 0,
      displacedLocal[base + 1] ?? 0,
      displacedLocal[base + 2] ?? 0,
      scratch,
    );
    out[base] = scratch[0];
    out[base + 1] = scratch[1];
    out[base + 2] = scratch[2];
  }
  return out;
}
