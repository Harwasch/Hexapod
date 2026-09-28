/**
 * The GPU motion path: the rig's per-frame motion evaluated in the splat vertex shader, so the
 * per-frame cost stops being linear in the splat count.
 *
 * The CPU path (`SplatDeformer`'s default) recomputes every displaced splat on the CPU and
 * re-uploads the attribute-texture rows they sit on: 1.7 ms of CPU for the 12,000-splat tree,
 * and a cost that grows with every gaussian of a level-of-detail view. Here the CPU does per
 * **node** work only — a few hundred affine rows and flutter coefficients, one small texture
 * upload — and the vertex shader applies them per splat through the engine patch's
 * `vertexMotion` hook (`patches/@cesium__engine@26.3.0.patch`).
 *
 * Three textures of our own, never the engine's:
 *
 * - **motion** (RGBA32F, 1024 texels wide). Row 0: the three rows of the bake matrix's linear
 *   part `L` (texels 0–2, for rotating flutter into the baked frame), the flutter axis table
 *   (texels 4–515, two per axis) and — from texel 520, ten texels — the **flutter frame**: which
 *   kind of flutter this frame carries, and for Living Mode's advected field its per-frame
 *   lookup maps (below). From row 1, four texels per rig node: the node's affine transform **in
 *   the baked frame**, `[M | T]` as three rows, and its flutter texel — the legacy field's four
 *   coefficients, or the advected field's amplitude in `x`. `M = L·R·L⁻¹` and
 *   `T = b − M·b + L·t` for `B = (L, b)` and the node's local `(R, t)`, so
 *   `M·p_b + T = B·(R·B⁻¹·p_b + t)` — the CPU path's arithmetic, folded per node. Re-uploaded
 *   every moving frame: a row per 256 nodes (16 KB for this tree) plus the flutter frame's ten
 *   texels (160 B).
 * - **binding** (RG32UI, addressed like the attribute texture). Per splat: its rig node and its
 *   flutter hash (`flutterHash(positionKey)`). Uploaded once per snapshot.
 * - **leaf flutter** (RGBA32F, `n × n`). Living Mode's periodic flutter motion texture
 *   (`leafFlutter.ts`), which depends on the sidecar seed alone, so it is uploaded once per
 *   seed. Packed for bilinear gathering: texel `(x, y)` holds `T[x, y]`, `T[x+1, y]`,
 *   `T[x, y+1]`, `T[x+1, y+1]` (wrapping), so each bilinear sample is one fetch rather than
 *   four — three a splat, 16 MB of GPU memory for the 1024² texture. Float, not half: half
 *   precision alone would cost ~0.2 mm of flutter at full amplitude.
 *
 * **Advected flutter** is looked up at the splat's canonical position, as on the CPU
 * (`applyAdvectedFlutter`), and the fetched position *is* canonical — in the baked frame. Each
 * lookup coordinate is affine in the rig-frame position `q = L⁻¹·(p_b − b)`, so the CPU folds
 * the un-bake, the wind frame, the texel scale and the advection into one affine row per
 * coordinate, `G·p_b + g` — six rows for three lookups — once per frame, in float64, with `g`
 * reduced modulo the texture size (the texture is periodic, so this changes nothing but the
 * float32 precision the shader works at). The three values come back in the wind frame
 * `(along, across, up)` and go to the baked frame through `C = L·W`, three more rows.
 *
 * **Never writing canonical data** is structural here: the attribute texture, `_positions` and
 * everything the sorter reads are untouched, because the displacement exists only inside the
 * vertex shader. **Calm is exact**: `u_splatMotionActive` is 0 whenever nothing moves, and the
 * shader then returns the fetched position without touching it; a node at rest is written as the
 * exact identity and the shader returns its splats' positions untouched too. And **the wrong
 * binding is never drawn**: the active flag is evaluated at draw time against the primitive's
 * committed snapshot generation, so a snapshot committed between our write and the draw renders
 * at rest rather than with another snapshot's bindings.
 *
 * What it does not do: re-sort. The sorter still orders by canonical positions, exactly as on
 * the CPU path (see `docs/LIVING_SURVEY.md`, "Draw order").
 */

import {
  FLUTTER_GPU_LAYOUT,
  flutterCoefficients,
  flutterHash,
  IDENTITY_TRANSFORM,
  isAdvectedFlutter,
  type FlutterField,
  type MotionTexture,
  type NodeTransform,
} from "@twin/world";

import { invertAffine, type Mat4 } from "./splatFrames";
import type {
  SplatPrimitive,
  SplatShaderBuilder,
  SplatTexture,
  SplatVertexMotion,
} from "./splatInternals";
import type { SplatTextureLayout } from "./splatTexels";

/** Texels per row of the motion texture. A power of two, so the shader addresses by shift. */
export const MOTION_TEXTURE_WIDTH = 1024;
/** Where the flutter axis table starts: after `L`'s three rows and one spare texel. */
export const AXIS_TEXEL_BASE = 4;
/** Where node data starts: the second row. */
export const NODE_TEXEL_BASE = MOTION_TEXTURE_WIDTH;
/** `[M|T]` as three rows, then four flutter coefficients. */
export const TEXELS_PER_NODE = 4;
/**
 * Where the per-frame flutter frame starts in row 0, after the axis table. Texel `+0`: the
 * kind (`x` = 1 for advected, 0 otherwise); `+1..+3`: the rows of `C`, wind frame to baked;
 * `+4..+9`: for lookup `j`, the affine rows `G·p_b + g` of its `x` then `y` texture coordinate.
 */
export const FLUTTER_FRAME_TEXEL = 520;
export const FLUTTER_FRAME_TEXELS = 10;

const FLOATS_PER_TEXEL = 4;

/** Rows the motion texture needs for `nodeCount` nodes. */
export function motionTextureHeight(nodeCount: number): number {
  return 1 + Math.max(1, Math.ceil((nodeCount * TEXELS_PER_NODE) / MOTION_TEXTURE_WIDTH));
}

/** A texture we own. */
export interface OwnedTexture extends SplatTexture {
  destroy(): void;
}

/** How textures are made. The real one (`splatGpuTextures.ts`) wraps `Renderer/Texture`. */
export interface MotionTextureFactory {
  createFloat(context: unknown, width: number, height: number, data: Float32Array): OwnedTexture;
  createUintPairs(context: unknown, width: number, height: number, data: Uint32Array): OwnedTexture;
  /** `ShaderDestination.VERTEX`. */
  readonly vertexDestination: number;
}

/** The linear part of `B` as three rows, plus the flutter axes: row 0 of the motion texture. */
export function writeMotionHeader(out: Float32Array, bake: Mat4): void {
  for (let r = 0; r < 3; r += 1) {
    out[r * 4] = bake[r] ?? 0;
    out[r * 4 + 1] = bake[4 + r] ?? 0;
    out[r * 4 + 2] = bake[8 + r] ?? 0;
    out[r * 4 + 3] = 0;
  }
  const axes = FLUTTER_GPU_LAYOUT.axes;
  for (let k = 0; k < FLUTTER_GPU_LAYOUT.axisCount; k += 1) {
    const base = (AXIS_TEXEL_BASE + k * 2) * FLOATS_PER_TEXEL;
    for (let j = 0; j < 3; j += 1) {
      out[base + j] = axes[k * 6 + j] ?? 0;
      out[base + 4 + j] = axes[k * 6 + 3 + j] ?? 0;
    }
    out[base + 3] = 0;
    out[base + 7] = 0;
  }
}

/**
 * Every node's baked-frame affine rows and flutter texel, into the motion texture's node rows.
 * A node flagged still in `nodeMoves` is written as the exact identity with no flutter, which
 * the shader recognises and leaves alone. The flutter texel is the legacy field's four
 * coefficients, or — for an advected field — the node's amplitude in `x`.
 */
export function writeNodeRows(
  out: Float32Array,
  transforms: readonly NodeTransform[],
  flutter: FlutterField,
  nodeMoves: Uint8Array,
  bake: Mat4,
  inverse: Mat4,
): void {
  const advected = isAdvectedFlutter(flutter);
  const coefficients = flutter.still || advected ? undefined : flutterCoefficients(flutter);
  const amplitudes = advected && !flutter.still ? flutter.amplitudeM : undefined;
  // L and L⁻¹ row-major (element (r, c) at r * 3 + c), b as a vector; scratch for R, L·R, M.
  const l = MATRIX_SCRATCH.subarray(0, 9);
  const li = MATRIX_SCRATCH.subarray(9, 18);
  const rot = MATRIX_SCRATCH.subarray(18, 27);
  const lr = MATRIX_SCRATCH.subarray(27, 36);
  const m = MATRIX_SCRATCH.subarray(36, 45);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      l[r * 3 + c] = bake[c * 4 + r] ?? 0;
      li[r * 3 + c] = inverse[c * 4 + r] ?? 0;
    }
  }
  const b0 = bake[12] ?? 0;
  const b1 = bake[13] ?? 0;
  const b2 = bake[14] ?? 0;
  for (let n = 0; n < nodeMoves.length; n += 1) {
    const base = (NODE_TEXEL_BASE + n * TEXELS_PER_NODE) * FLOATS_PER_TEXEL;
    if ((nodeMoves[n] ?? 0) === 0) {
      out.fill(0, base, base + TEXELS_PER_NODE * FLOATS_PER_TEXEL);
      out[base] = 1;
      out[base + 5] = 1;
      out[base + 10] = 1;
      continue;
    }
    const transform = transforms[n] ?? IDENTITY_TRANSFORM;
    const [x, y, z, w] = transform.rotation;
    rot[0] = 1 - 2 * (y * y + z * z);
    rot[1] = 2 * (x * y - z * w);
    rot[2] = 2 * (x * z + y * w);
    rot[3] = 2 * (x * y + z * w);
    rot[4] = 1 - 2 * (x * x + z * z);
    rot[5] = 2 * (y * z - x * w);
    rot[6] = 2 * (x * z - y * w);
    rot[7] = 2 * (y * z + x * w);
    rot[8] = 1 - 2 * (x * x + y * y);
    // M = L·R·L⁻¹
    multiply3(l, rot, lr);
    multiply3(lr, li, m);
    const [t0, t1, t2] = transform.translation;
    for (let i = 0; i < 3; i += 1) {
      const m0 = m[i * 3] ?? 0;
      const m1 = m[i * 3 + 1] ?? 0;
      const m2 = m[i * 3 + 2] ?? 0;
      // T = b − M·b + L·t
      out[base + i * 4] = m0;
      out[base + i * 4 + 1] = m1;
      out[base + i * 4 + 2] = m2;
      out[base + i * 4 + 3] =
        (i === 0 ? b0 : i === 1 ? b1 : b2) -
        (m0 * b0 + m1 * b1 + m2 * b2) +
        ((l[i * 3] ?? 0) * t0 + (l[i * 3 + 1] ?? 0) * t1 + (l[i * 3 + 2] ?? 0) * t2);
    }
    if (amplitudes !== undefined) {
      out[base + 12] = amplitudes[n] ?? 0;
      out[base + 13] = 0;
      out[base + 14] = 0;
      out[base + 15] = 0;
    } else {
      for (let c = 0; c < 4; c += 1) out[base + 12 + c] = coefficients?.[n * 4 + c] ?? 0;
    }
  }
}

/**
 * The flutter frame: row 0's texels from {@link FLUTTER_FRAME_TEXEL}. For an advected field
 * that moves, its kind, `C = L·W` (wind frame to baked frame) and the six lookup rows
 * `G·p_b + g`; anything else is written as kind 0, which the shader reads as the legacy field.
 *
 * All of it is folded here in float64 so the shader's float32 only ever sees small numbers:
 * `g` is reduced into `[−n/2, n/2]`, whatever the advection has grown to.
 */
export function writeFlutterFrame(
  out: Float32Array,
  flutter: FlutterField,
  bake: Mat4,
  inverse: Mat4,
): void {
  const base = FLUTTER_FRAME_TEXEL * FLOATS_PER_TEXEL;
  out.fill(0, base, base + FLUTTER_FRAME_TEXELS * FLOATS_PER_TEXEL);
  if (!isAdvectedFlutter(flutter) || flutter.still) return;
  const [ex, ey] = flutter.downwind;
  const k = flutter.texelsPerMeter;
  const adv = flutter.advectionTexels;
  const m = flutter.lookups;
  const n = flutter.texture.size;
  const l = (r: number, c: number): number => bake[c * 4 + r] ?? 0;
  const li = (r: number, c: number): number => inverse[c * 4 + r] ?? 0;
  const b = [bake[12] ?? 0, bake[13] ?? 0, bake[14] ?? 0];
  out[base] = 1;
  for (let r = 0; r < 3; r += 1) {
    const o = base + (1 + r) * FLOATS_PER_TEXEL;
    // W's columns: along → (ex, ey, 0), across → (ey, −ex, 0), up → (0, 0, 1).
    out[o] = l(r, 0) * ex + l(r, 1) * ey;
    out[o + 1] = l(r, 0) * ey - l(r, 1) * ex;
    out[o + 2] = l(r, 2);
  }
  for (let j = 0; j < 3; j += 1) {
    for (let axis = 0; axis < 2; axis += 1) {
      const mo = j * 8 + axis * 3;
      const mA = m[mo] ?? 0;
      const mC = m[mo + 1] ?? 0;
      const mH = m[mo + 2] ?? 0;
      // The coordinate over the rig-frame position q, then over p_b through q = L⁻¹·(p_b − b).
      const v = [k * (mA * ex + mC * ey), k * (mA * ey - mC * ex), k * mH];
      const g = [0, 1, 2].map(
        (c) => (v[0] ?? 0) * li(0, c) + (v[1] ?? 0) * li(1, c) + (v[2] ?? 0) * li(2, c),
      );
      let offset =
        (m[j * 8 + 6 + axis] ?? 0) -
        mA * adv -
        ((g[0] ?? 0) * (b[0] ?? 0) + (g[1] ?? 0) * (b[1] ?? 0) + (g[2] ?? 0) * (b[2] ?? 0));
      offset -= Math.round(offset / n) * n;
      const o = base + (4 + j * 2 + axis) * FLOATS_PER_TEXEL;
      out[o] = g[0] ?? 0;
      out[o + 1] = g[1] ?? 0;
      out[o + 2] = g[2] ?? 0;
      out[o + 3] = offset;
    }
  }
}

/**
 * A flutter motion texture packed for one-fetch bilinear sampling: texel `(x, y)` holds
 * `T[x, y]`, `T[x+1, y]`, `T[x, y+1]`, `T[x+1, y+1]`, wrapping at the edges.
 */
export function packFlutterTexture(texture: MotionTexture): Float32Array {
  const n = texture.size;
  const mask = n - 1;
  const data = texture.data;
  const out = new Float32Array(n * n * FLOATS_PER_TEXEL);
  for (let y = 0; y < n; y += 1) {
    const ya = y * n;
    const yb = ((y + 1) & mask) * n;
    for (let x = 0; x < n; x += 1) {
      const xb = (x + 1) & mask;
      const o = (ya + x) * FLOATS_PER_TEXEL;
      out[o] = data[ya + x] ?? 0;
      out[o + 1] = data[ya + xb] ?? 0;
      out[o + 2] = data[yb + x] ?? 0;
      out[o + 3] = data[yb + xb] ?? 0;
    }
  }
  return out;
}

/** Five row-major 3×3 scratch matrices for {@link writeNodeRows}. */
const MATRIX_SCRATCH = new Float64Array(45);

/** `out = a · b` for row-major 3×3. `out` must not alias either input. */
function multiply3(a: Float64Array, b: Float64Array, out: Float64Array): void {
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      out[i * 3 + j] =
        (a[i * 3] ?? 0) * (b[j] ?? 0) +
        (a[i * 3 + 1] ?? 0) * (b[3 + j] ?? 0) +
        (a[i * 3 + 2] ?? 0) * (b[6 + j] ?? 0);
    }
  }
}

/**
 * Per-splat binding words in the attribute texture's addressing: `(node, flutterHash)` at
 * texel `(i & rowMask, i >> rowShift)` of a `splatsPerRow`-wide texture.
 */
export function bindingTexels(
  assignment: Uint16Array,
  flutterKeys: Uint32Array,
  layout: SplatTextureLayout,
): { width: number; height: number; data: Uint32Array } {
  const width = layout.splatsPerRow;
  const height = layout.height;
  const data = new Uint32Array(width * height * 2);
  const count = Math.min(layout.numSplats, assignment.length, flutterKeys.length);
  for (let i = 0; i < count; i += 1) {
    const texel = (i >>> layout.rowShift) * width + (i & layout.rowMask);
    data[texel * 2] = assignment[i] ?? 0;
    data[texel * 2 + 1] = flutterHash(flutterKeys[i] ?? 0);
  }
  return { width, height, data };
}

/**
 * The shader side. `u_splatRowMask`/`u_splatRowShift` are the engine's own uniforms, declared
 * before any vertex lines; everything else is ours.
 *
 * Kept to a transcription of the CPU arithmetic, which `evaluateSplatMotion` below restates in
 * TypeScript for the unit tests: one affine row product per axis, and — only for a node with
 * flutter — either the legacy field's two phase look-ups, two table reads and the rotation into
 * the baked frame, or the advected field's three lookups (an affine row pair and one gathered
 * bilinear fetch each) and the rotation `C` out of the wind frame.
 */
export const SPLAT_MOTION_GLSL = `
const float SPLAT_MOTION_PHASE_STEP = ${((2 * Math.PI) / FLUTTER_GPU_LAYOUT.phaseSteps).toPrecision(
  17,
)};

vec4 splatMotionTexel(int index) {
    return texelFetch(u_splatMotion, ivec2(index & ${String(MOTION_TEXTURE_WIDTH - 1)}, index >> ${String(
      Math.log2(MOTION_TEXTURE_WIDTH),
    )}), 0);
}

// Bilinear, wrapping, at texel coordinates: \`sampleTexture\` in spectral.ts, one gathered fetch.
float splatFlutterSample(vec2 p) {
    int n = textureSize(u_splatFlutterTexture, 0).x;
    float size = float(n);
    vec2 f = p - floor(p / size) * size;
    vec2 f0 = floor(f);
    vec2 t = f - f0;
    vec4 q = texelFetch(u_splatFlutterTexture, ivec2(f0) & (n - 1), 0);
    return (q.x + (q.y - q.x) * t.x) * (1.0 - t.y) + (q.z + (q.w - q.z) * t.x) * t.y;
}

float splatFlutterLookup(int row, vec3 position) {
    vec4 gx = splatMotionTexel(row);
    vec4 gy = splatMotionTexel(row + 1);
    return splatFlutterSample(vec2(dot(gx.xyz, position) + gx.w, dot(gy.xyz, position) + gy.w));
}

vec3 splatVertexMotion(uint splatIndex, vec3 position) {
    if (u_splatMotionActive < 0.5) {
        return position;
    }
    uint rowMask = uint(u_splatRowMask);
    uint rowShift = uint(u_splatRowShift);
    uvec2 binding = texelFetch(
        u_splatMotionBinding,
        ivec2(int(splatIndex & rowMask), int(splatIndex >> rowShift)),
        0
    ).rg;
    int base = ${String(NODE_TEXEL_BASE)} + int(binding.r) * ${String(TEXELS_PER_NODE)};
    vec4 r0 = splatMotionTexel(base);
    vec4 r1 = splatMotionTexel(base + 1);
    vec4 r2 = splatMotionTexel(base + 2);
    vec4 c = splatMotionTexel(base + 3);
    if (r0 == vec4(1.0, 0.0, 0.0, 0.0) && r1 == vec4(0.0, 1.0, 0.0, 0.0) &&
        r2 == vec4(0.0, 0.0, 1.0, 0.0) && c == vec4(0.0)) {
        return position;
    }
    vec3 moved = vec3(
        dot(r0.xyz, position) + r0.w,
        dot(r1.xyz, position) + r1.w,
        dot(r2.xyz, position) + r2.w
    );
    if (c != vec4(0.0) && splatMotionTexel(${String(FLUTTER_FRAME_TEXEL)}).x > 0.5) {
        // Living Mode's advected field, looked up at the canonical (fetched) position.
        vec3 wind = vec3(
            splatFlutterLookup(${String(FLUTTER_FRAME_TEXEL + 4)}, position),
            splatFlutterLookup(${String(FLUTTER_FRAME_TEXEL + 6)}, position),
            splatFlutterLookup(${String(FLUTTER_FRAME_TEXEL + 8)}, position)
        );
        moved += c.x * vec3(
            dot(splatMotionTexel(${String(FLUTTER_FRAME_TEXEL + 1)}).xyz, wind),
            dot(splatMotionTexel(${String(FLUTTER_FRAME_TEXEL + 2)}).xyz, wind),
            dot(splatMotionTexel(${String(FLUTTER_FRAME_TEXEL + 3)}).xyz, wind)
        );
    } else if (c != vec4(0.0)) {
        uint h = binding.g;
        float p1 = float(h & 1023u) * SPLAT_MOTION_PHASE_STEP;
        float p2 = float((h >> 10u) & 1023u) * SPLAT_MOTION_PHASE_STEP;
        int axis = ${String(AXIS_TEXEL_BASE)} + int((h >> 20u) & 255u) * 2;
        float wave1 = c.x * cos(p1) + c.y * sin(p1);
        float wave2 = c.z * cos(p2) + c.w * sin(p2);
        vec3 local = wave1 * splatMotionTexel(axis).xyz + wave2 * splatMotionTexel(axis + 1).xyz;
        moved += vec3(
            dot(splatMotionTexel(0).xyz, local),
            dot(splatMotionTexel(1).xyz, local),
            dot(splatMotionTexel(2).xyz, local)
        );
    }
    return moved;
}
`;

/**
 * `splatVertexMotion`, in TypeScript and float64: what the shader computes for one splat, read
 * from the same texture contents. Exists so the texture packing and the shader's arithmetic
 * can be checked against the CPU path without a GPU.
 */
export function evaluateSplatMotion(
  motion: Float32Array,
  node: number,
  hash: number,
  position: readonly [number, number, number],
  /** The leaf-flutter texture as uploaded ({@link packFlutterTexture}), for an advected frame. */
  flutterTexture?: { readonly size: number; readonly data: Float32Array },
): [number, number, number] {
  const texel = (index: number): number[] =>
    [0, 1, 2, 3].map((k) => motion[index * FLOATS_PER_TEXEL + k] ?? 0);
  const base = NODE_TEXEL_BASE + node * TEXELS_PER_NODE;
  const [r0, r1, r2, c] = [texel(base), texel(base + 1), texel(base + 2), texel(base + 3)];
  const rows = [r0, r1, r2] as number[][];
  const still =
    rows.every((row, i) => row.every((v, j) => v === (j === i ? 1 : 0))) &&
    (c ?? []).every((v) => v === 0);
  if (still) return [position[0], position[1], position[2]];
  const out = rows.map(
    (row) =>
      (row[0] ?? 0) * position[0] +
      (row[1] ?? 0) * position[1] +
      (row[2] ?? 0) * position[2] +
      (row[3] ?? 0),
  ) as [number, number, number];
  const flutters = (c ?? []).some((v) => v !== 0);
  if (flutters && (texel(FLUTTER_FRAME_TEXEL)[0] ?? 0) > 0.5) {
    const size = flutterTexture?.size ?? 1;
    const data = flutterTexture?.data ?? new Float32Array(4);
    const sample = (x: number, y: number): number => {
      const fx = x - Math.floor(x / size) * size;
      const fy = y - Math.floor(y / size) * size;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const tx = fx - x0;
      const ty = fy - y0;
      const o = ((y0 & (size - 1)) * size + (x0 & (size - 1))) * FLOATS_PER_TEXEL;
      const [a, b, cc, d] = [data[o] ?? 0, data[o + 1] ?? 0, data[o + 2] ?? 0, data[o + 3] ?? 0];
      return (a + (b - a) * tx) * (1 - ty) + (cc + (d - cc) * tx) * ty;
    };
    const affine = (row: number[]): number =>
      (row[0] ?? 0) * position[0] +
      (row[1] ?? 0) * position[1] +
      (row[2] ?? 0) * position[2] +
      (row[3] ?? 0);
    const wind = [0, 1, 2].map((j) => {
      const row = FLUTTER_FRAME_TEXEL + 4 + j * 2;
      return sample(affine(texel(row)), affine(texel(row + 1)));
    });
    const amplitude = c?.[0] ?? 0;
    for (let r = 0; r < 3; r += 1) {
      const cRow = texel(FLUTTER_FRAME_TEXEL + 1 + r);
      out[r] =
        (out[r] ?? 0) +
        amplitude *
          ((cRow[0] ?? 0) * (wind[0] ?? 0) +
            (cRow[1] ?? 0) * (wind[1] ?? 0) +
            (cRow[2] ?? 0) * (wind[2] ?? 0));
    }
  } else if (flutters) {
    const step = (2 * Math.PI) / FLUTTER_GPU_LAYOUT.phaseSteps;
    const p1 = (hash & 1023) * step;
    const p2 = ((hash >>> 10) & 1023) * step;
    const axis = AXIS_TEXEL_BASE + ((hash >>> 20) & 255) * 2;
    const a = texel(axis);
    const b = texel(axis + 1);
    const wave1 = (c?.[0] ?? 0) * Math.cos(p1) + (c?.[1] ?? 0) * Math.sin(p1);
    const wave2 = (c?.[2] ?? 0) * Math.cos(p2) + (c?.[3] ?? 0) * Math.sin(p2);
    const local = [0, 1, 2].map((k) => wave1 * (a[k] ?? 0) + wave2 * (b[k] ?? 0));
    for (let r = 0; r < 3; r += 1) {
      const lRow = texel(r);
      out[r] =
        (out[r] ?? 0) +
        (lRow[0] ?? 0) * (local[0] ?? 0) +
        (lRow[1] ?? 0) * (local[1] ?? 0) +
        (lRow[2] ?? 0) * (local[2] ?? 0);
    }
  }
  return out;
}

/** One snapshot's binding, as the hook needs it at draw time. */
interface BoundSnapshot {
  readonly generation: number;
  readonly numSplats: number;
  readonly bake: readonly number[];
  readonly inverse: readonly number[];
}

/**
 * The hook object the patched primitive calls, plus the per-snapshot and per-frame writes.
 * One per deformer; installed on the primitive while the deformer is on the GPU path.
 */
export class SplatGpuMotion implements SplatVertexMotion {
  readonly #factory: MotionTextureFactory;
  readonly #nodeCount: number;
  readonly #motion: Float32Array;
  #context: unknown;
  #primitive: SplatPrimitive | undefined;
  #motionTexture: OwnedTexture | undefined;
  #bindingTexture: OwnedTexture | undefined;
  /** The leaf-flutter texture, and the motion texture it was packed from. */
  #flutterTexture: OwnedTexture | undefined;
  #flutterSource: MotionTexture | undefined;
  #pendingBinding: { width: number; height: number; data: Uint32Array } | undefined;
  #bound: BoundSnapshot | undefined;
  #headerWritten = false;
  #active = false;
  /** Words uploaded by the most recent write. */
  lastUploadWords = 0;

  constructor(factory: MotionTextureFactory, nodeCount: number) {
    this.#factory = factory;
    this.#nodeCount = nodeCount;
    this.#motion = new Float32Array(
      MOTION_TEXTURE_WIDTH * motionTextureHeight(nodeCount) * FLOATS_PER_TEXEL,
    );
  }

  /** Whether the engine has handed us a context yet (the first draw-command build). */
  get ready(): boolean {
    return this.#context !== undefined;
  }

  /** Called by the patched engine on every draw-command build. */
  addToShader(
    shaderBuilder: SplatShaderBuilder,
    uniformMap: Record<string, () => unknown>,
    context: unknown,
  ): void {
    this.#context = context;
    // A sampler must always have a texture behind it, so both exist from the first build.
    this.#ensureTextures();
    const vertex = this.#factory.vertexDestination;
    shaderBuilder.addUniform("highp sampler2D", "u_splatMotion", vertex);
    shaderBuilder.addUniform("highp usampler2D", "u_splatMotionBinding", vertex);
    shaderBuilder.addUniform("highp sampler2D", "u_splatFlutterTexture", vertex);
    shaderBuilder.addUniform("float", "u_splatMotionActive", vertex);
    shaderBuilder.addVertexLines(SPLAT_MOTION_GLSL);
    uniformMap.u_splatMotion = () => this.#motionTexture;
    uniformMap.u_splatMotionBinding = () => this.#bindingTexture;
    uniformMap.u_splatFlutterTexture = () => this.#flutterTexture;
    uniformMap.u_splatMotionActive = () => (this.#drawActive() ? 1 : 0);
  }

  /** Installs the hook on `primitive` (idempotent). */
  install(primitive: SplatPrimitive): void {
    if (this.#primitive === primitive && primitive.vertexMotion === this) return;
    this.#primitive = primitive;
    primitive.vertexMotion = this;
  }

  /** Binds a snapshot: its generation, and every splat's node and flutter hash. */
  bind(
    generation: number,
    layout: SplatTextureLayout,
    assignment: Uint16Array,
    flutterKeys: Uint32Array,
    bake: Mat4,
  ): boolean {
    const inverse = invertAffine(bake);
    if (inverse === undefined) return false;
    this.#bound = {
      generation,
      numSplats: layout.numSplats,
      bake: Array.from(bake),
      inverse,
    };
    this.#active = false;
    this.#headerWritten = false;
    this.#pendingBinding = bindingTexels(assignment, flutterKeys, layout);
    this.#ensureTextures();
    return true;
  }

  /**
   * One frame. Returns false when the textures cannot exist yet (no context), in which case
   * nothing will be drawn displaced.
   */
  write(
    transforms: readonly NodeTransform[],
    flutter: FlutterField,
    nodeMoves: Uint8Array,
    moving: boolean,
  ): boolean {
    const bound = this.#bound;
    if (bound === undefined || !this.#ensureTextures()) {
      this.#active = false;
      return false;
    }
    this.lastUploadWords = 0;
    if (!moving) {
      // Calm: the shader returns every fetched position untouched. Nothing to upload.
      this.#active = false;
      return true;
    }
    const texture = this.#motionTexture;
    if (texture === undefined) return false;
    if (isAdvectedFlutter(flutter) && !flutter.still) {
      this.lastUploadWords += this.#uploadFlutterTexture(flutter.texture);
    }
    writeFlutterFrame(this.#motion, flutter, bound.bake, bound.inverse);
    if (!this.#headerWritten) {
      writeMotionHeader(this.#motion, bound.bake);
      texture.copyFrom({
        source: {
          width: MOTION_TEXTURE_WIDTH,
          height: 1,
          arrayBufferView: this.#motion.subarray(0, MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL),
        },
        xOffset: 0,
        yOffset: 0,
      });
      this.#headerWritten = true;
      this.lastUploadWords += MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL;
    } else {
      // Only the flutter frame of row 0 changes from frame to frame: ten texels.
      const from = FLUTTER_FRAME_TEXEL * FLOATS_PER_TEXEL;
      texture.copyFrom({
        source: {
          width: FLUTTER_FRAME_TEXELS,
          height: 1,
          arrayBufferView: this.#motion.subarray(
            from,
            from + FLUTTER_FRAME_TEXELS * FLOATS_PER_TEXEL,
          ),
        },
        xOffset: FLUTTER_FRAME_TEXEL,
        yOffset: 0,
      });
      this.lastUploadWords += FLUTTER_FRAME_TEXELS * FLOATS_PER_TEXEL;
    }
    writeNodeRows(this.#motion, transforms, flutter, nodeMoves, bound.bake, bound.inverse);
    const rows = motionTextureHeight(this.#nodeCount) - 1;
    texture.copyFrom({
      source: {
        width: MOTION_TEXTURE_WIDTH,
        height: rows,
        arrayBufferView: this.#motion.subarray(MOTION_TEXTURE_WIDTH * FLOATS_PER_TEXEL),
      },
      xOffset: 0,
      yOffset: 1,
    });
    this.lastUploadWords += MOTION_TEXTURE_WIDTH * rows * FLOATS_PER_TEXEL;
    this.#active = true;
    return true;
  }

  /** Stops displacing without uninstalling: the next draw is the measured pose. */
  deactivate(): void {
    this.#active = false;
  }

  /** Uninstalls the hook and releases the textures. */
  destroy(): void {
    this.#active = false;
    this.#bound = undefined;
    const primitive = this.#primitive;
    this.#primitive = undefined;
    if (
      primitive !== undefined &&
      primitive.isDestroyed?.() !== true &&
      primitive.vertexMotion === this
    ) {
      // The patched engine rebuilds the draw command before it next pushes it, so the textures
      // below are never bound again after this.
      primitive.vertexMotion = undefined;
    }
    this.#motionTexture?.destroy();
    this.#bindingTexture?.destroy();
    this.#flutterTexture?.destroy();
    this.#motionTexture = undefined;
    this.#bindingTexture = undefined;
    this.#flutterTexture = undefined;
    this.#flutterSource = undefined;
  }

  /** Side of the leaf-flutter texture uploaded, or 0 before an advected field has arrived. */
  get flutterTextureSize(): number {
    return this.#flutterSource?.size ?? 0;
  }

  /** The words the current motion texture holds, for tests. */
  get motionData(): Float32Array {
    return this.#motion;
  }

  #drawActive(): boolean {
    const bound = this.#bound;
    const primitive = this.#primitive;
    return (
      this.#active &&
      bound !== undefined &&
      primitive !== undefined &&
      primitive._snapshot?.generation === bound.generation &&
      primitive._numSplats === bound.numSplats &&
      this.#motionTexture !== undefined &&
      this.#bindingTexture !== undefined &&
      this.#flutterTexture !== undefined &&
      this.#pendingBinding === undefined
    );
  }

  /** Creates what is missing and uploads a pending binding. False without a context. */
  #ensureTextures(): boolean {
    const context = this.#context;
    if (context === undefined) return false;
    if (this.#motionTexture === undefined || this.#motionTexture.isDestroyed()) {
      this.#motionTexture = this.#factory.createFloat(
        context,
        MOTION_TEXTURE_WIDTH,
        motionTextureHeight(this.#nodeCount),
        this.#motion,
      );
      this.#headerWritten = false;
    }
    const pending = this.#pendingBinding;
    if (pending !== undefined) {
      // A new texture per snapshot: its size follows the snapshot's splat count.
      this.#bindingTexture?.destroy();
      this.#bindingTexture = this.#factory.createUintPairs(
        context,
        pending.width,
        pending.height,
        pending.data,
      );
      this.#pendingBinding = undefined;
    } else if (this.#bindingTexture === undefined || this.#bindingTexture.isDestroyed()) {
      this.#bindingTexture = this.#factory.createUintPairs(context, 1, 1, new Uint32Array(2));
    }
    if (this.#flutterTexture === undefined || this.#flutterTexture.isDestroyed()) {
      // A placeholder until an advected field arrives; the shader never samples it before.
      this.#flutterTexture = this.#factory.createFloat(context, 1, 1, new Float32Array(4));
      this.#flutterSource = undefined;
    }
    return true;
  }

  /**
   * Uploads a leaf-flutter texture unless it is the one already bound. Returns the words
   * uploaded: `4·n²` the first time a seed is seen, 0 every frame after. The motion texture is
   * memoised per seed in `@twin/world`, so identity is the right test.
   */
  #uploadFlutterTexture(source: MotionTexture): number {
    const context = this.#context;
    if (context === undefined || this.#flutterSource === source) return 0;
    const packed = packFlutterTexture(source);
    this.#flutterTexture?.destroy();
    this.#flutterTexture = this.#factory.createFloat(context, source.size, source.size, packed);
    this.#flutterSource = source;
    return packed.length;
  }
}
