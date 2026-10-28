/** Joint rotation spectra + immutable per-splat bindings. Lab only; all motion simulated. */
export interface ArticulationNode {
  parent: number;
  band?: string;
  stiffness?: number;
  position: [number, number, number];
}
export interface ArticulationHeader {
  format: "hexapod.living-articulation";
  version: 1;
  motionEvidence: "simulated";
  nodes: ArticulationNode[];
  frequenciesHz: number[];
  referenceBearingDeg: number;
  motionBinary: string;
  leafBinary: string;
  bindingBinary: string;
  leafCount: number;
  tiles: Record<string, { offset: number; count: number }>;
}
export interface Articulation {
  header: ArticulationHeader;
  coefficients: Float32Array;
  /** pivot xyz, axis xyz, Hz, phase, amplitude radians. */
  leaves: Float32Array;
  bindings: Uint32Array;
  breeze?: BreezeModel;
}
export const ARTICULATION_STREAM = "hexapodArticulation";
export const ARTICULATION_WIDTH = 1024;
export function parseArticulationHeader(value: unknown): ArticulationHeader {
  const h = value as Partial<ArticulationHeader> | null;
  if (
    h?.format !== "hexapod.living-articulation" ||
    h.version !== 1 ||
    h.motionEvidence !== "simulated" ||
    !Array.isArray(h.nodes) ||
    h.nodes.length < 1 ||
    h.nodes.length > 1024 ||
    !h.nodes.every(
      (n, i) =>
        Number.isInteger(n.parent) &&
        n.parent >= -1 &&
        n.parent < i &&
        Array.isArray(n.position) &&
        n.position.length === 3 &&
        n.position.every(Number.isFinite),
    ) ||
    !Array.isArray(h.frequenciesHz) ||
    h.frequenciesHz.length < 1 ||
    h.frequenciesHz.length > 16 ||
    !h.frequenciesHz.every(
      (f, i, a) => Number.isFinite(f) && f >= 0 && (i === 0 || f > (a[i - 1] ?? -1)),
    ) ||
    !Number.isFinite(h.referenceBearingDeg) ||
    !Number.isInteger(h.leafCount) ||
    (h.leafCount ?? -1) < 0 ||
    (h.leafCount ?? 0) > 500_000 ||
    typeof h.motionBinary !== "string" ||
    typeof h.leafBinary !== "string" ||
    typeof h.bindingBinary !== "string" ||
    !h.tiles ||
    typeof h.tiles !== "object" ||
    !Object.values(h.tiles).every(
      (t) =>
        Number.isSafeInteger(t.offset) &&
        t.offset >= 0 &&
        t.offset % 16 === 0 &&
        Number.isSafeInteger(t.count) &&
        t.count > 0 &&
        t.count <= 10_000_000,
    )
  )
    throw new Error("Invalid articulated spectra header");
  return h as ArticulationHeader;
}
export function decodeArticulation(
  header: ArticulationHeader,
  motion: ArrayBuffer,
  leaf: ArrayBuffer,
  binding: ArrayBuffer,
): Articulation {
  const h = parseArticulationHeader(header);
  const size = Math.max(0, ...Object.values(h.tiles).map((t) => t.offset + t.count * 16));
  if (
    motion.byteLength !== h.nodes.length * h.frequenciesHz.length * 24 ||
    leaf.byteLength !== h.leafCount * 36 ||
    binding.byteLength !== size
  )
    throw new Error("Articulation binary length mismatch");
  const coefficients = new Float32Array(motion),
    leaves = new Float32Array(leaf),
    bindings = new Uint32Array(binding);
  if (!coefficients.every(Number.isFinite) || !leaves.every(Number.isFinite))
    throw new Error("Non-finite articulation data");
  for (let i = 0; i < bindings.length; i += 4) {
    if (
      (bindings[i] ?? 0) > h.nodes.length ||
      (bindings[i + 1] ?? 0) > h.nodes.length ||
      (bindings[i + 2] ?? 0) > 65535 ||
      (bindings[i + 3] ?? 0) > h.leafCount
    )
      throw new Error("Out-of-range articulated binding");
  }
  for (let i = 0; i < h.leafCount; i++) {
    const o = i * 9,
      axis = Math.hypot(leaves[o + 3] ?? 0, leaves[o + 4] ?? 0, leaves[o + 5] ?? 0);
    if (
      Math.abs(axis - 1) > 1e-5 ||
      (leaves[o + 6] ?? 0) < 2 ||
      (leaves[o + 6] ?? 0) > 6 ||
      (leaves[o + 8] ?? 0) < 0 ||
      (leaves[o + 8] ?? 0) > 0.15
    )
      throw new Error("Invalid leaf flutter parameters");
  }
  return { header: h, coefficients, leaves, bindings };
}
export async function loadArticulation(url: string, signal?: AbortSignal): Promise<Articulation> {
  const get = async (at: string) => {
    const r = await fetch(at, { signal });
    if (!r.ok) throw new Error(`Articulation HTTP ${String(r.status)}`);
    return r;
  };
  const h = parseArticulationHeader(await (await get(url)).json());
  const buffers = await Promise.all(
    [h.motionBinary, h.leafBinary, h.bindingBinary].map(async (name) =>
      (await get(new URL(name, new URL(url, window.location.href)).toString())).arrayBuffer(),
    ),
  );
  const [motion, leaf, binding] = buffers;
  if (!motion || !leaf || !binding) throw new Error("Missing articulation payload");
  const a = decodeArticulation(h, motion, leaf, binding);
  const preset = new URL(url, window.location.href).searchParams.get("breeze");
  if (preset === "calm" || preset === "breezy" || preset === "windy")
    a.breeze = prepareBreeze(a, preset);
  return a;
}
/** Reject unmatched tiles rather than applying a scan's binding to unrelated geometry. */
export function orderedArticulationBinding(
  a: Articulation | null,
  checksum: string,
  order: Uint32Array,
  out: Uint32Array,
): boolean {
  out.fill(0);
  const entry = a?.header.tiles[checksum];
  if (!a || entry?.count !== order.length) return false;
  if (out.length < order.length * 4) throw new Error("Binding output too small");
  for (let i = 0; i < order.length; i++) {
    const source = order[i] ?? 0;
    if (source >= entry.count) throw new Error("Invalid resource order");
    const from = entry.offset / 4 + source * 4;
    for (let c = 0; c < 4; c++) out[i * 4 + c] = a.bindings[from + c] ?? 0;
  }
  return true;
}
export type Quat = [number, number, number, number]; // PlayCanvas hook: XYZW, not WXYZ
export type Point = [number, number, number];
export function multiplyQuaternion(a: Quat, b: Quat): Quat {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}
export function rotatePoint(q: Quat, p: Point): Point {
  const x = 2 * (q[1] * p[2] - q[2] * p[1]),
    y = 2 * (q[2] * p[0] - q[0] * p[2]),
    z = 2 * (q[0] * p[1] - q[1] * p[0]);
  return [
    p[0] + q[3] * x + q[1] * z - q[2] * y,
    p[1] + q[3] * y + q[2] * x - q[0] * z,
    p[2] + q[3] * z + q[0] * y - q[1] * x,
  ];
}
export function articulationPixels(count: number): Float32Array<ArrayBuffer> {
  return new Float32Array(
    Math.max(1, Math.ceil(((count + 1) * 2) / ARTICULATION_WIDTH)) * ARTICULATION_WIDTH * 4,
  );
}
/** Two texels per node: global quaternion then translation. Zero row is identity. */
export function evaluateArticulation(
  a: Articulation,
  t: number,
  strength: number,
  bearing: number,
  poses: Float32Array,
  flutter: Float32Array,
): void {
  const h = a.header,
    k = h.frequenciesHz.length;
  const cos = (a.breeze ? [] : h.frequenciesHz).map((f) => Math.cos(2 * Math.PI * f * t)),
    sin = (a.breeze ? [] : h.frequenciesHz).map((f) => Math.sin(2 * Math.PI * f * t));
  const angle = ((h.referenceBearingDeg - bearing) * Math.PI) / 180,
    ca = Math.cos(angle),
    sa = Math.sin(angle);
  const leafSin = a.breeze?.frequenciesHz.map((f) => Math.sin(2 * Math.PI * f * t));
  const leafCos = a.breeze?.frequenciesHz.map((f) => Math.cos(2 * Math.PI * f * t));
  poses.fill(0, 0, 8);
  poses[3] = 1;
  flutter.fill(0, 0, 8);
  flutter[7] = 1;
  for (const [i, node] of h.nodes.entries()) {
    const v: Point = [0, 0, 0];
    if (a.breeze) {
      const yaw = breezeAngle(a.breeze, i, t, bearing) * strength;
      v[0] = yaw * 0.28;
      v[1] = yaw;
    } else
      for (let f = 0; f < k; f++)
        for (let c = 0; c < 3; c++) {
          const o = (i * k + f) * 6 + c * 2;
          v[c] =
            (v[c] ?? 0) +
            ((a.coefficients[o] ?? 0) * (cos[f] ?? 0) -
              (a.coefficients[o + 1] ?? 0) * (sin[f] ?? 0)) *
              strength;
        }
    const x = ca * v[0] - sa * v[1],
      y = sa * v[0] + ca * v[1],
      z = v[2],
      length = Math.hypot(x, y, z);
    const s = length < 1e-12 ? 0.5 : Math.sin(length / 2) / length;
    const local: Quat = [x * s, y * s, z * s, Math.cos(length / 2)];
    const p = node.parent,
      po = (p + 1) * 8;
    const pq: Quat = [poses[po] ?? 0, poses[po + 1] ?? 0, poses[po + 2] ?? 0, poses[po + 3] ?? 1];
    const q = multiplyQuaternion(pq, local);
    const joint = p < 0 ? node.position : (h.nodes[p]?.position ?? node.position);
    const parentJoint = rotatePoint(pq, joint),
      turned = rotatePoint(q, joint);
    const o = (i + 1) * 8;
    poses.set(q, o);
    for (let c = 0; c < 3; c++)
      poses[o + 4 + c] = (parentJoint[c] ?? 0) + (poses[po + 4 + c] ?? 0) - (turned[c] ?? 0);
  }
  for (let i = 0; i < h.leafCount; i++) {
    const l = i * 9,
      o = (i + 1) * 8;
    flutter[o] = a.leaves[l] ?? 0;
    flutter[o + 1] = a.leaves[l + 1] ?? 0;
    flutter[o + 2] = a.leaves[l + 2] ?? 0;
    const basis = a.breeze?.flutter;
    const fi = basis?.[i * 3] ?? 0;
    const wave = basis
      ? (leafSin?.[fi] ?? 0) * (basis[i * 3 + 1] ?? 0) +
        (leafCos?.[fi] ?? 0) * (basis[i * 3 + 2] ?? 0)
      : Math.sin(2 * Math.PI * (a.leaves[l + 6] ?? 0) * t + (a.leaves[l + 7] ?? 0));
    const half =
      0.5 *
      strength *
      (a.breeze ? BREEZE_PRESETS[a.breeze.preset].leaf : 1) *
      (a.leaves[l + 8] ?? 0) *
      wave;
    // At |half| <= .15 (all presets through strength 3), Taylor remainders are
    // <= .15^7/5040 for sin and .15^6/720 for cos. No machine-calibrated tolerance.
    const fast = a.breeze && Math.abs(half) <= 0.15;
    const half2 = half * half;
    const s = fast ? half * (1 - half2 / 6 + (half2 * half2) / 120) : Math.sin(half),
      x = a.leaves[l + 3] ?? 0,
      y = a.leaves[l + 4] ?? 0;
    flutter[o + 4] = (ca * x - sa * y) * s;
    flutter[o + 5] = (sa * x + ca * y) * s;
    flutter[o + 6] = (a.leaves[l + 5] ?? 0) * s;
    flutter[o + 7] = fast ? 1 - half2 / 2 + (half2 * half2) / 24 : Math.cos(half);
    if (a.breeze) {
      // Compose once per packet, rather than twice per splat in the vertex hook.
      const po = (a.breeze.owners[i] ?? 0) * 8;
      const px = poses[po] ?? 0,
        py = poses[po + 1] ?? 0,
        pz = poses[po + 2] ?? 0,
        pw = poses[po + 3] ?? 1;
      const lx = flutter[o + 4] ?? 0,
        ly = flutter[o + 5] ?? 0,
        lz = flutter[o + 6] ?? 0,
        lw = flutter[o + 7] ?? 1;
      const qx = pw * lx + px * lw + py * lz - pz * ly,
        qy = pw * ly - px * lz + py * lw + pz * lx,
        qz = pw * lz + px * ly - py * lx + pz * lw,
        qw = pw * lw - px * lx - py * ly - pz * lz;
      const vx = a.leaves[l] ?? 0,
        vy = a.leaves[l + 1] ?? 0,
        vz = a.leaves[l + 2] ?? 0;
      const ax = 2 * (py * vz - pz * vy),
        ay = 2 * (pz * vx - px * vz),
        az = 2 * (px * vy - py * vx);
      const bx = 2 * (qy * vz - qz * vy),
        by = 2 * (qz * vx - qx * vz),
        bz = 2 * (qx * vy - qy * vx);
      flutter[o] = pw * ax + py * az - pz * ay - qw * bx - qy * bz + qz * by + (poses[po + 4] ?? 0);
      flutter[o + 1] =
        pw * ay + pz * ax - px * az - qw * by - qz * bx + qx * bz + (poses[po + 5] ?? 0);
      flutter[o + 2] =
        pw * az + px * ay - py * ax - qw * bz - qx * by + qy * bx + (poses[po + 6] ?? 0);
      flutter[o + 4] = qx;
      flutter[o + 5] = qy;
      flutter[o + 6] = qz;
      flutter[o + 7] = qw;
    }
  }
}

/** Literature bands and the explicit artistic amplitude limits are documented in LIVING_SPECTRA. */
export const BREEZE_PRESETS = {
  calm: { angle: 0.006, leaf: 0.35, speed: 1.5 },
  breezy: { angle: 0.014, leaf: 0.65, speed: 2.5 },
  windy: { angle: 0.028, leaf: 1, speed: 4 },
} as const;
export type BreezePreset = keyof typeof BREEZE_PRESETS;
export interface BreezeModel {
  preset: BreezePreset;
  /** Prepared once: local angular cap, modal Hz, phase, xy position. */
  branches: Float64Array;
  owners: Uint32Array;
  frequenciesHz: number[];
  /** Frequency index, cos phase, sin phase, prepared once. */
  flutter: Float64Array;
}
export function prepareBreeze(a: Articulation, preset: BreezePreset): BreezeModel {
  const branches = new Float64Array(a.header.nodes.length * 5);
  const order: number[] = [];
  for (const [i, n] of a.header.nodes.entries()) {
    const parent = a.header.nodes[n.parent];
    const depth = parent ? (order[n.parent] ?? 0) + (n.band === "trunk" ? 0 : 1) : 0;
    order[i] = depth;
    const length = parent
      ? Math.hypot(...n.position.map((v, c) => v - (parent.position[c] ?? 0)))
      : 0;
    // The rig stiffness is a geometric prior, not measured EI. Bound its response.
    const compliance =
      Math.min(1, Math.max(0.15, length / 0.7)) / Math.sqrt(Math.max(1, n.stiffness ?? 1));
    const cap =
      !parent || n.band === "trunk"
        ? 0
        : (BREEZE_PRESETS[preset].angle * compliance) / (1 + 0.45 * Math.max(0, depth - 1));
    const seed = ((i + 1) * 0.61803398875) % 1;
    branches.set(
      [
        cap,
        Math.min(1.1, 0.24 + 0.11 * depth + 0.12 * seed),
        seed * 2 * Math.PI,
        n.position[0],
        n.position[1],
      ],
      i * 5,
    );
  }
  const owners = new Uint32Array(a.header.leafCount);
  for (let j = 0; j < a.bindings.length; j += 4) {
    const packet = a.bindings[j + 3] ?? 0;
    if (packet > 0) owners[packet - 1] = a.bindings[j] ?? 0;
  }
  const frequenciesHz: number[] = [];
  const flutter = new Float64Array(a.header.leafCount * 3);
  for (let i = 0; i < a.header.leafCount; i++) {
    const f = a.leaves[i * 9 + 6] ?? 0;
    let index = frequenciesHz.indexOf(f);
    if (index < 0) {
      index = frequenciesHz.length;
      frequenciesHz.push(f);
    }
    const phase = a.leaves[i * 9 + 7] ?? 0;
    flutter.set([index, Math.cos(phase), Math.sin(phase)], i * 3);
  }
  return { preset, branches, owners, frequenciesHz, flutter };
}
/** Continuous bounded forcing: shared 20/31 s gusts, spatial delay, small independent resonance. */
export function breezeAngle(model: BreezeModel, node: number, t: number, bearing = 90): number {
  const o = node * 5,
    b = model.branches;
  const cap = b[o] ?? 0;
  if (cap === 0) return 0;
  const phase = b[o + 2] ?? 0;
  const tau =
    t -
    ((b[o + 3] ?? 0) * Math.sin((bearing * Math.PI) / 180) +
      (b[o + 4] ?? 0) * Math.cos((bearing * Math.PI) / 180)) /
      BREEZE_PRESETS[model.preset].speed;
  const gust =
    0.55 +
    0.25 * Math.sin((2 * Math.PI * tau) / 20) +
    0.2 * Math.sin((2 * Math.PI * tau) / 31 + 1.7);
  return (
    cap *
    (0.65 * gust +
      0.25 * Math.sin(2 * Math.PI * (b[o + 1] ?? 0) * t + phase) +
      0.1 * Math.sin(2 * Math.PI * 0.19 * tau + phase * 0.15))
  );
}

/** Cached bindings are copied through the work buffer. No colour/branch search in this hook. */
export const ARTICULATION_GLSL = `
uniform highp sampler2D uArticulationPoses;
uniform highp sampler2D uArticulationLeaves;
uniform float uArticulationReady;
vec4 livingArtRotation = vec4(0.0,0.0,0.0,1.0);
float livingArtLeaf = 0.0;
float livingArtActive = 0.0;
float livingArtLight = 1.0;
vec4 livingArtRead(sampler2D t, uint i) {
    return texelFetch(t,ivec2(int(i)%1024,int(i)/1024),0);
}
vec4 livingArtMul(vec4 a, vec4 b) {
    return vec4(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));
}
vec3 livingArtRotate(vec4 q,vec3 p) {
    return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);
}
void livingArtCenter(inout vec3 p) {
    livingArtRotation=vec4(0.0,0.0,0.0,1.0); livingArtLeaf=0.0; livingArtLight=1.0; livingArtActive=0.0;
#ifdef HEXAPOD_ARTICULATION
    if(uArticulationReady<0.5 || uLivingT.w==0.0) return;
    uvec4 b=loadHexapodArticulation();
    if(b.x==0u) return;
    livingArtActive=1.0;
    if (b.w>0u && uLivingVeg.z>0.5) {
        livingArtRotation=livingArtRead(uArticulationLeaves,b.w*2u+1u);
        p=livingArtRotate(livingArtRotation,p)+livingArtRead(uArticulationLeaves,b.w*2u).xyz;
        return;
    }
    vec4 q=livingArtRead(uArticulationPoses,b.x*2u);
    vec3 shift=livingArtRead(uArticulationPoses,b.x*2u+1u).xyz;
    if(b.z<65535u) {
        float w=float(b.z)/65535.0;
        vec4 q2=livingArtRead(uArticulationPoses,b.y*2u);
        q=normalize(mix(dot(q,q2)<0.0?-q2:q2,q,w));
        shift=mix(livingArtRead(uArticulationPoses,b.y*2u+1u).xyz,shift,w);
    }
    if(b.w>0u) {
        vec3 pivot=livingArtRead(uArticulationLeaves,b.w*2u).xyz;
        vec4 leaf=livingArtRead(uArticulationLeaves,b.w*2u+1u);
        p=pivot+livingArtRotate(leaf,p-pivot);
        livingArtRotation=livingArtMul(q,leaf);
        livingArtLeaf=1.0;
    } else livingArtRotation=q;
    p=livingArtRotate(q,p)+shift;
#endif
}
void livingArtCovariance(inout vec4 rotation,vec3 scale) {
    if(livingArtActive<0.5) return;
    vec3 axis=scale.x<=scale.y && scale.x<=scale.z?vec3(1,0,0):scale.y<=scale.z?vec3(0,1,0):vec3(0,0,1);
    vec3 before=livingArtRotate(rotation,axis);
    rotation=normalize(livingArtMul(livingArtRotation,rotation));
    if(livingArtLeaf>0.5) {
        vec3 after=livingArtRotate(rotation,axis);
        // Relative two-sided Lambert change; preserve the photograph at zero motion.
        livingArtLight=(0.55+0.45*abs(dot(after,uLivingSun.xyz)))/(0.55+0.45*abs(dot(before,uLivingSun.xyz)));
    }
}
`;
