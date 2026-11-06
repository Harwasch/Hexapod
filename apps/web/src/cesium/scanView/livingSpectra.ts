/** Sparse Fourier playback, independent of PlayCanvas. See docs/LIVING_SPECTRA.md. */
export interface SpectraHeader {
  format: "hexapod.living-spectra";
  version: 1;
  encoding: "float16-le";
  motionEvidence: "simulated";
  binary: string;
  origin: [number, number, number];
  size: [number, number, number];
  resolution: number;
  occupied: number;
  frequenciesHz: number[];
  referenceBearingDeg: number;
  coefficientOffset: number;
  byteLength: number;
}

export interface LivingSpectra {
  header: SpectraHeader;
  /** Dense x-fastest lattice -> sparse row + 1; zero is empty. */
  index: Float32Array;
  /** voxel, frequency, xyz, real/imaginary (decoded once). */
  coefficients: Float32Array;
}

export const SPECTRA_INDEX_WIDTH = 2048;
export const SPECTRA_DISPLACEMENT_WIDTH = 256;

export function parseSpectraHeader(value: unknown): SpectraHeader {
  const h = value as Partial<SpectraHeader> | null;
  const triple = (v: unknown): boolean =>
    Array.isArray(v) &&
    v.length === 3 &&
    v.every((n: unknown) => typeof n === "number" && Number.isFinite(n));
  if (
    h?.format !== "hexapod.living-spectra" ||
    h.version !== 1 ||
    h.encoding !== "float16-le" ||
    h.motionEvidence !== "simulated" ||
    typeof h.binary !== "string" ||
    !h.binary ||
    !triple(h.origin) ||
    !triple(h.size) ||
    !h.size?.every((n) => n > 0) ||
    !Number.isInteger(h.resolution) ||
    (h.resolution ?? 0) < 2 ||
    (h.resolution ?? 0) > 128 ||
    !Number.isInteger(h.occupied) ||
    (h.occupied ?? 0) < 1 ||
    (h.occupied ?? 0) > 50_000 ||
    !Array.isArray(h.frequenciesHz) ||
    h.frequenciesHz.length < 1 ||
    h.frequenciesHz.length > 16 ||
    !h.frequenciesHz.every(
      (f, i, a) => Number.isFinite(f) && f >= 0 && (i === 0 || f > (a[i - 1] ?? -1)),
    ) ||
    !Number.isFinite(h.referenceBearingDeg) ||
    h.coefficientOffset !== (h.occupied ?? 0) * 4 ||
    h.byteLength !== (h.occupied ?? 0) * (4 + 12 * h.frequenciesHz.length)
  ) {
    throw new Error("Invalid or unsupported living spectra header (CPU limit: 50,000 voxels)");
  }
  return h as SpectraHeader;
}

function half(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) throw new Error("Non-finite spectra coefficient");
  return (
    sign * (exponent === 0 ? fraction * 2 ** -24 : (1 + fraction / 1024) * 2 ** (exponent - 15))
  );
}

export function decodeSpectra(header: SpectraHeader, binary: ArrayBuffer): LivingSpectra {
  const h = parseSpectraHeader(header);
  if (binary.byteLength !== h.byteLength) throw new Error("Spectra binary length mismatch");
  const view = new DataView(binary);
  const cells = h.resolution ** 3;
  const index = new Float32Array(Math.ceil(cells / SPECTRA_INDEX_WIDTH) * SPECTRA_INDEX_WIDTH);
  for (let i = 0; i < h.occupied; i++) {
    const cell = view.getUint32(i * 4, true);
    if (cell >= cells || index[cell] !== 0) throw new Error("Invalid or duplicate spectra voxel");
    index[cell] = i + 1;
  }
  const coefficients = new Float32Array(h.occupied * h.frequenciesHz.length * 6);
  for (let i = 0; i < coefficients.length; i++)
    coefficients[i] = half(view.getUint16(h.coefficientOffset + i * 2, true));
  return { header: h, index, coefficients };
}

export async function loadSpectra(url: string, signal?: AbortSignal): Promise<LivingSpectra> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Spectra HTTP ${String(response.status)}`);
  const h = parseSpectraHeader(await response.json());
  const data = await fetch(new URL(h.binary, new URL(url, window.location.href)), { signal });
  if (!data.ok) throw new Error(`Spectra binary HTTP ${String(data.status)}`);
  return decodeSpectra(h, await data.arrayBuffer());
}

/** One RGBA texel per occupied voxel. Trig is K times per frame, never per voxel or splat. */
export function evaluateSpectra(
  spectra: LivingSpectra,
  t: number,
  strength: number,
  bearing: number,
  out: Float32Array,
): void {
  const { header: h, coefficients: c } = spectra;
  if (out.length < h.occupied * 4) throw new Error("Displacement buffer too small");
  const k = h.frequenciesHz.length;
  const cos = h.frequenciesHz.map((f) => Math.cos(2 * Math.PI * f * t));
  const sin = h.frequenciesHz.map((f) => Math.sin(2 * Math.PI * f * t));
  const angle = ((h.referenceBearingDeg - bearing) * Math.PI) / 180;
  const ca = Math.cos(angle) * strength;
  const sa = Math.sin(angle) * strength;
  for (let v = 0; v < h.occupied; v++) {
    let x = 0,
      y = 0,
      z = 0;
    for (let f = 0; f < k; f++) {
      const o = (v * k + f) * 6;
      const re = cos[f] ?? 0,
        im = sin[f] ?? 0;
      x += (c[o] ?? 0) * re - (c[o + 1] ?? 0) * im;
      y += (c[o + 2] ?? 0) * re - (c[o + 3] ?? 0) * im;
      z += (c[o + 4] ?? 0) * re - (c[o + 5] ?? 0) * im;
    }
    out[v * 4] = ca * x - sa * y;
    out[v * 4 + 1] = sa * x + ca * y;
    out[v * 4 + 2] = z * strength;
    out[v * 4 + 3] = 1;
  }
}

/** Eight lattice corners; missing samples are zero (no nearest-branch extrapolation). */
export const SPECTRA_GLSL = `
uniform highp sampler2D uSpectraIndex;
uniform highp sampler2D uSpectraDisplacement;
uniform vec4 uSpectraOrigin; // xyz origin, w resolution (zero until loaded)
uniform vec3 uSpectraStep; // lattice spacing, not voxel count
vec3 livingSpectrum(vec3 p) {
    int n = int(uSpectraOrigin.w);
    if (n < 2) return vec3(0.0);
    vec3 q = (p - uSpectraOrigin.xyz) / uSpectraStep;
    if (any(lessThan(q, vec3(0.0))) || any(greaterThanEqual(q, vec3(float(n - 1))))) return vec3(0.0);
    ivec3 base = ivec3(floor(q));
    vec3 f = fract(q);
    vec3 sum = vec3(0.0);
    for (int j = 0; j < 8; j++) {
        ivec3 corner = ivec3(j & 1, (j >> 1) & 1, (j >> 2) & 1);
        ivec3 cell = base + corner;
        int flatIndex = cell.x + n * (cell.y + n * cell.z);
        int row = int(texelFetch(uSpectraIndex, ivec2(flatIndex % 2048, flatIndex / 2048), 0).r) - 1;
        if (row >= 0) {
            vec3 w = mix(vec3(1.0) - f, f, vec3(corner));
            sum += texelFetch(uSpectraDisplacement, ivec2(row % 256, row / 256), 0).xyz * w.x * w.y * w.z;
        }
    }
    return sum;
}
`;
