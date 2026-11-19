import { prepareWeatherWind, type WeatherWind } from "@twin/world";
import {
  ARTICULATION_GLSL,
  weatherArticulationTier,
  weatherArticulationGround,
} from "./livingArticulation";
import { SPECTRA_GLSL } from "./livingSpectra";

/**
 * The living lab under PlayCanvas (state/livingLab.ts): water and wind drawn every frame by the
 * final-draw hook, `gsplatModifyVS` on the user material -- the place the sunlight draws from
 * (sunHook.ts) -- so nothing is copied into the work buffer again and a frame costs only the
 * hook's arithmetic.
 *
 * The hook wraps whatever hook is in already (the sun's, the comparison's, or PlayCanvas's
 * empty one): their three functions are renamed and called first, then the lab adds its part.
 *
 * PlayCanvas calls `modifySplatCenter` before it reads the splat's colour, rotation and scales
 * (gsplat.js: `initSource`, `getCenter`, the hook, then `initCorner`), so the hook reads the
 * work buffer's own streams there (`loadDataColor`, `loadDataTransformA/B`: the packed format
 * `dataFormat = "large"` has, containerPackedRead.js) to know what a splat is:
 *
 * - **water**: inside the lab's box, within its thickness of the level;
 * - **plant**: green enough (excess green over brightness) and above the ground -- a coarse
 *   ground grid the back-end builds from the tiles' own splats (the 10th percentile of z in
 *   each cell, `groundGrid`), or a flat ground where it has no splats to read (a streamed scan).
 *
 * Gust/clump/voxel arms move centres only. The articulated arm also rotates covariance
 * through a cached branch binding. Sorting still uses centres at rest for every lab arm.
 *
 * GLSL only, as the sunlight: the WebGPU trial does not draw it.
 */

/** Cells a side of the ground grid. */
export const GROUND_CELLS = 64;

/** The lab's uniforms, set every frame (`livingUniforms`). */
export interface LivingUniforms {
  t: Float32Array;
  wind: Float32Array;
  water: Float32Array;
  waterBox: Float32Array;
  sun: Float32Array;
  groundGrid: Float32Array;
  veg: Float32Array;
}

export function livingUniforms(): LivingUniforms {
  return {
    t: new Float32Array(4),
    wind: new Float32Array(4),
    water: new Float32Array(4),
    waterBox: new Float32Array(4),
    sun: new Float32Array(4),
    groundGrid: new Float32Array(4),
    veg: new Float32Array(4),
  };
}

const LIVING_GLSL = `
${SPECTRA_GLSL}
uniform vec4 uLivingT;
uniform vec4 uLivingWind;
uniform vec4 uLivingWater;
uniform vec4 uLivingWaterBox;
uniform vec4 uLivingSun;
uniform vec4 uLivingGroundGrid;
uniform vec4 uLivingVeg;
uniform highp sampler2D uLivingGround;
${ARTICULATION_GLSL}
float livingWaterW = 0.0;
vec3 livingWaterN = vec3(0.0, 0.0, 1.0);
float livingFoam = 0.5;
float livingVegW = 0.0;
float livingShimmer = 0.0;
float livingHash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float livingNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(livingHash(i), livingHash(i + vec2(1.0, 0.0)), u.x),
               mix(livingHash(i + vec2(0.0, 1.0)), livingHash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float livingFbm(vec2 p) {
    return 0.5 * livingNoise(p) + 0.3 * livingNoise(p * 2.03 + 7.1) + 0.2 * livingNoise(p * 4.01 + 3.7);
}
vec4 livingColor() {
#ifdef GSPLAT_COLOR_FLOAT
    return loadDataColor();
#else
    uvec4 p = loadDataColor();
    return vec4(unpackHalf2x16(p.r | (p.g << 16u)), unpackHalf2x16(p.b | (p.a << 16u)));
#endif
}
vec3 livingScale() {
    uvec2 b = loadDataTransformB().xy;
    return vec3(unpackHalf2x16(b.x).y, unpackHalf2x16(b.y));
}
float livingGroundZ(vec2 xy) {
    if (uLivingGroundGrid.w < 0.5) return uLivingVeg.w;
    int n = int(uLivingGroundGrid.w);
    vec2 g = (xy - uLivingGroundGrid.xy) / uLivingGroundGrid.z - 0.5;
    vec2 i = floor(g);
    vec2 f = clamp(g - i, 0.0, 1.0);
    ivec2 a = clamp(ivec2(i), ivec2(0), ivec2(n - 1));
    ivec2 b = clamp(ivec2(i) + 1, ivec2(0), ivec2(n - 1));
    float z00 = texelFetch(uLivingGround, ivec2(a.x, a.y), 0).r;
    float z10 = texelFetch(uLivingGround, ivec2(b.x, a.y), 0).r;
    float z01 = texelFetch(uLivingGround, ivec2(a.x, b.y), 0).r;
    float z11 = texelFetch(uLivingGround, ivec2(b.x, b.y), 0).r;
    return mix(mix(z00, z10, f.x), mix(z01, z11, f.x), f.y);
}
// Gust strength 0.3-1 at a point: patches about 8 m across drifting downwind.
float livingGust(float along, float across, float t) {
    float g = livingFbm(vec2(along - uLivingWind.z * t, across * 0.7) * 0.12);
    return mix(0.3, 1.0, smoothstep(0.3, 0.7, g));
}
// Ripples: (height, dh/dx, dh/dy) of four short waves about the flow, deep-water speeds.
vec3 livingWaves(vec2 x, vec2 flow, float t) {
    vec3 o = vec3(0.0);
    for (int k = 0; k < 4; k++) {
        float fk = float(k);
        float lambda = 0.9 * pow(0.62, fk);
        float ang = (fk - 1.5) * 0.55 + 0.3 * sin(fk * 2.1);
        vec2 d = vec2(flow.x * cos(ang) - flow.y * sin(ang), flow.x * sin(ang) + flow.y * cos(ang));
        float kw = 6.2832 / lambda;
        float a = 0.0065 * lambda;
        float ph = kw * dot(d, x) - sqrt(9.81 * kw) * t + fk * 1.7;
        o += vec3(a * sin(ph), a * kw * cos(ph) * d);
    }
    // Chop: a noise slope on top.
    float e = 0.05;
    float n0 = livingNoise(x * 4.0);
    o.yz += 0.08 * vec2(livingNoise((x + vec2(e, 0.0)) * 4.0) - n0, livingNoise((x + vec2(0.0, e)) * 4.0) - n0) / e * 0.25;
    return o;
}
float livingStreak(vec2 x, vec2 flow) {
    return livingFbm(vec2(dot(x, flow) * 0.7, dot(x, vec2(-flow.y, flow.x)) * 4.0));
}
// Arm 1, the gust field: a wave train downwind plus the gusts' lean.
vec3 livingGustArm(vec3 p, float h, float amp, vec2 wd, vec2 perp, float t, float jitter) {
    float along = dot(p.xy, wd);
    float G = livingGust(along, dot(p.xy, perp), t);
    float wave = sin(6.2832 * (along - uLivingWind.z * t) / 4.0);
    float lean = amp * G * (0.55 + 0.45 * wave);
    vec3 d = vec3(wd * lean, -0.25 * lean * lean / max(h, 0.1));
    d.xy += perp * amp * 0.15 * G * sin(6.2832 * 0.9 * t + jitter * 6.2832 + along * 0.7);
    return d;
}
// Arm 2, resonant clumps: each 1.5 m cell an oscillator forced by the gust at its centre,
// blended over the four nearest cells so a crown never tears at a cell edge; branch flutter.
vec3 livingClumpArm(vec3 p, float h, float amp, vec2 wd, vec2 perp, float t) {
    vec2 q = p.xy / 1.5 - 0.5;
    vec2 i = floor(q);
    vec2 f = q - i;
    f = f * f * (3.0 - 2.0 * f);
    vec2 sum = vec2(0.0);
    for (int k = 0; k < 4; k++) {
        vec2 o = vec2(float(k & 1), float(k >> 1));
        vec2 cell = i + o;
        float r = livingHash(cell);
        float r2 = livingHash(cell + 17.0);
        vec2 cc = (cell + 0.5) * 1.5;
        float G = livingGust(dot(cc, wd), dot(cc, perp), t - 0.4);
        float fr = (0.35 + 0.45 * r) / sqrt(1.0 + 0.3 * h);
        float ph = 6.2832 * (fr * t + r2);
        vec2 m = wd * (0.5 * G + 0.5 * G * sin(ph)) + perp * 0.35 * G * sin(1.37 * ph + 1.3);
        float w = (o.x > 0.5 ? f.x : 1.0 - f.x) * (o.y > 0.5 ? f.y : 1.0 - f.y);
        sum += m * w;
    }
    vec3 d = vec3(sum * amp, 0.0);
    float b = livingNoise(p.xy * 3.0 + p.z * 1.7 + 9.0);
    d.xy += (wd * 0.6 + perp) * amp * 0.18 * sin(6.2832 * (1.6 + 1.4 * b) * t + b * 20.0);
    return d;
}
void modifySplatCenter(inout vec3 center) {
    hexapodLivingInnerCenter(center);
    livingWaterW = 0.0;
    livingVegW = 0.0;
    livingShimmer = 0.0;
    float t = uLivingT.x;
    if (uLivingT.y > 0.5) {
        // A flooded hollow: inside the ellipse, splats from just above the level down to
        // the depth below it are the water, drawn as a level surface; the shore follows the ground.
        float r = length((center.xy - uLivingWaterBox.xy) / max(uLivingWaterBox.zw, vec2(0.01)));
        float inside = 1.0 - smoothstep(0.85, 1.0, r);
        float below = uLivingWater.x - center.z;
        float w = inside * smoothstep(-0.02, 0.01, below) * (1.0 - smoothstep(uLivingWater.y * 0.8, uLivingWater.y, below));
        if (w > 0.0) {
            vec2 flow = vec2(cos(uLivingWater.w), sin(uLivingWater.w));
            float period = 4.0;
            float p0 = fract(t / period);
            float p1 = fract(t / period + 0.5);
            float w0 = 1.0 - abs(1.0 - 2.0 * p0);
            vec2 adv = flow * uLivingWater.z * period;
            vec2 x0 = center.xy - adv * (p0 - 0.5);
            vec2 x1 = center.xy - adv * (p1 - 0.5) + vec2(3.7, 1.3);
            vec3 h = mix(livingWaves(x1, flow, t), livingWaves(x0, flow, t), w0);
            livingWaterN = normalize(vec3(-h.y, -h.z, 1.0));
            livingFoam = mix(livingStreak(x1, flow), livingStreak(x0, flow), w0);
            center.z = mix(center.z, uLivingWater.x + h.x, w);
            livingWaterW = w;
        }
    }
    if (uLivingT.z > 4.5) {
        vec3 c=livingColor().rgb;
        float exg=(2.0*c.g-c.r-c.b)/(c.r+c.g+c.b+0.001);
        float veg=uLivingVeg.x < -0.5 ? 1.0 : smoothstep(uLivingVeg.x,uLivingVeg.x+0.08,exg);
        float h=max(0.0,center.z-livingGroundZ(center.xy));
        veg*=smoothstep(0.035,0.15,h);
        vec2 wd=uLivingWind.xy, perp=vec2(-wd.y,wd.x);
        float a=6.28318530718*(dot(center.xy,wd)-uLivingWind.z*t)/uLivingWaterBox.x;
        float b=6.28318530718*dot(center.xy,perp)/uLivingWaterBox.y;
        float wave=0.6*sin(a+0.27*sin(b)+uLivingWaterBox.w)+0.4*sin(1.618*a-0.43*b+1.7+uLivingWaterBox.w);
        float ratio=log((h+0.3)/0.3)/log(10.3/0.3)*exp(2.0*min(0.0,h/6.0-1.0));
        float u=max(0.0,uLivingWind.z+uLivingWaterBox.z*wave)*ratio;
        float load=u*u/sqrt(1.0+u*u/16.0);
        float angle=0.10*load/(28.0+load);
        // Smooth height weighting keeps terrain exact; no per-splat CPU work.
        center.xy+=veg*uLivingT.w*h*angle*(wd+0.14*perp*sin(0.73*a+b+0.9+uLivingWaterBox.w));
        livingVegW=veg;
        return;
    }
    if (uLivingT.z > 3.5) {
        livingArtCenter(center);
        return;
    }
    if (uLivingT.z > 2.5) {
        center += livingSpectrum(center);
        return;
    }
    if (uLivingT.z > 0.5 && livingWaterW < 0.5) {
        vec3 c = livingColor().rgb;
        float exg = (2.0 * c.g - c.r - c.b) / (c.r + c.g + c.b + 0.001);
        float veg = uLivingVeg.x < -0.5 ? 1.0 : smoothstep(uLivingVeg.x, uLivingVeg.x + 0.08, exg);
        if (veg > 0.0) {
            float h = center.z - livingGroundZ(center.xy);
            veg *= smoothstep(uLivingVeg.y, uLivingVeg.y + 0.05, h);
            if (veg > 0.0) {
                vec2 wd = uLivingWind.xy;
                vec2 perp = vec2(-wd.y, wd.x);
                float amp = uLivingT.w * (0.12 * min(h, 0.5) + 0.03 * max(h - 0.5, 0.0));
                float jitter = livingHash(center.xy * 37.0 + center.z * 11.0);
                vec3 d = uLivingT.z < 1.5
                    ? livingGustArm(center, h, amp, wd, perp, t, jitter)
                    : livingClumpArm(center, h, amp, wd, perp, t);
                // Leaf shimmer: small splats flick quickly, and catch the light as they do.
                vec3 s = livingScale();
                float small = 1.0 - smoothstep(0.015, 0.05, max(s.x, max(s.y, s.z)));
                float G = livingGust(dot(center.xy, wd), dot(center.xy, perp), t);
                float sh = sin(6.2832 * (2.0 + 3.0 * jitter) * t + jitter * 40.0);
                float leaf = small * G * smoothstep(0.1, 0.6, h);
                d += vec3(perp * 0.004, 0.003) * sh * leaf * uLivingT.w;
                livingShimmer = leaf * sh;
                center += d * veg;
                livingVegW = veg;
            }
        }
    }
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
    hexapodLivingInnerRotationScale(originalCenter, modifiedCenter, rotation, scale);
    if (uLivingT.z > 3.5) livingArtCovariance(rotation, scale);
    // Water lies flat: a thin disc on the surface, its long axis along the flow. The rotation
    // here is (w, x, y, z) (containerPackedRead.js, quatToMat3).
    if (livingWaterW > 0.0) {
        float a = max(scale.x, max(scale.y, scale.z));
        float c = min(scale.x, min(scale.y, scale.z));
        float b = scale.x + scale.y + scale.z - a - c;
        float halfAngle = 0.5 * uLivingWater.w;
        vec4 level = vec4(cos(halfAngle), 0.0, 0.0, sin(halfAngle));
        float w = smoothstep(0.3, 0.9, livingWaterW);
        rotation = normalize(mix(rotation, dot(rotation, level) < 0.0 ? -level : level, w));
        scale = mix(scale, vec3(a, b, min(c, 0.004)), w);
    }
}
void modifySplatColor(vec3 center, inout vec4 color) {
    hexapodLivingInnerColor(center, color);
    if (uLivingT.z > 3.5) color.rgb *= livingArtLight;
    if (uLivingWind.w > 0.5) {
        if (livingWaterW > 0.0) color.rgb = mix(color.rgb, vec3(0.1, 0.4, 1.0), 0.75 * livingWaterW);
        if (livingVegW > 0.0) color.rgb = mix(color.rgb, vec3(1.0, 0.15, 0.8), 0.65 * livingVegW);
        return;
    }
    if (livingWaterW > 0.0) {
        vec3 v = normalize(center - view_position);
        vec3 n = livingWaterN;
        vec3 r = reflect(v, n);
        float cosi = max(dot(-v, n), 0.0);
        float fres = 0.02 + 0.98 * pow(1.0 - cosi, 5.0);
        // Sky: pale at the horizon, deep overhead; the ripples tilt the reflected ray across it.
        vec3 sky = mix(vec3(0.74, 0.80, 0.86), vec3(0.22, 0.36, 0.60), sqrt(clamp(r.z, 0.0, 1.0)));
        float glint = pow(max(dot(r, uLivingSun.xyz), 0.0), 900.0) * 3.0 * uLivingSun.w;
        // The bed seen through shallow water: the captured ground, wet (darker), and the streaks
        // the flow carries.
        vec3 body = color.rgb * 0.5 * (0.85 + 0.45 * (livingFoam - 0.5)) + vec3(0.01, 0.02, 0.03);
        vec3 water = mix(body, sky, clamp(0.08 + fres * 0.9, 0.0, 0.9)) + vec3(glint);
        color.rgb = mix(color.rgb, water, livingWaterW);
    }
    if (livingVegW > 0.0) color.rgb *= 1.0 + 0.08 * livingShimmer * livingVegW;
}
`;

/** PlayCanvas's own empty hook, for when no other hook is in. */
const EMPTY_HOOK = `
void modifySplatCenter(inout vec3 center) {
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
}
void modifySplatColor(vec3 center, inout vec4 color) {
}
`;

/** The lab's hook around `inner` (the hook that would be in without it; none: PlayCanvas's). */
export function livingHookVs(inner: string | undefined): string {
  const own = (inner ?? EMPTY_HOOK)
    .replace("void modifySplatCenter(", "void hexapodLivingInnerCenter(")
    .replace("void modifySplatRotationScale(", "void hexapodLivingInnerRotationScale(")
    .replace("void modifySplatColor(", "void hexapodLivingInnerColor(");
  const eye = /uniform\s+vec3\s+view_position\s*;/.test(own) ? "" : "uniform vec3 view_position;\n";
  return `${own}\n${eye}${LIVING_GLSL}`;
}

/** The lab's knobs, as the hook's uniforms read them. */
export interface LivingSettings {
  t: number;
  water: boolean;
  motion: "off" | "gust" | "clumps" | "spectra" | "articulated";
  strength: number;
  windBearingDeg: number;
  gustSpeed: number;
  greenness: number;
  mask: boolean;
  waterLevel: number;
  waterThickness: number;
  waterBox: readonly [number, number, number, number];
  flowBearingDeg: number;
  flowSpeed: number;
  /** Unit direction to the sun (scan frame) and how far up it is (0-1). */
  sunDir: readonly [number, number, number];
  sunUp: number;
  flatGround: number;
  articulationUrl?: string;
  weather?: WeatherWind | null;
}

const RAD = Math.PI / 180;
let weatherTime = 0;
export function labWeatherClock(): number {
  return weatherTime;
}

/** Fills `u` from `s`. A bearing is clockwise from north: east is +x, north +y. */
export function packLiving(u: LivingUniforms, s: LivingSettings, ground: GroundGrid | null): void {
  weatherTime = s.t;
  const arm =
    s.motion === "gust"
      ? 1
      : s.motion === "clumps"
        ? 2
        : s.motion === "spectra"
          ? 3
          : s.motion === "articulated"
            ? 4
            : 0;
  u.t.set([s.t, s.water ? 1 : 0, arm, s.strength]);
  const wb = s.windBearingDeg * RAD;
  u.wind.set([Math.sin(wb), Math.cos(wb), s.gustSpeed, s.mask ? 1 : 0]);
  // The flow's angle from +x, counter-clockwise, as the shader's `vec2(cos, sin)` takes it.
  const fb = (90 - s.flowBearingDeg) * RAD;
  u.water.set([s.waterLevel, s.waterThickness, s.flowSpeed, fb]);
  u.waterBox.set(s.waterBox);
  u.sun.set([s.sunDir[0], s.sunDir[1], s.sunDir[2], s.sunUp]);
  u.groundGrid.set(ground ? [ground.x0, ground.y0, ground.cell, ground.cells] : [0, 0, 1, 0]);
  u.veg.set([
    s.greenness,
    0.03,
    /[?&]weather=/.test(s.articulationUrl ?? "")
      ? weatherArticulationTier(s.articulationUrl ?? "") === "C"
        ? 1
        : weatherArticulationTier(s.articulationUrl ?? "") === "A"
          ? -2
          : -1
      : /[?&]breeze=(calm|breezy|windy)(?:&|$)/.test(s.articulationUrl ?? "")
        ? 1
        : 0,
    weatherArticulationGround(s.articulationUrl ?? "") ?? s.flatGround,
  ]);
  if (s.weather && s.motion === "gust") {
    const f = prepareWeatherWind(s.weather);
    u.t[2] = 5;
    u.t[1] = 0;
    u.wind.set([f.east, f.north, f.meanMps, s.mask ? 1 : 0]);
    u.waterBox.set([f.alongM, f.crossM, f.gustMps - f.meanMps, f.phase]);
  }
}

/** The direction to a sun at `azimuthDeg` (from north, clockwise) and `elevationDeg`. */
export function sunDirection(azimuthDeg: number, elevationDeg: number): [number, number, number] {
  const az = azimuthDeg * RAD;
  const el = elevationDeg * RAD;
  return [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)];
}

/** The ground under a scan, coarse: the 10th percentile of z in each cell of a square grid. */
export interface GroundGrid {
  x0: number;
  y0: number;
  cell: number;
  cells: number;
  /** `cells × cells` heights, row by row from y0 (R32F). */
  z: Float32Array;
}

/** Where a tile's splats are, for the ground grid. */
export interface GroundSample {
  count: number;
  at(i: number, out: [number, number, number]): void;
}

/**
 * The ground grid from the splats of `samples`, at most `budget` of them read (evenly strided).
 * A cell's ground is the lowest cell's within `reachM`. Empty cells take their nearest filled neighbour's height (grown a ring at a time); null when
 * there are no splats.
 */
export function groundGrid(
  samples: readonly GroundSample[],
  cells = GROUND_CELLS,
  budget = 400_000,
  reachM = 4,
): GroundGrid | null {
  const total = samples.reduce((n, s) => n + s.count, 0);
  if (total === 0) return null;
  const stride = Math.max(1, Math.floor(total / budget));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const p: [number, number, number] = [0, 0, 0];
  for (const s of samples) {
    for (let i = 0; i < s.count; i += stride) {
      s.at(i, p);
      if (!Number.isFinite(p[0] + p[1] + p[2])) continue;
      xs.push(p[0]);
      ys.push(p[1]);
      zs.push(p[2]);
    }
  }
  if (xs.length === 0) return null;
  // The bounds of the middle 98%: a few floaters far out do not stretch the grid.
  const sorted = (a: number[]) => Float64Array.from(a).sort();
  const sx = sorted(xs);
  const sy = sorted(ys);
  const q = (a: Float64Array, f: number) =>
    a[Math.min(a.length - 1, Math.floor(f * a.length))] ?? 0;
  const x0 = q(sx, 0.01);
  const y0 = q(sy, 0.01);
  const span = Math.max(q(sx, 0.99) - x0, q(sy, 0.99) - y0, 0.5);
  const cell = span / cells;
  const buckets: number[][] = Array.from({ length: cells * cells }, () => []);
  for (let i = 0; i < xs.length; i += 1) {
    const cx = Math.floor(((xs[i] ?? 0) - x0) / cell);
    const cy = Math.floor(((ys[i] ?? 0) - y0) / cell);
    if (cx < 0 || cy < 0 || cx >= cells || cy >= cells) continue;
    buckets[cy * cells + cx]?.push(zs[i] ?? 0);
  }
  const z = new Float32Array(cells * cells).fill(Number.NaN);
  buckets.forEach((b, i) => {
    if (b.length < 4) return;
    b.sort((a, c) => a - c);
    z[i] = b[Math.floor(b.length * 0.1)] ?? Number.NaN;
  });
  // Under a crown a cell holds no ground, only the crown's underside: each cell takes the lowest
  // cell within `reachM` (terrain rises little over a few metres; a crown floats well above it).
  const reach = Math.max(1, Math.round(reachM / cell));
  const lowest = Float32Array.from(z);
  for (let y = 0; y < cells; y += 1) {
    for (let x = 0; x < cells; x += 1) {
      if (Number.isNaN(z[y * cells + x] ?? 0)) continue;
      let low = Number.POSITIVE_INFINITY;
      for (let dy = -reach; dy <= reach; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= cells) continue;
        for (let dx = -reach; dx <= reach; dx += 1) {
          const nx = x + dx;
          if (nx < 0 || nx >= cells || dx * dx + dy * dy > reach * reach) continue;
          const v = z[ny * cells + nx] ?? Number.NaN;
          if (v < low) low = v;
        }
      }
      lowest[y * cells + x] = low;
    }
  }
  z.set(lowest);
  // Grow filled cells into empty ones, a ring at a time.
  for (let pass = 0; pass < cells; pass += 1) {
    let empty = 0;
    const next = Float32Array.from(z);
    for (let y = 0; y < cells; y += 1) {
      for (let x = 0; x < cells; x += 1) {
        if (!Number.isNaN(z[y * cells + x] ?? 0)) continue;
        let sum = 0;
        let n = 0;
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ] as const) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= cells || ny >= cells) continue;
          const v = z[ny * cells + nx] ?? Number.NaN;
          if (!Number.isNaN(v)) {
            sum += v;
            n += 1;
          }
        }
        if (n > 0) next[y * cells + x] = sum / n;
        else empty += 1;
      }
    }
    z.set(next);
    if (empty === 0) break;
  }
  const fallback = q(sorted(zs), 0.1);
  for (let i = 0; i < z.length; i += 1) if (Number.isNaN(z[i] ?? 0)) z[i] = fallback;
  return { x0, y0, cell, cells, z };
}
