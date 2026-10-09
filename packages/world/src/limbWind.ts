/**
 * Wind on a **limbs skin** (docs/SCENE_OBJECTS.md §9, "Limbs: today's rig as a skin"): a
 * plant's skin whose handles are its limbs, trunk first (`tools/captures/skin_methods.py`
 * `fit_limbs_from_rig`), swayed by the Living Survey's per-limb model (`living.ts`, ADR 0008)
 * rather than by an object's eigenmodes (`skinWind.ts`).
 *
 * ### What a handle does
 *
 * Handle `j ≥ 1` is limb `j`: each frame it is a rotation `R_j` about the limb's pivot `p_j`
 * (the joint it hangs from), written in the skin's one format,
 *
 * ```text
 * Z_j = [R_j − I | −(R_j − I)(p_j − o)]      so a splat moves by  Σ_j w_j (R_j − I)(x − p_j)
 * ```
 *
 * with `o` the skin's origin: the pivot stays where it is, a splat on the limb swings about
 * it, and a splat on a child limb moves by its own limb's bend plus every bend it hangs from
 * (its weights are non-zero on its limb and its ancestors only). Handle 0, the constant one,
 * stays still: a plant is rooted. `R_j` turns by `gain_j·θ_j`, softly capped at the limb's
 * `limitRad` as each of the rig's joints is capped; `gain_j` is the scale the converter took
 * out of the limb's weights (its largest weight over the plant).
 *
 * ### The sway: today's, limb for limb
 *
 * `θ_j` is the bend per unit gain the rig gives every joint of the limb whose oscillator has
 * the same key (living.ts `livingTransforms`): with `q = ((U/U_ref)(1 + gust))²`,
 *
 * ```text
 * θ_j = q · [ d_j × ŵ + a_j·e1_j + c_j·e2_j ]
 * a_j = 2I·(B_j·u_j(t) + R_j·s_j(t)),  c_j = 0.75I·(B_j·v_j(t) + R_j·s'_j(t))
 * ```
 *
 * `d_j` the limb's chord (attachment to far end), `e1`, `e2` the axes across it, the
 * background `u_j, v_j` the frozen EN 1991-1-4 field read at the limb's own centroid through a
 * low-pass at its frequency, the resonance `s_j` its texture read along its trajectory. The
 * same seed, the same keys, the same textures and the same field as the rig's, so a limbs skin
 * converted from a rig sways, limb by limb, as that rig does (`limbWind.test.ts` holds the two
 * equal on the Minnetonka rig). The wind control means what it means for the rig:
 * `U = speedFromStrength(strength)`, the bearing downwind, the sidecar's gusts.
 *
 * ### Leaf flutter
 *
 * The rig's flutter (`leafFlutter.ts`): an advected field of wavelengths 4–10 leaf sizes, carried
 * downwind at `canopyAdvection·U` (4–9 Hz at the default wind), 6 mm at the reference speed,
 * saturating at twice that. A skin cannot carry the rig's 1024² texture to every renderer, so
 * here each displacement component (along, across, up) is a sum of `LIMB_FLUTTER_WAVES` plane
 * waves on the rig's three lookup planes, their wavelengths at the quantiles of the texture's
 * spectrum, unit RMS: the same band, the same advection, the same amplitude, not the same
 * pattern. A splat's share is the last byte of its row (the Shepard blend of its joints'
 * flutter, as the rig's per-splat amplitude is). The driver writes the frame's waves after the
 * handles (`LIMB_FLUTTER_FLOATS`, rest frame); the renderers fold them into their frames.
 *
 * Pure: a function of the skin, the wind settings and the time.
 */

import {
  branchTrajectory,
  cappedRotation,
  FLUTTER_SATURATION,
  gustEnvelope,
  limbCrossAxes,
  livingDownwind,
  sampleTrajectory,
  speedFromStrength,
  type BranchTrajectory,
  type LivingMotion,
  type LivingWind,
  type Season,
} from "./living";
import { FLUTTER_PLANE_NORMALS } from "./leafFlutter";
import type { GustSettings } from "./motionParams";
import { branchStructure } from "./motionParams";
import {
  branchMotionTexture,
  FLUTTER_MAX_WAVELENGTH_LEAVES,
  FLUTTER_MIN_WAVELENGTH_LEAVES,
  unitUniform,
  type MotionTexture,
} from "./spectral";
import {
  backgroundResponse,
  buffetingFactors,
  frozenTurbulence,
  turbulenceClock,
  turbulencePhases,
  type FrozenTurbulence,
} from "./turbulence";
import { type Vec3 } from "./vec";
import type { WindSettings } from "./wind";

/** Plane waves per flutter component. */
export const LIMB_FLUTTER_WAVES = 4;
/**
 * Numbers a limbs skin's flutter takes after its `12·m` handle numbers, rest frame: a header
 * `(on, byte, waves, 0)`, then per component (along, across, up) and wave `(κx, κy, κz, φ)` —
 * the displacement component is `Σ_k cos(κ·x + φ)` at rest position `x` — then per component
 * the vector it moves a splat of share 1 along, `(x, y, z, 0)`. Sixteen texels in the handle
 * table, after the 32 handles' 97.
 */
export const LIMB_FLUTTER_FLOATS = 4 + 3 * LIMB_FLUTTER_WAVES * 4 + 3 * 4;

/** One limb of a limbs skin: `skin.json`'s `limbs.handles[j − 1]`. */
export interface LimbHandle {
  /** The rig's oscillator key (its base joint): its texture trajectories hash from this. */
  readonly key: number;
  /** The joint it hangs from, rest frame (tileset local ENU, metres): it turns about this. */
  readonly pivot: Vec3;
  /** The handle whose motion carries its pivot (0: none, it hangs from the root). */
  readonly parent: number;
  /** Branching order: 0 the trunk, 1 a limb on it, ... */
  readonly level: number;
  /** How far its subtree reaches from its pivot, metres: its frequency's length. */
  readonly spanM: number;
  readonly frequencyHz: number;
  /** Structural damping ratio, in leaf. */
  readonly damping: number;
  /** The whole-plant mode (its frequency does not change leafless). */
  readonly tree: boolean;
  /** Radians of turn at the splats of weight 1 per unit of `θ` (the rig's gain there). */
  readonly gain: number;
  /** The turn is softly capped here, radians (Infinity: not capped). */
  readonly limitRad: number;
  /** Unit chord, attachment to far end: the limb bends across it. */
  readonly direction: Vec3;
  /** Where it reads the wind: the centroid of the joints it carries. */
  readonly samplePoint: Vec3;
  /** What it carries spans this much across and up, metres (EN 1991-1-4's `b`, `h`). */
  readonly widthM: number;
  readonly heightM: number;
  /** Its own bend's static tip deflection at the reference speed, metres. */
  readonly staticTipM: number;
  /** Its leaves' flutter at the reference speed, metres (0: none). */
  readonly flutterM: number;
}

/** What a limbs skin carries for its driver: `skin.json`'s `limbs`, with the skin's origin. */
export interface LimbSkinSource {
  /** The skin's rest-frame origin: where handle translations are taken about. */
  readonly origin: Vec3;
  /** The motion sidecar's seed: textures, trajectories and the frozen field. */
  readonly seed: number;
  readonly referenceSpeedMps: number;
  readonly leafSizeM: number;
  /** Sway RMS over the mean lean along and across the wind (`2I`, `0.75I`). */
  readonly turbulence: { readonly along: number; readonly across: number };
  readonly lengthScaleM: number;
  readonly gust: GustSettings;
  readonly canopyAdvection: number;
  readonly winter: {
    readonly dampingScale: number;
    readonly branchFrequencyScale: number;
    readonly flutterScale: number;
  };
  /** A flutter share of 1 is this, metres at the reference speed (0: no flutter). */
  readonly flutterReferenceM: number;
  /** Which byte of a splat's row holds its flutter share: the row's last. */
  readonly flutterByte: number;
  /** Per learned handle, trunk first: handle `j` is `handles[j − 1]`. */
  readonly handles: readonly LimbHandle[];
}

interface LimbRuntime {
  readonly frequencyHz: number;
  readonly damping: number;
  readonly texture: MotionTexture;
  readonly along: BranchTrajectory;
  readonly across: BranchTrajectory;
}

/** One flutter wave in the wind frame `(a, c, z)`, per metre, and its phase at `t = 0`. */
interface FlutterWave {
  readonly ka: number;
  readonly kc: number;
  readonly kz: number;
  readonly phase: number;
}

/** A limbs skin's driver state: everything derived once. Immutable after construction. */
export interface LimbWindModel {
  readonly source: LimbSkinSource;
  /** @internal the frozen turbulence field (the sidecar's, by seed). */
  readonly field: FrozenTurbulence;
  /** @internal per season, per limb. */
  readonly seasons: Map<Season, readonly LimbRuntime[]>;
  /** @internal the field's phases at every limb for the last bearing. */
  readonly lastPhases: { bearing: number; phases: readonly Float64Array[] };
  /** @internal per component (along, across, up), `LIMB_FLUTTER_WAVES` waves. */
  readonly waves: readonly FlutterWave[];
}

const FLUTTER_WAVE_SEED = 0x1eaf5;
/** The most a flutter wave leans off the wind in its plane, radians: it is carried, not still. */
const FLUTTER_WAVE_SPREAD = Math.PI / 3;

/**
 * The wavelengths (leaf sizes) of the flutter waves: the `(k + ½)/K` quantiles of the rig's
 * flutter texture's power over its band. Its spectrum falls as `ρ^(−8/3)` over 2D wavenumbers,
 * so the power per logarithmic band goes as `λ^(2/3)`: `λ_k = (λ_lo^(2/3) + u(λ_hi^(2/3) −
 * λ_lo^(2/3)))^(3/2)`, 4.7, 6.0, 7.5 and 9.2 leaf sizes.
 */
export function limbFlutterWavelengths(waves = LIMB_FLUTTER_WAVES): number[] {
  const lo = Math.pow(FLUTTER_MIN_WAVELENGTH_LEAVES, 2 / 3);
  const hi = Math.pow(FLUTTER_MAX_WAVELENGTH_LEAVES, 2 / 3);
  return Array.from({ length: waves }, (_, k) =>
    Math.pow(lo + ((k + 0.5) / waves) * (hi - lo), 1.5),
  );
}

function flutterWaves(seed: number, leafSizeM: number): FlutterWave[] {
  const lengths = limbFlutterWavelengths();
  const out: FlutterWave[] = [];
  FLUTTER_PLANE_NORMALS.forEach((raw, c) => {
    // The rig's lookup plane for this component (leafFlutter.ts): through the downwind axis.
    const length = Math.hypot(raw[0], raw[1], raw[2]);
    const n = [raw[0] / length, raw[1] / length, raw[2] / length] as const;
    const ua = 1 - n[0] * n[0];
    const uc = -n[0] * n[1];
    const uz = -n[0] * n[2];
    const ul = Math.hypot(ua, uc, uz);
    const u = [ua / ul, uc / ul, uz / ul] as const;
    const w = [
      n[1] * u[2] - n[2] * u[1],
      n[2] * u[0] - n[0] * u[2],
      n[0] * u[1] - n[1] * u[0],
    ] as const;
    lengths.forEach((leaves, k) => {
      const counter = c * LIMB_FLUTTER_WAVES + k;
      const angle = (2 * unitUniform(counter, seed ^ FLUTTER_WAVE_SEED) - 1) * FLUTTER_WAVE_SPREAD;
      const wavenumber = (2 * Math.PI) / (leaves * leafSizeM);
      const ca = Math.cos(angle);
      const sa = Math.sin(angle);
      out.push({
        ka: wavenumber * (ca * u[0] + sa * w[0]),
        kc: wavenumber * (ca * u[1] + sa * w[1]),
        kz: wavenumber * (ca * u[2] + sa * w[2]),
        phase: 2 * Math.PI * unitUniform(counter + 64, seed ^ FLUTTER_WAVE_SEED),
      });
    });
  });
  return out;
}

/** The driver of one limbs skin. */
export function limbWindModel(source: LimbSkinSource): LimbWindModel {
  return {
    source,
    field: frozenTurbulence(source.seed),
    seasons: new Map(),
    lastPhases: { bearing: Number.NaN, phases: [] },
    waves: flutterWaves(source.seed, source.leafSizeM),
  };
}

/** The wind a limbs skin sways in for the scene's settings: what the rig's would be. */
export function limbWindFromSettings(
  settings: WindSettings,
  source: Pick<LimbSkinSource, "gust">,
  season: Season = "summer",
): LivingWind {
  return {
    speedMps: speedFromStrength(settings.strength),
    bearingDeg: Number.isFinite(settings.bearingDeg) ? settings.bearingDeg : 0,
    gust: source.gust,
    season,
  };
}

/** Per limb, this season's oscillator: as living.ts `seasonRuntime` builds the rig's. */
function seasonRuntime(model: LimbWindModel, season: Season): readonly LimbRuntime[] {
  const cached = model.seasons.get(season);
  if (cached !== undefined) return cached;
  const { source } = model;
  const winter = source.winter;
  const runtime = source.handles.map((limb): LimbRuntime => {
    const frequencyHz =
      season === "winter" && !limb.tree
        ? limb.frequencyHz * winter.branchFrequencyScale
        : limb.frequencyHz;
    const damping =
      Math.round((season === "winter" ? limb.damping * winter.dampingScale : limb.damping) * 1e4) /
      1e4;
    return {
      frequencyHz,
      damping,
      texture: branchMotionTexture(damping, source.seed),
      along: branchTrajectory(limb.key * 2, source.seed, frequencyHz),
      across: branchTrajectory(limb.key * 2 + 1, source.seed, frequencyHz),
    };
  });
  model.seasons.set(season, runtime);
  return runtime;
}

/** Builds every texture a season needs now, rather than on the first frame. */
export function prepareLimbWind(model: LimbWindModel, season: Season = "summer"): void {
  seasonRuntime(model, season);
}

function limbPhases(model: LimbWindModel, downwind: Vec3): readonly Float64Array[] {
  const bearing = Math.atan2(downwind[0], downwind[1]);
  if (Object.is(model.lastPhases.bearing, bearing)) return model.lastPhases.phases;
  const phases = model.source.handles.map((limb) =>
    turbulencePhases(model.field, limb.samplePoint, downwind, model.source.lengthScaleM),
  );
  model.lastPhases.bearing = bearing;
  model.lastPhases.phases = phases;
  return phases;
}

/** One frame of a limbs skin's sway. */
export interface LimbBends {
  /** Per limb, the bend per unit gain `θ_j` (rotation vector, 3 numbers a limb). */
  readonly theta: Float64Array;
  /** Per limb, its background's along-wind part: the gust its leaves feel. */
  readonly gust: Float64Array;
  /** Per limb, the load factor `((U/U_ref)(1 + gust envelope))²` at its centroid. */
  readonly load: Float64Array;
}

/**
 * Every limb's bend per unit gain at time `t`: living.ts `branchSway` and the per-joint rotation
 * vector of `livingTransforms`, divided by the joint's gain. All zero at calm, by value.
 */
export function limbBends(model: LimbWindModel, t: number, wind: LivingWind): LimbBends {
  const { source } = model;
  const count = source.handles.length;
  const theta = new Float64Array(count * 3);
  const gust = new Float64Array(count);
  const load = new Float64Array(count);
  if (!(wind.speedMps > 0) || !Number.isFinite(wind.speedMps)) return { theta, gust, load };
  const time = Number.isFinite(t) ? t : 0;
  const runtime = seasonRuntime(model, wind.season);
  const w = livingDownwind(wind.bearingDeg);
  const clock = turbulenceClock(model.field, source.lengthScaleM, wind.speedMps, time);
  const phases = limbPhases(model, w);
  const ratio = wind.speedMps / source.referenceSpeedMps;
  const background = new Float64Array(2);
  for (let j = 0; j < count; j += 1) {
    const limb = source.handles[j];
    const branch = runtime[j];
    if (limb === undefined || branch === undefined) continue;
    const factors = buffetingFactors(
      branch.frequencyHz,
      branch.damping,
      limb.widthM,
      limb.heightM,
      limb.staticTipM * ratio * ratio,
      wind.speedMps,
      source.lengthScaleM,
    );
    const b = Math.sqrt(factors.background2);
    const r = Math.sqrt(factors.resonance2);
    background[0] = 0;
    background[1] = 0;
    const at = phases[j];
    if (at !== undefined) backgroundResponse(at, clock, branch.frequencyHz, background);
    const gusting = source.turbulence.along * b * (background[0] ?? 0);
    const along =
      gusting + source.turbulence.along * r * sampleTrajectory(branch.texture, branch.along, time);
    const across =
      source.turbulence.across *
      (b * (background[1] ?? 0) + r * sampleTrajectory(branch.texture, branch.across, time));
    // The gust envelope is convected at the mean speed: read at the limb's centroid.
    const p = limb.samplePoint;
    const delay = (p[0] * w[0] + p[1] * w[1]) / wind.speedMps;
    const u = ratio * (1 + gustEnvelope(wind.gust, time - delay, source.seed));
    const q = u * u;
    const d = limb.direction;
    const [e1, e2] = limbCrossAxes(d, w);
    // Mean lean: d × ŵ, whose length is the sine between limb and wind.
    const lean: Vec3 = [
      d[1] * w[2] - d[2] * w[1],
      d[2] * w[0] - d[0] * w[2],
      d[0] * w[1] - d[1] * w[0],
    ];
    for (let c = 0; c < 3; c += 1)
      theta[j * 3 + c] = q * ((lean[c] ?? 0) + along * (e1[c] ?? 0) + across * (e2[c] ?? 0));
    gust[j] = gusting;
    load[j] = q;
  }
  return { theta, gust, load };
}

/** The rotation by `r`, softly capped at `limit` as the rig's joints are; uncapped when none. */
function turn(r: Vec3, limit: number): readonly [number, number, number, number] {
  if (Number.isFinite(limit) && limit > 0) return cappedRotation(r, limit);
  const angle = Math.hypot(r[0], r[1], r[2]);
  if (angle === 0) return [0, 0, 0, 1];
  const s = Math.sin(angle / 2) / angle;
  return [r[0] * s, r[1] * s, r[2] * s, Math.cos(angle / 2)];
}

/** Numbers `limbHandles` writes for a skin of `handles` handles: `12·m` plus its flutter. */
export function limbHandleFloats(handles: number): number {
  return handles * 12 + LIMB_FLUTTER_FLOATS;
}

/**
 * The skin's handles at time `t` under `wind` (`12·m` numbers, `Z_j` row-major, rest frame,
 * about the skin's origin; handle 0 at rest), then the frame's flutter (`LIMB_FLUTTER_FLOATS`).
 * Calm gives zeros throughout, flutter off.
 */
export function limbHandles(
  model: LimbWindModel,
  t: number,
  wind: LivingWind,
  out?: Float64Array,
): Float64Array {
  const { source } = model;
  const m = source.handles.length + 1;
  const target = out ?? new Float64Array(limbHandleFloats(m));
  target.fill(0, 0, Math.min(target.length, limbHandleFloats(m)));
  if (!(wind.speedMps > 0) || !Number.isFinite(wind.speedMps)) return target;
  const bends = limbBends(model, t, wind);
  const o = source.origin;
  source.handles.forEach((limb, j) => {
    const angle: Vec3 = [
      limb.gain * (bends.theta[j * 3] ?? 0),
      limb.gain * (bends.theta[j * 3 + 1] ?? 0),
      limb.gain * (bends.theta[j * 3 + 2] ?? 0),
    ];
    const [x, y, z, w] = turn(angle, limb.limitRad);
    // R − I from the quaternion, so a small turn keeps its precision.
    const a = [
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
    const px = limb.pivot[0] - o[0];
    const py = limb.pivot[1] - o[1];
    const pz = limb.pivot[2] - o[2];
    const at = (j + 1) * 12;
    for (let r = 0; r < 3; r += 1) {
      const a0 = a[r * 3] ?? 0;
      const a1 = a[r * 3 + 1] ?? 0;
      const a2 = a[r * 3 + 2] ?? 0;
      target[at + r * 4] = a0;
      target[at + r * 4 + 1] = a1;
      target[at + r * 4 + 2] = a2;
      // The pivot stays: t = −(R − I)(p − o).
      target[at + r * 4 + 3] = -(a0 * px + a1 * py + a2 * pz);
    }
  });
  writeFlutter(model, t, wind, bends, target, m * 12);
  return target;
}

/** The frame's flutter waves and amplitude into `out` from `at` (see LIMB_FLUTTER_FLOATS). */
function writeFlutter(
  model: LimbWindModel,
  t: number,
  wind: LivingWind,
  bends: LimbBends,
  out: Float64Array,
  at: number,
): void {
  const { source } = model;
  if (out.length < at + LIMB_FLUTTER_FLOATS) return;
  const seasonScale = wind.season === "winter" ? source.winter.flutterScale : 1;
  const reference = source.flutterReferenceM * seasonScale;
  if (!(reference > 0)) return;
  // The leaves feel their limbs' gusts (living.ts `livingFlutter`): one factor for the skin,
  // the mean over its leafy limbs, at the load of its trunk.
  let gusts = 0;
  let leafy = 0;
  source.handles.forEach((limb, j) => {
    if (!(limb.flutterM > 0)) return;
    gusts += Math.max(0, 1 + (bends.gust[j] ?? 0));
    leafy += 1;
  });
  const gust = leafy > 0 ? gusts / leafy : 1;
  const load = bends.load[0] ?? 0;
  // softLimit(ref·q·gust, 2·ref) = 2·ref·tanh(q·gust / 2): linear in the reference, so a
  // splat's share scales it exactly as the rig's per-splat blend of joint amplitudes does.
  const amplitude = FLUTTER_SATURATION * reference * Math.tanh((load * gust) / FLUTTER_SATURATION);
  if (!(amplitude > 0)) return;
  const w = livingDownwind(wind.bearingDeg);
  const ex = w[0];
  const ey = w[1];
  const advection = wind.speedMps * source.canopyAdvection * (Number.isFinite(t) ? t : 0);
  out[at] = 1;
  out[at + 1] = source.flutterByte;
  out[at + 2] = LIMB_FLUTTER_WAVES;
  out[at + 3] = 0;
  model.waves.forEach((wave, k) => {
    const o = at + 4 + k * 4;
    // Wind frame (a, c, z) to the rest frame: a along (ex, ey), c along (ey, −ex).
    out[o] = wave.ka * ex + wave.kc * ey;
    out[o + 1] = wave.ka * ey - wave.kc * ex;
    out[o + 2] = wave.kz;
    // Carried downwind: the field at x and t is the field at x − advection·ŵ and 0.
    const phase = wave.phase - wave.ka * advection;
    out[o + 3] = phase - 2 * Math.PI * Math.floor(phase / (2 * Math.PI));
  });
  // Unit RMS per component: K cosines of random phase, √(2/K) each.
  const scale = amplitude * Math.sqrt(2 / LIMB_FLUTTER_WAVES);
  const directions: readonly Vec3[] = [
    [ex, ey, 0],
    [ey, -ex, 0],
    [0, 0, 1],
  ];
  directions.forEach((d, c) => {
    const o = at + 4 + 3 * LIMB_FLUTTER_WAVES * 4 + c * 4;
    out[o] = scale * d[0];
    out[o + 1] = scale * d[1];
    out[o + 2] = scale * d[2];
    out[o + 3] = 0;
  });
}

/**
 * A splat's flutter offset (rest frame) from the numbers `limbHandles` wrote at `at`, for a
 * splat of flutter share `share` at rest position `x`: what the shaders compute.
 */
export function limbFlutterOffset(
  flutter: ArrayLike<number>,
  at: number,
  share: number,
  x: Vec3,
): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  if (!((flutter[at] ?? 0) > 0.5) || !(share > 0)) return out;
  const waves = Math.round(flutter[at + 2] ?? 0);
  for (let c = 0; c < 3; c += 1) {
    let v = 0;
    for (let k = 0; k < waves; k += 1) {
      const o = at + 4 + (c * waves + k) * 4;
      v += Math.cos(
        (flutter[o] ?? 0) * x[0] +
          (flutter[o + 1] ?? 0) * x[1] +
          (flutter[o + 2] ?? 0) * x[2] +
          (flutter[o + 3] ?? 0),
      );
    }
    const o = at + 4 + 3 * waves * 4 + c * 4;
    out[0] += share * v * (flutter[o] ?? 0);
    out[1] += share * v * (flutter[o + 1] ?? 0);
    out[2] += share * v * (flutter[o + 2] ?? 0);
  }
  return out;
}

/**
 * A rig's limbs as a limbs skin's handles would carry them, from its bound motion (living.ts):
 * every oscillator whose joints bend, trunk first, at `gain` 1 and uncapped — the twin of
 * `skin_methods.limb_rig` without the weights, for tests and for a driver built straight from a
 * rig.
 */
export function limbHandlesFromRig(motion: LivingMotion): LimbHandle[] {
  const { rig, sidecar } = motion;
  const structure = branchStructure(rig);
  const nodes = rig.nodes;
  const bases = new Set<number>();
  sidecar.nodes.branch.forEach((base, i) => {
    if (i > 0 && (nodes[i]?.parent ?? -1) >= 0 && (sidecar.nodes.gainRad[i] ?? 0) !== 0)
      bases.add(base);
  });
  const tree = structure.treeBranch;
  const ordered = [...bases].sort((a, b) => (a === tree ? -1 : b === tree ? 1 : a - b));
  const column = new Map(ordered.map((b, j) => [b, j] as const));
  return ordered.map((base): LimbHandle => {
    const attach = nodes[base]?.parent ?? 0;
    const carriedBy = attach > 0 ? column.get(sidecar.nodes.branch[attach] ?? -1) : undefined;
    const geometry = motion.oscillators.get(base);
    let flutterM = 0;
    sidecar.nodes.branch.forEach((b, i) => {
      if (b === base) flutterM = Math.max(flutterM, sidecar.nodes.flutterM[i] ?? 0);
    });
    return {
      key: base,
      pivot: nodes[attach]?.position ?? [0, 0, 0],
      parent: carriedBy === undefined ? 0 : carriedBy + 1,
      level: structure.order[base] ?? 0,
      spanM: structure.branchLengthM.get(base) ?? 0,
      frequencyHz: sidecar.nodes.frequencyHz[base] ?? 1,
      damping: sidecar.nodes.damping[base] ?? 0.1,
      tree: sidecar.nodes.mode[base] === 0,
      gain: 1,
      limitRad: Number.POSITIVE_INFINITY,
      direction: motion.limbDirection[base] ?? [0, 0, 1],
      samplePoint: geometry?.samplePoint ?? [0, 0, 0],
      widthM: geometry?.widthM ?? 0,
      heightM: geometry?.heightM ?? 0,
      staticTipM: geometry?.staticTipM ?? 0,
      flutterM,
    };
  });
}

/** A limbs skin source straight from a bound rig (gains 1): see {@link limbHandlesFromRig}. */
export function limbSourceFromRig(
  motion: LivingMotion,
  origin: Vec3 = [0, 0, 0],
  flutterByte = 31,
): LimbSkinSource {
  const { sidecar } = motion;
  let flutter = 0;
  for (const value of sidecar.nodes.flutterM) flutter = Math.max(flutter, value);
  return {
    origin,
    seed: sidecar.seed,
    referenceSpeedMps: sidecar.referenceSpeedMps,
    leafSizeM: sidecar.leafSizeM,
    turbulence: sidecar.wind.turbulence,
    lengthScaleM: motion.lengthScaleM,
    gust: sidecar.wind.gust,
    canopyAdvection: sidecar.wind.canopyAdvection,
    winter: sidecar.seasons.winter,
    flutterReferenceM: flutter,
    flutterByte,
    handles: limbHandlesFromRig(motion),
  };
}
