/**
 * Living Mode, phase 1: stateless modal wind animation of a rigged tree from its skeleton alone.
 *
 * Every rig joint belongs to one **branch** (see `branchStructure`), and every branch is one damped
 * harmonic oscillator — the trunk's is the whole-tree pendulum mode. Parent motion is carried to
 * children as `deform` does — each joint's local rotation, then its parent's transform — with one
 * difference: joint `i` is a hinge at its **parent's** rest position, so its rotation bends the
 * segment `parent → i` and everything beyond it.
 *
 * What a branch does is its buffeting response to the wind, in two parts (`turbulence.ts`,
 * after EN 1991-1-4 Annex B):
 *
 * ```text
 * q        = (U/U_ref)² · (1 + gust(t − x_i/U))²       drag: mean deflection ∝ U² (no gust by default)
 * u_b(t)   = LowPass_fb[ T(x_b − U·t·ŵ) ]              background: the frozen field at the branch,
 *                                                     through its static compliance up to f_b
 * s_b(t)   = T_ζb(p0_b + v_b·t),  |v_b| = f_b·λ        resonance: a narrow band at f_b (texture)
 * a_b      = 2I · (B_b·u_b(t) + R_b(U)·s_b(t))          along-wind sway, EN 1991-1-4 eq. 6.3
 * c_b      = 0.75I · (B_b·v_b(t) + R_b(U)·s'_b(t))      across-wind sway
 * r_i(t)   = g_i · q · [ d_i × ŵ + a_b·e1_i + c_b·e2_i ]
 * ```
 *
 * `T` is a frozen turbulence field whose spectrum along the wind is EN eq. B.2, advected at the
 * mean speed; `B²` (eq. B.3) and `R²` (eq. B.6, with the aerodynamic damping of F.18) weigh its
 * slow part against the resonance. The spectrum's energy sits at `f ≈ 0.15·U/L`, `L` ≈ 35 m for a
 * 6 m tree, so at 2 m/s a branch follows the gusts almost quasi-statically — slowly, in step with
 * its neighbours, a gust reaching the downwind side `Δx/U` later — and its resonance is a few per
 * cent of the motion; the resonance grows with the wind, as `S_L(f_b)` and the admittances do.
 *
 * `r_i` is the joint's rotation vector: a mean lean about `d_i × ŵ` (the limb's own direction —
 * its chord, attachment to far end, shared by all its joints so it bends in one plane — crossed
 * with downwind, so a limb pointing downwind is not bent by it — Habel eqs. 18–19) plus
 * turbulent sway about the two axes across the limb. `g_i` is the joint's share of its branch's
 * bend.
 *
 * **Pure.** The output is a function of `(t, wind, sidecar seed, rig)` and nothing else. The
 * textures, trajectories and field are memoised, which is not observable. At `U = 0` every
 * rotation is the identity *by value*, so calm restores the canonical bytes exactly.
 */

import { IDENTITY_TRANSFORM, type NodeTransform } from "./deform";
import { FLUTTER_STILL } from "./flutter";
import { advectedFlutterUnitBound, flutterLookups, type AdvectedFlutterField } from "./leafFlutter";
import {
  branchStructure,
  parseMotionSidecar,
  validateMotionSidecar,
  type GustSettings,
  type MotionSidecar,
} from "./motionParams";
import { hash32 } from "./noise";
import { nodeAngleLimit, type MotionRig } from "./rig";
import {
  BRANCH_TEXELS_PER_CYCLE,
  BRANCH_TEXTURE_SIZE,
  branchMotionTexture,
  FLUTTER_MAX_WAVELENGTH_LEAVES,
  FLUTTER_MIN_WAVELENGTH_LEAVES,
  FLUTTER_TEXTURE_SIZE,
  flutterTexture,
  sampleTexture,
  trajectoryDirection,
  unitUniform,
  type MotionTexture,
} from "./spectral";
import {
  backgroundResponse,
  BACKGROUND_SOFT_CLIP,
  buffetingFactors,
  frozenTurbulence,
  turbulenceLengthScaleM,
  turbulenceClock,
  turbulencePhases,
  type BuffetingFactors,
  type FrozenTurbulence,
} from "./turbulence";
import {
  add,
  quatMultiply,
  quatRotate,
  softLimit,
  subtract,
  DEG_TO_RAD,
  type Quat,
  type Vec3,
} from "./vec";
import { type WindSettings } from "./wind";

export type Season = "summer" | "winter";

/** The wind a Living Mode frame is evaluated under. Physical units, unlike `WindSettings`. */
export interface LivingWind {
  /** Mean wind speed, m/s. `0` is calm and restores the measured pose exactly. */
  readonly speedMps: number;
  /** Downwind bearing, degrees clockwise from north. */
  readonly bearingDeg: number;
  readonly gust: GustSettings;
  readonly season: Season;
}

/**
 * Mean speed at the top of the scene's dimensionless wind control, m/s (Beaufort 8).
 *
 * The control's `strength` is read as **dynamic pressure**: `U = 20·√strength`, so deflection —
 * which goes as `U²` — is linear in the control. The default strength of 0.1 is then 6.3 m/s,
 * a moderate breeze. The amplitude at that speed is still not validated against any real tree;
 * only the scaling law is.
 */
export const WIND_FULL_SCALE_MPS = 20;

export function speedFromStrength(strength: number): number {
  const s = Number.isFinite(strength) ? Math.min(1, Math.max(0, strength)) : 0;
  return s === 0 ? 0 : WIND_FULL_SCALE_MPS * Math.sqrt(s);
}

/** The scene's `WindSettings`, read through a sidecar's gust defaults. */
export function livingWindFromSettings(
  settings: WindSettings,
  sidecar: MotionSidecar,
  season: Season = "summer",
): LivingWind {
  return {
    speedMps: speedFromStrength(settings.strength),
    bearingDeg: Number.isFinite(settings.bearingDeg) ? settings.bearingDeg : 0,
    gust: sidecar.wind.gust,
    season,
  };
}

/** No gusts: a steady mean speed. For tests that want the turbulence alone. */
export const NO_GUSTS: GustSettings = {
  strength: 0,
  variance: 0,
  frequencyPerMin: 1,
  durationS: 1,
};

// ---------------------------------------------------------------------------------------------
// Gusts
// ---------------------------------------------------------------------------------------------

const GUST_SEED = 0x6057;

/**
 * The gust envelope at time `t`: the fractional increase of the mean speed, `≥ 0`.
 *
 * SpeedTree-style: gusts arrive `frequencyPerMin` times a minute, last about `durationS`, and
 * peak at about `strength`, each varying by `±variance`. Time is cut into slots one gust-period
 * long; each slot holds exactly one gust whose start, length and peak come from a hash of the
 * slot number, and which ends inside its slot, so the value needs only the current slot. A
 * `sin²` envelope keeps the speed smooth.
 */
export function gustEnvelope(gust: GustSettings, t: number, seed: number): number {
  if (!(gust.strength > 0) || !(gust.frequencyPerMin > 0) || !Number.isFinite(t)) return 0;
  const period = 60 / gust.frequencyPerMin;
  const slot = Math.floor(t / period);
  const variance = Math.min(1, Math.max(0, gust.variance));
  const key = hash32(slot, seed ^ GUST_SEED);
  const u1 = unitUniform(key, 1);
  const u2 = unitUniform(key, 2);
  const u3 = unitUniform(key, 3);
  const duration = Math.min(period, Math.max(0.2, gust.durationS * (1 + variance * (2 * u1 - 1))));
  const start = slot * period + u2 * (period - duration);
  const local = t - start;
  if (local <= 0 || local >= duration) return 0;
  const peak = gust.strength * (1 + variance * (2 * u3 - 1));
  const s = Math.sin((Math.PI * local) / duration);
  return peak * s * s;
}

/** The largest value {@link gustEnvelope} can return. */
export function maxGust(gust: GustSettings): number {
  if (!(gust.strength > 0)) return 0;
  return gust.strength * (1 + Math.min(1, Math.max(0, gust.variance)));
}

// ---------------------------------------------------------------------------------------------
// The runtime model
// ---------------------------------------------------------------------------------------------

/** One oscillator's trajectory through its texture. */
interface Trajectory {
  readonly x0: number;
  readonly y0: number;
  /** Texels per second. */
  readonly vx: number;
  readonly vy: number;
}

interface BranchRuntime {
  readonly base: number;
  readonly frequencyHz: number;
  readonly damping: number;
  readonly texture: MotionTexture;
  readonly along: Trajectory;
  readonly across: Trajectory;
}

interface SeasonRuntime {
  readonly branches: ReadonlyMap<number, BranchRuntime>;
  readonly flutterLookups: Float64Array;
}

/** Where and how big each oscillator is: what the wind field and the admittances need. */
interface OscillatorGeometry {
  /** Where it reads the wind: the centroid of every joint it carries. */
  readonly samplePoint: Vec3;
  /** Horizontal and vertical extent of what it carries, metres (EN 1991-1-4's `b` and `h`). */
  readonly widthM: number;
  readonly heightM: number;
  /** Static tip deflection of its own bend at the reference speed, metres. */
  readonly staticTipM: number;
}

/** One branch's sway at one instant, in units of the mean lean. */
interface BranchSway {
  readonly along: number;
  readonly across: number;
  /** The background's along-wind part alone: the gust the branch's leaves feel. */
  readonly gust: number;
}

/** A rig, its sidecar, and everything derived from them once. Immutable after construction. */
export interface LivingMotion {
  readonly rig: MotionRig;
  readonly sidecar: MotionSidecar;
  /** Per node: unit direction of the limb it lies on — the limb's chord, attachment to far end. */
  readonly limbDirection: readonly Vec3[];
  /** Turbulent length scale the wind is read at, metres: the sidecar's, or EN eq. B.1's. */
  readonly lengthScaleM: number;
  /** @internal the frozen turbulence field. */
  readonly field: FrozenTurbulence;
  /** @internal per branch base: where it reads the wind and how big it is. */
  readonly oscillators: ReadonlyMap<number, OscillatorGeometry>;
  /** @internal memoised per season. */
  readonly seasons: Map<Season, SeasonRuntime>;
  /** @internal the last frame's branch sway, so a frame's two halves share one evaluation. */
  readonly lastSway: { key: string; sway: ReadonlyMap<number, BranchSway> | undefined };
  /** @internal the field's phases at every oscillator for the last bearing. */
  readonly lastPhases: { bearing: number; phases: ReadonlyMap<number, Float64Array> };
}

/** The frozen field's phases at every oscillator's sample point, for this wind direction. */
function pointPhases(motion: LivingMotion, downwind: Vec3): ReadonlyMap<number, Float64Array> {
  const bearing = Math.atan2(downwind[0], downwind[1]);
  if (Object.is(motion.lastPhases.bearing, bearing)) return motion.lastPhases.phases;
  const phases = new Map<number, Float64Array>();
  for (const [base, geometry] of motion.oscillators)
    phases.set(
      base,
      turbulencePhases(motion.field, geometry.samplePoint, downwind, motion.lengthScaleM),
    );
  motion.lastPhases.bearing = bearing;
  motion.lastPhases.phases = phases;
  return phases;
}

/**
 * Every oscillator's geometry: the joints it carries (its own and everything hanging from it),
 * their centroid and extent, and the static tip deflection of its own bend.
 */
function oscillatorGeometry(
  rig: MotionRig,
  sidecar: MotionSidecar,
): Map<number, OscillatorGeometry> {
  const nodes = rig.nodes;
  const branchOf = sidecar.nodes.branch;
  interface Accumulator {
    readonly sum: [number, number, number];
    count: number;
    readonly min: [number, number, number];
    readonly max: [number, number, number];
    readonly members: number[];
  }
  const acc = new Map<number, Accumulator>();
  const grow = (base: number, p: Vec3, member: number): void => {
    let a = acc.get(base);
    if (a === undefined) {
      a = {
        sum: [0, 0, 0],
        count: 0,
        min: [p[0], p[1], p[2]],
        max: [p[0], p[1], p[2]],
        members: [],
      };
      acc.set(base, a);
    }
    for (let c = 0; c < 3; c += 1) {
      a.min[c] = Math.min(a.min[c] ?? 0, p[c] ?? 0);
      a.max[c] = Math.max(a.max[c] ?? 0, p[c] ?? 0);
    }
    if (member >= 0) {
      a.sum[0] += p[0];
      a.sum[1] += p[1];
      a.sum[2] += p[2];
      a.count += 1;
      a.members.push(member);
    }
  };
  for (let i = 1; i < nodes.length; i += 1) {
    const p = nodes[i]?.position ?? [0, 0, 0];
    // Every oscillator this joint rides: its own, then the one its oscillator hangs from, ...
    let base = branchOf[i] ?? 0;
    while (base > 0) {
      grow(base, p, i);
      const attach = nodes[base]?.parent ?? 0;
      grow(base, nodes[attach]?.position ?? p, -1);
      base = attach > 0 ? (branchOf[attach] ?? 0) : 0;
    }
  }
  const out = new Map<number, OscillatorGeometry>();
  for (const [base, a] of acc) {
    const samplePoint: Vec3 = [a.sum[0] / a.count, a.sum[1] / a.count, a.sum[2] / a.count];
    const attach = nodes[nodes[base]?.parent ?? 0]?.position ?? samplePoint;
    // The tip: the carried joint farthest from the attachment.
    let tip = samplePoint;
    let far = -1;
    for (const m of a.members) {
      const p = nodes[m]?.position ?? samplePoint;
      const r = Math.hypot(p[0] - attach[0], p[1] - attach[1], p[2] - attach[2]);
      if (r > far) {
        far = r;
        tip = p;
      }
    }
    // Its own joints' bends, each pivoting at its parent, carry the tip this far.
    let staticTipM = 0;
    for (const m of a.members) {
      if ((branchOf[m] ?? 0) !== base) continue;
      const gain = sidecar.nodes.gainRad[m] ?? 0;
      if (gain === 0) continue;
      const pivot = nodes[nodes[m]?.parent ?? 0]?.position ?? attach;
      staticTipM += gain * Math.hypot(tip[0] - pivot[0], tip[1] - pivot[1], tip[2] - pivot[2]);
    }
    out.set(base, {
      samplePoint,
      widthM: Math.max(a.max[0] - a.min[0], a.max[1] - a.min[1]),
      heightM: a.max[2] - a.min[2],
      staticTipM,
    });
  }
  return out;
}

/**
 * Binds a sidecar to its rig. Throws when they do not belong together — a sidecar for another
 * capture would move the wrong joints with confidence.
 */
export function createLivingMotion(rig: MotionRig, sidecar: MotionSidecar): LivingMotion {
  const issues = validateMotionSidecar(sidecar, rig);
  if (issues.length > 0)
    throw new Error(`motion sidecar does not fit this rig:\n  ${issues.join("\n  ")}`);
  const structure = branchStructure(rig);
  // Topological order: a limb's far end is its highest-indexed joint.
  const far = new Map<number, number>();
  rig.nodes.forEach((_, i) => {
    if (i > 0) far.set(structure.limb[i] ?? i, i);
  });
  const limbDirection = rig.nodes.map((node, i): Vec3 => {
    if (i === 0) return [0, 0, 1];
    const base = structure.limb[i] ?? i;
    const from = rig.nodes[rig.nodes[base]?.parent ?? 0]?.position;
    const to = rig.nodes[far.get(base) ?? i]?.position ?? node.position;
    if (from === undefined) return [0, 0, 1];
    const d = subtract(to, from);
    const length = Math.hypot(d[0], d[1], d[2]);
    return length > 0 ? [d[0] / length, d[1] / length, d[2] / length] : [0, 0, 1];
  });
  // Sidecars written before the length scale was recorded get EN eq. B.1 at their height.
  const lengthScaleM = sidecar.wind.lengthScaleM ?? turbulenceLengthScaleM(sidecar.treeHeightM);
  const field = frozenTurbulence(sidecar.seed);
  return {
    rig,
    sidecar,
    limbDirection,
    lengthScaleM,
    field,
    oscillators: oscillatorGeometry(rig, sidecar),
    seasons: new Map(),
    lastSway: { key: "", sway: undefined },
    lastPhases: { bearing: Number.NaN, phases: new Map() },
  };
}

/** Parses a sidecar and binds it to its rig in one step. */
export function loadLivingMotion(rig: MotionRig, sidecarText: string): LivingMotion {
  return createLivingMotion(rig, parseMotionSidecar(sidecarText, rig));
}

/** Seconds of motion the aperiodicity budget is spent on: an hour, as the tests check. */
const APERIODIC_HORIZON_S = 3600;
/** Clearance wanted between a trajectory and its own start over that horizon, wavelengths. */
const WANTED_CLEARANCE_WAVELENGTHS = 1.5;

function trajectory(key: number, seed: number, frequencyHz: number): Trajectory {
  const speed = frequencyHz * BRANCH_TEXELS_PER_CYCLE;
  const [dx, dy] = trajectoryDirection(
    key,
    seed,
    BRANCH_TEXTURE_SIZE,
    speed * APERIODIC_HORIZON_S,
    WANTED_CLEARANCE_WAVELENGTHS * BRANCH_TEXELS_PER_CYCLE,
  );
  return {
    x0: unitUniform(key * 2, seed ^ 0x51a7) * BRANCH_TEXTURE_SIZE,
    y0: unitUniform(key * 2 + 1, seed ^ 0x51a7) * BRANCH_TEXTURE_SIZE,
    vx: dx * speed,
    vy: dy * speed,
  };
}

function seasonRuntime(motion: LivingMotion, season: Season): SeasonRuntime {
  const cached = motion.seasons.get(season);
  if (cached !== undefined) return cached;
  const { sidecar } = motion;
  const winter = sidecar.seasons.winter;
  const branches = new Map<number, BranchRuntime>();
  const seed = sidecar.seed;
  sidecar.nodes.branch.forEach((base, i) => {
    if (i === 0 || branches.has(base)) return;
    const isTree = sidecar.nodes.mode[i] === 0;
    const summerHz = sidecar.nodes.frequencyHz[i] ?? 1;
    const summerZeta = sidecar.nodes.damping[i] ?? 0.1;
    const frequencyHz =
      season === "winter" && !isTree ? summerHz * winter.branchFrequencyScale : summerHz;
    // Damping is quantised by the generator, and rounded here, so branches share textures.
    const damping =
      Math.round((season === "winter" ? summerZeta * winter.dampingScale : summerZeta) * 1e4) / 1e4;
    branches.set(base, {
      base,
      frequencyHz,
      damping,
      texture: branchMotionTexture(damping, seed),
      along: trajectory(base * 2, seed, frequencyHz),
      across: trajectory(base * 2 + 1, seed, frequencyHz),
    });
  });
  const reference = sidecar.referenceSpeedMps * sidecar.wind.canopyAdvection;
  const runtime: SeasonRuntime = {
    branches,
    flutterLookups: flutterLookups(
      seed,
      FLUTTER_TEXTURE_SIZE,
      (reference * APERIODIC_HORIZON_S) / sidecar.leafSizeM,
      0.5 * (FLUTTER_MIN_WAVELENGTH_LEAVES + FLUTTER_MAX_WAVELENGTH_LEAVES),
    ),
  };
  motion.seasons.set(season, runtime);
  return runtime;
}

/** Builds every texture a season needs now, rather than on the first frame. */
export function prepareLivingMotion(motion: LivingMotion, season: Season = "summer"): void {
  seasonRuntime(motion, season);
  flutterTexture(motion.sidecar.seed);
}

/** The value a trajectory reads at time `t`. Unit RMS. */
function sampleAt(texture: MotionTexture, path: Trajectory, t: number): number {
  return sampleTexture(texture, path.x0 + path.vx * t, path.y0 + path.vy * t);
}

/** One branch's buffeting under one wind (EN 1991-1-4 Annex B). */
export interface BranchBuffeting extends BuffetingFactors {
  readonly base: number;
  readonly frequencyHz: number;
  /** Structural damping ratio of the branch this season. */
  readonly damping: number;
}

function branchFactors(
  motion: LivingMotion,
  branch: BranchRuntime,
  wind: LivingWind,
): BuffetingFactors {
  const geometry = motion.oscillators.get(branch.base);
  const ratio = wind.speedMps / motion.sidecar.referenceSpeedMps;
  return buffetingFactors(
    branch.frequencyHz,
    branch.damping,
    geometry?.widthM ?? 0,
    geometry?.heightM ?? 0,
    (geometry?.staticTipM ?? 0) * ratio * ratio,
    wind.speedMps,
    motion.lengthScaleM,
  );
}

/**
 * What every branch's sway is made of under this wind: its background `B²`, resonance `R²` and
 * aerodynamic damping. `R²/(B² + R²)` is the share of the sway at the branch's own frequency.
 */
export function livingBuffeting(motion: LivingMotion, wind: LivingWind): BranchBuffeting[] {
  const runtime = seasonRuntime(motion, wind.season);
  return [...runtime.branches.values()].map((branch) => ({
    base: branch.base,
    frequencyHz: branch.frequencyHz,
    damping: branch.damping,
    ...branchFactors(motion, branch, wind),
  }));
}

/** Every branch's sway at `t`. Memoised for the last frame, so a frame's two halves share it. */
function branchSway(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): ReadonlyMap<number, BranchSway> {
  const key = `${t}|${wind.speedMps}|${wind.bearingDeg}|${wind.season}`;
  if (motion.lastSway.key === key && motion.lastSway.sway !== undefined)
    return motion.lastSway.sway;
  const { sidecar } = motion;
  const runtime = seasonRuntime(motion, wind.season);
  const w = downwindOf(wind.bearingDeg);
  const alongRatio = sidecar.wind.turbulence.along;
  const acrossRatio = sidecar.wind.turbulence.across;
  const background = new Float64Array(2);
  const sway = new Map<number, BranchSway>();
  // The field carried downwind to time t: every branch reads it at its own point.
  const clock = turbulenceClock(motion.field, motion.lengthScaleM, wind.speedMps, t);
  const phases = pointPhases(motion, w);
  for (const branch of runtime.branches.values()) {
    const factors = branchFactors(motion, branch, wind);
    const b = Math.sqrt(factors.background2);
    const r = Math.sqrt(factors.resonance2);
    background[0] = 0;
    background[1] = 0;
    const at = phases.get(branch.base);
    if (at !== undefined) backgroundResponse(at, clock, branch.frequencyHz, background);
    const gust = alongRatio * b * (background[0] ?? 0);
    sway.set(branch.base, {
      along: gust + alongRatio * r * sampleAt(branch.texture, branch.along, t),
      across:
        acrossRatio * (b * (background[1] ?? 0) + r * sampleAt(branch.texture, branch.across, t)),
      gust,
    });
  }
  motion.lastSway.key = key;
  motion.lastSway.sway = sway;
  return sway;
}

function downwindOf(bearingDeg: number): Vec3 {
  const bearing = (Number.isFinite(bearingDeg) ? bearingDeg : 0) * DEG_TO_RAD;
  return [Math.sin(bearing), Math.cos(bearing), 0];
}

/** Two unit axes across a limb: `e1` along `d × ŵ` (the lean axis), `e2 = d × e1`. */
function crossAxes(d: Vec3, w: Vec3): [Vec3, Vec3] {
  let e1: Vec3 = [d[1] * w[2] - d[2] * w[1], d[2] * w[0] - d[0] * w[2], d[0] * w[1] - d[1] * w[0]];
  let length = Math.hypot(e1[0], e1[1], e1[2]);
  if (!(length > 1e-9)) {
    // The limb points along the wind: any axis across it will do.
    const helper: Vec3 = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    e1 = [
      d[1] * helper[2] - d[2] * helper[1],
      d[2] * helper[0] - d[0] * helper[2],
      d[0] * helper[1] - d[1] * helper[0],
    ];
    length = Math.hypot(e1[0], e1[1], e1[2]);
  }
  e1 = [e1[0] / length, e1[1] / length, e1[2] / length];
  const e2: Vec3 = [
    d[1] * e1[2] - d[2] * e1[1],
    d[2] * e1[0] - d[0] * e1[2],
    d[0] * e1[1] - d[1] * e1[0],
  ];
  return [e1, e2];
}

/** A rotation by the vector `r` (axis × angle), with the angle smoothly capped at `limit`. */
function rotationFromVector(r: Vec3, limit: number): Quat {
  const angle = Math.hypot(r[0], r[1], r[2]);
  if (angle === 0) return IDENTITY_TRANSFORM.rotation;
  const capped = softLimit(angle, limit);
  const s = Math.sin(capped / 2) / angle;
  return [r[0] * s, r[1] * s, r[2] * s, Math.cos(capped / 2)];
}

/** Per-node load factor `q_i(t) = ((U/U_ref)(1 + gust(t − x_i/U)))²`, into `out`. */
function loadFactors(motion: LivingMotion, t: number, wind: LivingWind, out: Float64Array): void {
  const { rig, sidecar } = motion;
  const w = downwindOf(wind.bearingDeg);
  const ratio = wind.speedMps / sidecar.referenceSpeedMps;
  for (let i = 0; i < rig.nodes.length; i += 1) {
    const p = rig.nodes[i]?.position ?? [0, 0, 0];
    // The gust is convected at the mean speed: it reaches the downwind side of the crown later.
    const delay = (p[0] * w[0] + p[1] * w[1]) / wind.speedMps;
    const u = ratio * (1 + gustEnvelope(wind.gust, t - delay, sidecar.seed));
    out[i] = u * u;
  }
}

/**
 * Per-node rest-to-displaced transforms at time `t`, parent-composed, as `deform` returns them.
 * `wind.speedMps === 0` returns the identity for every node, by value.
 */
export function livingTransforms(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): NodeTransform[] {
  const { rig, sidecar } = motion;
  const count = rig.nodes.length;
  const transforms: NodeTransform[] = [];
  if (!(wind.speedMps > 0) || !Number.isFinite(wind.speedMps)) {
    for (let i = 0; i < count; i += 1) transforms.push(IDENTITY_TRANSFORM);
    return transforms;
  }
  const time = Number.isFinite(t) ? t : 0;
  const w = downwindOf(wind.bearingDeg);
  const q = new Float64Array(count);
  loadFactors(motion, time, wind, q);
  // One evaluation per branch per frame, shared by every joint of the branch.
  const sway = branchSway(motion, time, wind);
  for (let i = 0; i < count; i += 1) {
    const node = rig.nodes[i];
    const gain = sidecar.nodes.gainRad[i] ?? 0;
    if (node === undefined || node.parent < 0 || gain === 0) {
      transforms.push(
        node === undefined || node.parent < 0
          ? IDENTITY_TRANSFORM
          : (transforms[node.parent] ?? IDENTITY_TRANSFORM),
      );
      continue;
    }
    const d = motion.limbDirection[i] ?? [0, 0, 1];
    const [e1, e2] = crossAxes(d, w);
    const own = sway.get(sidecar.nodes.branch[i] ?? i);
    const s1 = own?.along ?? 0;
    const s2 = own?.across ?? 0;
    const k = gain * (q[i] ?? 0);
    // Mean lean: d × ŵ, whose length is the sine between limb and wind.
    const lean: Vec3 = [
      d[1] * w[2] - d[2] * w[1],
      d[2] * w[0] - d[0] * w[2],
      d[0] * w[1] - d[1] * w[0],
    ];
    const r: Vec3 = [
      k * (lean[0] + s1 * e1[0] + s2 * e2[0]),
      k * (lean[1] + s1 * e1[1] + s2 * e2[1]),
      k * (lean[2] + s1 * e1[2] + s2 * e2[2]),
    ];
    const local = rotationFromVector(r, nodeAngleLimit(node));
    const parent = transforms[node.parent] ?? IDENTITY_TRANSFORM;
    // A hinge at the joint this segment hangs from.
    const pivot = rig.nodes[node.parent]?.position ?? node.position;
    const pivotOffset = subtract(pivot, quatRotate(local, pivot));
    transforms.push({
      rotation: quatMultiply(parent.rotation, local),
      translation: add(quatRotate(parent.rotation, pivotOffset), parent.translation),
    });
  }
  return transforms;
}

/**
 * Saturation of leaf flutter, as a multiple of its reference amplitude. Leaves reconfigure and
 * streamline in strong wind (Vogel 1989), so their drag grows more slowly than `U²`; the shape of
 * this cap is not measured.
 */
export const FLUTTER_SATURATION = 2;

/** The advected leaf-flutter field for time `t`. Still (and exactly inert) at calm and in winter. */
export function livingFlutter(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): AdvectedFlutterField {
  const { rig, sidecar } = motion;
  const count = rig.nodes.length;
  const texture = flutterTexture(sidecar.seed);
  const runtime = seasonRuntime(motion, wind.season);
  const w = downwindOf(wind.bearingDeg);
  const amplitudeM = new Float64Array(count);
  const seasonScale = wind.season === "winter" ? sidecar.seasons.winter.flutterScale : 1;
  const base = {
    kind: "advected" as const,
    texture,
    downwind: [w[0], w[1]] as const,
    texelsPerMeter: 1 / sidecar.leafSizeM,
    lookups: runtime.flutterLookups,
    phase: FLUTTER_STILL.phase,
  };
  if (!(wind.speedMps > 0) || !Number.isFinite(wind.speedMps) || seasonScale === 0) {
    return { ...base, amplitudeM, advectionTexels: 0, still: true };
  }
  const time = Number.isFinite(t) ? t : 0;
  const q = new Float64Array(count);
  loadFactors(motion, time, wind, q);
  // The leaves feel the gusts their branch feels: the background's along-wind part, the same
  // `2I·B·u` that sways the wood, raising and lowering the drag on them.
  const sway = branchSway(motion, time, wind);
  let still = true;
  for (let i = 0; i < count; i += 1) {
    const reference = (sidecar.nodes.flutterM[i] ?? 0) * seasonScale;
    if (reference === 0) continue;
    const gust = Math.max(0, 1 + (sway.get(sidecar.nodes.branch[i] ?? i)?.gust ?? 0));
    const amplitude = softLimit(reference * (q[i] ?? 0) * gust, FLUTTER_SATURATION * reference);
    if (amplitude === 0) continue;
    amplitudeM[i] = amplitude;
    still = false;
  }
  const advection = wind.speedMps * sidecar.wind.canopyAdvection * time;
  return { ...base, amplitudeM, advectionTexels: advection / sidecar.leafSizeM, still };
}

/** Both halves of a frame. */
export function livingFrame(
  motion: LivingMotion,
  t: number,
  wind: LivingWind,
): { transforms: NodeTransform[]; flutter: AdvectedFlutterField } {
  return { transforms: livingTransforms(motion, t, wind), flutter: livingFlutter(motion, t, wind) };
}

// ---------------------------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------------------------

/**
 * The largest local rotation each joint can reach under this wind, radians. A proven bound: the
 * lean is at most 1, each background read at most `BACKGROUND_SOFT_CLIP`, each texture read at
 * most its `maxAbs`, the gust at most `maxGust`.
 */
export function livingMaxNodeAngles(motion: LivingMotion, wind: LivingWind): number[] {
  const { rig, sidecar } = motion;
  if (!(wind.speedMps > 0)) return rig.nodes.map(() => 0);
  const runtime = seasonRuntime(motion, wind.season);
  const u = (wind.speedMps / sidecar.referenceSpeedMps) * (1 + maxGust(wind.gust));
  const qMax = u * u;
  return rig.nodes.map((node, i) => {
    const gain = sidecar.nodes.gainRad[i] ?? 0;
    if (node.parent < 0 || gain === 0) return 0;
    const branch = runtime.branches.get(sidecar.nodes.branch[i] ?? i);
    const turbulence = sidecar.wind.turbulence;
    let sway = 0;
    if (branch !== undefined) {
      // The background is soft-clipped to BACKGROUND_SOFT_CLIP; the resonance is a texture read.
      const factors = branchFactors(motion, branch, wind);
      const peak =
        Math.sqrt(factors.background2) * BACKGROUND_SOFT_CLIP +
        Math.sqrt(factors.resonance2) * branch.texture.maxAbs;
      sway = peak * (turbulence.along + turbulence.across);
    }
    // Lean is at most 1; the sway axes are perpendicular to it, so this over-counts, safely.
    const raw = gain * qMax * (1 + sway);
    return softLimit(raw, nodeAngleLimit(node));
  });
}

/**
 * Worst-case displacement of any splat, metres: the chain sum of joint rotations times lever
 * arms (as `maxNodeDisplacements`), plus the flutter bound. A proven bound, not a sample.
 */
export function livingMaxDisplacement(motion: LivingMotion, wind: LivingWind): number {
  const { rig, sidecar } = motion;
  if (!(wind.speedMps > 0)) return 0;
  const angles = livingMaxNodeAngles(motion, wind);
  let worst = 0;
  rig.nodes.forEach((node, index) => {
    let bound = 0;
    let cursor = index;
    while (cursor > 0) {
      const joint = rig.nodes[cursor];
      if (joint === undefined) break;
      // Each joint's rotation pivots at its parent's rest position.
      const pivot = rig.nodes[joint.parent]?.position ?? joint.position;
      const dx = node.position[0] - pivot[0];
      const dy = node.position[1] - pivot[1];
      const dz = node.position[2] - pivot[2];
      bound += (angles[cursor] ?? 0) * Math.hypot(dx, dy, dz);
      cursor = joint.parent;
    }
    if (bound > worst) worst = bound;
  });
  const seasonScale = wind.season === "winter" ? sidecar.seasons.winter.flutterScale : 1;
  let flutter = 0;
  for (const value of sidecar.nodes.flutterM) flutter = Math.max(flutter, value);
  flutter *=
    FLUTTER_SATURATION * seasonScale * advectedFlutterUnitBound(flutterTexture(sidecar.seed));
  return worst + flutter;
}
