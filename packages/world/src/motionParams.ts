/**
 * The motion sidecar: what a rig's motion is, where every number came from, and how much
 * evidence stands behind it.
 *
 * A rig says where a tree's joints are. A sidecar says how each joint moves: which oscillator it
 * belongs to, that oscillator's frequency and damping, how much of its bend lands at this joint,
 * how hard its leaves flutter — plus the wind defaults, the seed that makes the motion
 * reproducible, and `motionEvidence`, the rung of the evidence ladder the parameters stand on:
 *
 * | rung               | what stands behind the parameters                          |
 * | ------------------ | ---------------------------------------------------------- |
 * | `recorded`         | a replay of this object's own filmed motion                |
 * | `fitted-real`      | fitted to real footage of this object, with an anemometer  |
 * | `fitted-generated` | fitted to video-model clips of this object (Teacher A)     |
 * | `allometric`       | tree size and branch lengths alone — this module           |
 *
 * See docs/DECISIONS/0008-living-mode.md. **The derivation here is mirrored in Python by
 * `tools/captures/motion_params.py`**, which is what writes the committed sidecars; the
 * TypeScript copy exists so the runtime and the tests can derive parameters for a rig that has
 * none, and `motionParams.test.ts` pins the two together on the synthetic tree.
 *
 * ### The allometric rules, and their sources
 *
 * - **Limbs** (`branchStructure`, `limbModes`): at every joint the child carrying the most tips
 *   continues the axis, the straightest on a tie, with no angle limit. A limb too short to ring
 *   inside the tree's band ({@link MODE_BAND}·f0) or deeper than {@link MAX_BRANCH_ORDER} is a
 *   twig: it rides the oscillator of the limb it hangs from and bends not at all itself.
 * - **Branch frequency** `f = 2.55·L^-0.59` Hz, no lower than `f0`, `L` the limb's span (the
 *   farthest its subtree reaches from its attachment) in metres — Coder (2000,
 *   "Sway frequency in tree stems", UGA FOR00-24), as used by Habel, Kusternig & Wimmer, EG 2009,
 *   eq. (17), for broadleaf trees in leaf. Habel gives no unit and the primary could not be
 *   retrieved. Metres, because only in metres does the law agree with measured whole trees: it
 *   puts a 6 m stem at 0.89 Hz against the pendulum law's 0.98 Hz (below), where feet would give
 *   0.44 Hz. Leafless branches ring at ~2.5× that (same source). The exponent agrees with beam
 *   scaling `f ~ D/L²` under the measured allometry `D ~ L^1.37–1.38` (Rodriguez et al. 2008),
 *   which holds for a whole branch — hence the span, not a segment.
 * - **Whole-tree frequency** `f0 = C / √H` — the simple-pendulum law that best predicts open-grown
 *   broadleaves (Jackson et al. 2021, Biogeosciences 18, 4059, Table 2: R² 0.67, n = 89). The
 *   paper publishes no intercept; `C = 2.4 Hz·m^½` is **an estimate read by eye from its Fig. 2a**
 *   (open-grown broadleaves: ~1.1 Hz at 6 m, ~0.7 Hz at 13–15 m, ~0.45 Hz at 25 m), ±30 %.
 * - **Whole-tree damping** ζ = 0.086 in leaf, 0.039 leafless — pull-and-release tests on four
 *   broadleaves (Jackson et al. 2019, J. R. Soc. Interface 16: 20190116): 8.6 ± 2.2 % summer,
 *   3.9 ± 1.3 % winter.
 * - **Branch damping rising with what a limb carries**, 0.045 → 0.106 with the limb's share of
 *   leaf tips — pluck tests on a tree's branches one by one (James & Haritos 2010, AEES 2010
 *   conference, Table 1 and text): single branches 3.5–4.5 % at small amplitude (up to 7.5 % at
 *   large), the tree with its branches 10.6 %, because sub-branches act as tuned mass dampers on
 *   what they hang from. The endpoints are measured; the interpolation by tip share is an estimate. Aerodynamic damping,
 *   which grows with the wind, is added at run time (`turbulence.ts`, EN 1991-1-4 eq. F.18).
 * - **Wind** (`turbulence.ts`): turbulence intensity `1/ln(H/z0)` and length scale
 *   `300·(H/200)^(0.67 + 0.05 ln z0)` at the tree's height, terrain category III (EN 1991-1-4:2005
 *   eqs. 4.7, B.1, Table 4.1); the sway RMS is `2·I` along the wind (eq. 6.3's linearised drag)
 *   and `0.75·I` across it (estimate). The explicit gust envelope is off: the turbulence
 *   spectrum (eq. B.2) carries the gusts.
 * - **Bend per limb**: elastic similarity (McMahon & Kronauer 1976) — every limb deflects by
 *   the same angle whatever its length — times `min(1, f0/f)^0.305`, so that its tip deflection
 *   falls as `1/f²` as a sub-resonant oscillator's does ({@link RESPONSE_EXPONENT}), spread over
 *   its joints in proportion to the limb length each stands for. **This is why no radius is read**:
 *   the woody radius `skeleton.py` reports is a resolution limit on most crown nodes, and the
 *   model does not need it. The constants themselves are estimates (Teacher A's job to fit).
 */

import { type MotionRig } from "./rig";
import {
  LATERAL_TURBULENCE_RATIO,
  turbulenceIntensity,
  turbulenceLengthScaleM,
} from "./turbulence";
import { distance, type Vec3 } from "./vec";

export const MOTION_SIDECAR_FORMAT = "hexapod.motion";
export const MOTION_SIDECAR_VERSION = 1;

/** The evidence ladder, strongest first. A higher rung replaces a lower one as evidence arrives. */
export const MOTION_EVIDENCE_LADDER = [
  "recorded",
  "fitted-real",
  "fitted-generated",
  "allometric",
] as const;

export type MotionEvidence = (typeof MOTION_EVIDENCE_LADDER)[number];

/** A short description of each rung, for UI. */
export const MOTION_EVIDENCE_LABEL: Readonly<Record<MotionEvidence, string>> = {
  recorded: "recorded motion of this object",
  "fitted-real": "fitted to real footage of this object",
  "fitted-generated": "fitted to generated video of this object",
  allometric: "estimated from tree size and branch lengths",
};

/** SpeedTree-style gust envelope: gusts raise the mean speed for a while, then let it fall. */
export interface GustSettings {
  /** Peak fractional increase of the mean speed, e.g. 0.35 → a gust peaks at 1.35·U. */
  readonly strength: number;
  /** Relative spread of each gust's strength and duration, `0..1`. */
  readonly variance: number;
  /** Gusts per minute. */
  readonly frequencyPerMin: number;
  /** Typical gust length, seconds. */
  readonly durationS: number;
}

export interface SidecarWind {
  /** Default mean wind speed, m/s. The scene's wind control overrides it. */
  readonly meanSpeedMps: number;
  /** Downwind bearing, degrees clockwise from north (the direction the wind blows towards). */
  readonly bearingDeg: number;
  readonly gust: GustSettings;
  /** RMS of the along- and across-wind sway relative to the mean lean. */
  readonly turbulence: { readonly along: number; readonly across: number };
  /** In-canopy advection speed of the leaf-flutter field, as a fraction of the mean speed. */
  readonly canopyAdvection: number;
  /**
   * Turbulent length scale at the tree's height, metres (EN 1991-1-4 eq. B.1): where the wind
   * spectrum's energy sits, `f ≈ 0.15·U/L`. Absent in sidecars written before it existed; the
   * runtime then derives it from `treeHeightM` the same way.
   */
  readonly lengthScaleM?: number;
}

/** Per-node columns. Every array has one entry per rig node, in rig order. */
export interface SidecarNodes {
  /** Index of the first node of this node's branch: nodes sharing it share one oscillator. */
  readonly branch: readonly number[];
  /** `0` for the whole-tree (pendulum) mode, `1` for a branch mode. */
  readonly mode: readonly number[];
  /** Share of the branch's bend taken at this joint. Sums to 1 over a branch. */
  readonly share: readonly number[];
  /** The branch's natural frequency, Hz (summer). */
  readonly frequencyHz: readonly number[];
  /** The branch's damping ratio (summer). */
  readonly damping: readonly number[];
  /** Mean bend at this joint at the reference speed, radians. */
  readonly gainRad: readonly number[];
  /** Leaf-flutter amplitude of this node's splats at the reference speed, metres. */
  readonly flutterM: readonly number[];
}

/**
 * One plant of a forest rig's sidecar (`tools/captures/scene_plants.py`): the per-node columns
 * already carry each plant's own frequencies, damping and gains, so what is left per plant is
 * what the wind needs to know about it — how tall it is, so the turbulence it feels is taken
 * at its own height (EN 1991-1-4 eqs. 4.7 and B.1), and which rig the evidence supported.
 * Optional in the format: a single-plant sidecar has none, and a sidecar written before
 * forests loads unchanged.
 */
export interface SidecarPlant {
  readonly id: string;
  readonly class: string;
  /** `skeleton` (skeleton.py), `crown` (a few-bone crown rig) or `trunk` (a snag's axis). */
  readonly rig: string;
  readonly nodeStart: number;
  readonly nodeEnd: number;
  readonly heightM: number;
  /** Sway RMS along and across the mean lean at the plant's height (`2I`, `0.75I`). */
  readonly turbulence: { readonly along: number; readonly across: number };
  /** EN 1991-1-4 eq. B.1 at the plant's height, metres: its admittances are read at it. */
  readonly lengthScaleM: number;
}

export interface ProvenanceEntry {
  readonly rule: string;
  readonly source: string;
  readonly status: "cited" | "estimate" | "unverified";
}

export interface MotionSidecar {
  readonly format: typeof MOTION_SIDECAR_FORMAT;
  readonly version: typeof MOTION_SIDECAR_VERSION;
  readonly motionEvidence: MotionEvidence;
  /** `canonicalChecksum` of the rig these parameters belong to. */
  readonly rigChecksum: string;
  readonly nodeCount: number;
  /** Seeds every motion texture and trajectory: same seed, same wind, same frame. */
  readonly seed: number;
  readonly treeHeightM: number;
  readonly leafSizeM: number;
  /** Speed at which `gainRad` and `flutterM` are quoted, m/s. Deflection scales as `(U/this)²`. */
  readonly referenceSpeedMps: number;
  readonly wind: SidecarWind;
  readonly seasons: {
    readonly winter: {
      readonly dampingScale: number;
      readonly branchFrequencyScale: number;
      readonly flutterScale: number;
    };
  };
  readonly nodes: SidecarNodes;
  /**
   * A forest rig's plants, when the sidecar covers many (`SidecarPlant`). The top-level
   * `treeHeightM` is then the tallest plant's, at which the one shared wind field is read.
   */
  readonly plants?: readonly SidecarPlant[];
  readonly provenance: Readonly<Record<string, ProvenanceEntry>>;
  readonly generator: string;
}

// ---------------------------------------------------------------------------------------------
// Constants of the allometric derivation. Mirrored in tools/captures/motion_params.py.
// ---------------------------------------------------------------------------------------------

/** Coder (2000) via Habel et al. (2009), eq. (17): `f = 2.55·L^-0.59`. */
export const CODER_COEFFICIENT_HZ = 2.55;
export const CODER_EXPONENT = -0.59;
/** Habel et al. (2009) §5.4: leafless branches ring at ~2.5× eq. (17). */
export const LEAFLESS_FREQUENCY_SCALE = 2.5;
/** `f0 = C/√H`, Hz·m^½. Estimate read off Jackson et al. (2021) Fig. 2a. */
export const PENDULUM_COEFFICIENT = 2.4;
/** Jackson et al. (2019): 8.6 ± 2.2 % in leaf, 3.9 ± 1.3 % leafless. */
export const TREE_DAMPING_SUMMER = 0.086;
export const TREE_DAMPING_WINTER = 0.039;
/**
 * Damping of the smallest and the largest leafy limb: a single branch (3.5–4.5 % at small
 * amplitude) and a stem carrying its branches (10.6 %), from pluck tests (James & Haritos 2010,
 * AEES conference, Table 1).
 */
export const LIMB_DAMPING_MIN = 0.045;
export const LIMB_DAMPING_MAX = 0.106;
/** Damping levels between the two: `LEVELS + 1` distinct values, so textures can be shared. */
export const DAMPING_LEVELS = 2;
/** Shortest branch length used in eq. (17), metres: caps a stub's frequency near 14 Hz. */
export const MIN_BRANCH_LENGTH_M = 0.05;
/**
 * The band a tree's modes ring in, as a multiple of its whole-tree frequency `f0`. A limb whose
 * eq. (17) frequency would lie above `MODE_BAND·f0` is too short to be a mode of its own and
 * becomes a twig. Estimate from Rodriguez, de Langre & Moulia 2008 (AJB 95:1523): the first 25
 * modes of a 7.9 m walnut all lie in 1.4–2.6 Hz, and trees show "fundamental modes in the range
 * of 1–1.5 Hz with a large number of their branch modes in the 2.5–3 Hz band".
 */
export const MODE_BAND = 3;
/**
 * Deepest branching order that gets a mode of its own; deeper limbs are twigs. SpeedTree moves
 * one or two branch levels independently and Unreal's Pivot Painter 2 at most four hierarchy
 * levels; a rig of hundreds of independent parts is not identifiable from video (Chen & Lou,
 * "Wind on Trees", 2026). A design choice within that practice.
 */
export const MAX_BRANCH_ORDER = 3;
/**
 * Exponent of a limb's bend in `f0/f`. Below resonance a limb's tip deflects by
 * `(F/m)/(2πf)²` (Habel eq. 15 at `f ≪ f_h`), and by eq. (17) its length goes as
 * `f^(1/CODER_EXPONENT)`, so its bend angle — deflection over length — goes as
 * `f^-(2 + 1/CODER_EXPONENT)` = `f^-0.305`. Derived, at equal drag per unit mass (an estimate).
 */
export const RESPONSE_EXPONENT = 2 + 1 / CODER_EXPONENT;
/** Reference speed the gains are quoted at, m/s. */
export const REFERENCE_SPEED_MPS = 10;
/** Whole-tree lean at the reference speed, radians. Estimate (≈1.1°). */
export const TREE_BEND_REF_RAD = 0.02;
/** A limb's lean at the reference speed at the tree's own frequency, radians. Estimate (≈2.9°). */
export const BRANCH_BEND_REF_RAD = 0.05;
/** Leaf flutter at the reference speed for a node at a tip, metres. Estimate (was 12 mm). */
export const FLUTTER_REF_M = 0.006;
/** Flutter falls off as `exp(−reach / this)` away from the tips, metres. */
export const FLUTTER_REACH_M = 0.5;
/** Below this share of `FLUTTER_REF_M` a node does not flutter at all. */
export const FLUTTER_CUT = 0.05;

/**
 * SpeedTree-style gust envelope, **off**: the EN 1991-1-4 turbulence spectrum the runtime reads
 * already holds the gusts, and a scripted bump on top of it would count them twice. Kept in the
 * format (a `strength` above 0 still adds it) so older sidecars load unchanged.
 */
export const DEFAULT_GUST: GustSettings = {
  strength: 0,
  variance: 0.3,
  frequencyPerMin: 3,
  durationS: 5,
};

/** The wind defaults for a tree `treeHeightM` tall: EN 1991-1-4 turbulence at its height. */
export function defaultSidecarWind(treeHeightM: number): SidecarWind {
  const intensity = turbulenceIntensity(treeHeightM);
  return {
    meanSpeedMps: 5,
    bearingDeg: 0,
    gust: DEFAULT_GUST,
    turbulence: {
      along: roundTo(2 * intensity, 4),
      across: roundTo(LATERAL_TURBULENCE_RATIO * intensity, 4),
    },
    canopyAdvection: 0.3,
    lengthScaleM: roundTo(turbulenceLengthScaleM(treeHeightM), 2),
  };
}

/** The synthetic tree's leaf size, metres: its leaf discs are 2–5 cm half-widths. */
export const DEFAULT_LEAF_SIZE_M = 0.06;
/**
 * Smallest leaf size a capture's splats may set, metres. A splat is a piece of a leaf: the
 * Minnetonka tree's foliage splats are 2.15 cm, and flutter wavelengths of 4–10 of those,
 * advected at 0.3·U, flicker at 10–20 Hz. Estimate: a broadleaf's leaf is 5 cm or more.
 * Applied by `motion_params.py` when it estimates the size from a PLY.
 */
export const MIN_LEAF_SIZE_M = 0.05;

export const ALLOMETRIC_PROVENANCE: Readonly<Record<string, ProvenanceEntry>> = {
  branchFrequency: {
    rule: "f = 2.55 * L^-0.59 Hz, L = limb span (farthest reach of its subtree from its attachment) in metres; unit not stated by Habel, primary not retrieved; metres because only then does it agree with whole-tree data (6 m: 0.89 Hz vs 2.4/sqrt(H) = 0.98; in feet 0.44)",
    source:
      "Coder 2000, Sway frequency in tree stems, UGA FOR00-24, via Habel, Kusternig & Wimmer, Physically Guided Animation of Trees, EG 2009, eq. 17",
    status: "cited",
  },
  limbGrouping: {
    rule: "each joint continues into its child with the most tips (straightest on a tie), no angle limit; a limb is a mode when eq. 17 puts it within 3 f0 and it is at most third-order, otherwise a twig riding the limb it hangs from with no bend of its own; a mode rings no lower than f0",
    source:
      "da Vinci's rule for the main axis; band from Rodriguez, de Langre & Moulia 2008, AJB 95:1523 (walnut: first 25 modes in 1.4-2.6 Hz; branch modes 2.5-3 Hz over 1-1.5 Hz fundamentals); order cap within game practice (SpeedTree 1-2 branch levels, Pivot Painter 2 up to 4)",
    status: "estimate",
  },
  leaflessFrequency: {
    rule: "leafless branches ~2.5x eq. 17",
    source: "Habel et al. EG 2009, sec. 5.4",
    status: "cited",
  },
  treeFrequency: {
    rule: "f0 = 2.4 / sqrt(H) Hz (simple pendulum); C = 2.4 read by eye from Fig. 2a, +-30%",
    source:
      "Jackson et al. 2021, Biogeosciences 18, 4059, Table 2 (pendulum best for open-grown broadleaves) and Fig. 2a",
    status: "estimate",
  },
  treeDamping: {
    rule: "zeta = 0.086 summer, 0.039 winter",
    source:
      "Jackson et al. 2019, J. R. Soc. Interface 16: 20190116 (pull-and-release, 4 broadleaves)",
    status: "cited",
  },
  limbDamping: {
    rule: "zeta rises from 0.045 to 0.106 with sqrt(limb leaf tips / largest limb's), in 2 steps; aerodynamic damping 2 pi f x_s / U added at run time",
    source:
      "James & Haritos 2010, The Role of Branches in the Dynamic Response Characteristics of Trees, AEES conference, Table 1 (single branches 3.5-4.5 % at small amplitude, 7.5 % at large; the tree with its branches 10.6 %); EN 1991-1-4:2005 eq. F.18 for the aerodynamic part; the interpolation by tip share is chosen",
    status: "estimate",
  },
  windSpectrum: {
    rule: "S_L(f_L) = 6.8 f_L / (1 + 10.2 f_L)^(5/3), f_L = f L / U, as a frozen field of random Fourier modes advected at U; each limb follows it through its static compliance, low-passed at its own frequency; resonance R^2 = pi^2/(2 delta) S_L(f_n) R_h R_b on top; background B^2 = 1/(1 + 0.9 ((b + h)/L)^0.63)",
    source:
      "EN 1991-1-4:2005+A1:2010 Annex B, eqs. B.2, B.3, B.6-B.8; frozen turbulence (Taylor's hypothesis) as in Habel et al. EG 2009 sec. 7.2",
    status: "cited",
  },
  dragScaling: {
    rule: "mean deflection proportional to U^2; fluctuation 2 I sqrt(B^2 + R^2) of it",
    source:
      "Jackson et al. 2021 (deflection fitted linear in squared wind speed); EN 1991-1-4:2005 eq. 6.3",
    status: "cited",
  },
  turbulenceScale: {
    rule: "I = 1/ln(H/z0), L = 300 (H/200)^(0.67 + 0.05 ln z0), H held at >= z_min; terrain category III (z0 = 0.3 m, z_min = 5 m)",
    source:
      "EN 1991-1-4:2005+A1:2010 eqs. 4.7 (k_I = c0 = 1) and B.1, Table 4.1; the category is chosen for a garden or park tree",
    status: "estimate",
  },
  bendGains: {
    rule: "tree lean 0.02 rad, a limb 0.05 rad * min(1, f0/f)^(2 - 1/0.59) at 10 m/s, spread over its joints by segment length",
    source:
      "elastic similarity (McMahon & Kronauer 1976) for the spread; tip deflection (F/m)/(2 pi f)^2 below resonance (Habel et al. EG 2009 eq. 15) with L from eq. 17 gives the angle's f^-0.305, at equal drag per unit mass (assumed); magnitudes chosen",
    status: "estimate",
  },
  turbulence: {
    rule: "sway RMS 2 I along / 0.75 I across the mean lean (2 I from the linearised drag); no scripted gust envelope",
    source: "EN 1991-1-4:2005 eq. 6.3 for 2 I; the across ratio is chosen (the previous 0.6 : 0.8)",
    status: "estimate",
  },
  leafFlutter: {
    rule: "advected field, wavelengths 4-10 leaf sizes (leaf size >= 5 cm), carried at 0.3 U; 6 mm at 10 m/s at the tips",
    source:
      "Habel et al. EG 2009 sec. 7.2 (min wavelength >= 4x leaf size, advected by -W t); Tadrist et al. 2018, J. R. Soc. Interface 15: 20180010 (flutter dominates only at low wind, branch buffeting above); amplitudes and the 5 cm floor chosen",
    status: "estimate",
  },
};

// ---------------------------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------------------------

export interface DeriveOptions {
  /** Tree height, metres. Default: highest rig node above the root. */
  readonly treeHeightM?: number;
  readonly leafSizeM?: number;
  readonly seed?: number;
  readonly wind?: SidecarWind;
  readonly generator?: string;
}

/** Branch structure of a rig: which node continues which limb. Pure geometry. */
export interface BranchStructure {
  /** Per node: the child that continues its limb, or -1. */
  readonly continuation: readonly number[];
  /** Per node: first node of the limb (chain of continuations) it lies on. */
  readonly limb: readonly number[];
  /** Per node: its limb's branching order — 0 for the trunk, 1 for a limb on it, and so on. */
  readonly order: readonly number[];
  /** Per node: length of the segment from its parent, metres. 0 for the root. */
  readonly segmentM: readonly number[];
  /** Per limb base node: the sum of its joints' segments, metres. Spreads the bend. */
  readonly pathLengthM: ReadonlyMap<number, number>;
  /**
   * Per limb base node: its span — the farthest any joint of its subtree (its axis and all
   * that hangs from it) reaches from the joint it hangs from, metres. Sets the frequency.
   */
  readonly branchLengthM: ReadonlyMap<number, number>;
  /** Per node: number of tips (childless nodes) at or below it. */
  readonly tips: readonly number[];
  /** Per node: longest path to a tip below it, metres. */
  readonly reachM: readonly number[];
  /** Base node of the whole-tree branch (the root's continuation), or -1 for a one-node rig. */
  readonly treeBranch: number;
}

function unit(a: Vec3, b: Vec3): Vec3 | undefined {
  const d = distance(a, b);
  if (!(d > 0)) return undefined;
  return [(b[0] - a[0]) / d, (b[1] - a[1]) / d, (b[2] - a[2]) / d];
}

/**
 * Splits a rig into limbs — chains of joints that continue one axis — and limbs into modes.
 *
 * At every joint exactly one child continues the axis: the one carrying the most tips (by da
 * Vinci's rule a limb's cross-section follows what it carries), the straightest of those on a
 * tie, measured against the direction the joint was arrived from (straight up for the root).
 * There is no angle limit, because an extracted skeleton zigzags — a median 56° turn per joint
 * on the Minnetonka rig — and an unbranched chain is one limb however much it wanders. Every
 * other child starts a limb of its own. A limb's length is its chord, from the joint it hangs
 * from to its far end, so that zigzag does not lengthen it either.
 *
 * The root's continuation is the **whole-tree** limb — the trunk — whose mode is the pendulum
 * sway. Which limbs are modes and which are twigs depends on the tree's own frequency, so it is
 * decided in {@link limbModes}.
 *
 * The first version of this rule (continue within 35° of the incoming direction, when at
 * least half as long as the longest sibling) cut the 200-joint Minnetonka rig into 150
 * branches, 121 of them a single joint 0.2–0.7 m long, ringing at 2–10 Hz: the vibration.
 */
export function branchStructure(rig: MotionRig): BranchStructure {
  const nodes = rig.nodes;
  const count = nodes.length;
  const children: number[][] = nodes.map(() => []);
  const segmentM = new Array<number>(count).fill(0);
  nodes.forEach((node, i) => {
    if (node.parent >= 0) {
      children[node.parent]?.push(i);
      const parent = nodes[node.parent];
      if (parent !== undefined) segmentM[i] = distance(node.position, parent.position);
    }
  });
  const reachM = new Array<number>(count).fill(0);
  const tips = new Array<number>(count).fill(0);
  for (let i = count - 1; i >= 0; i -= 1) {
    const kids = children[i] ?? [];
    if (kids.length === 0) tips[i] = 1;
    for (const kid of kids) {
      tips[i] = (tips[i] ?? 0) + (tips[kid] ?? 0);
      const through = (segmentM[kid] ?? 0) + (reachM[kid] ?? 0);
      if (through > (reachM[i] ?? 0)) reachM[i] = through;
    }
  }
  const continuation = new Array<number>(count).fill(-1);
  for (let i = 0; i < count; i += 1) {
    const node = nodes[i];
    const kids = children[i] ?? [];
    if (node === undefined || kids.length === 0) continue;
    const parent = node.parent >= 0 ? nodes[node.parent] : undefined;
    const incoming: Vec3 = (parent === undefined
      ? undefined
      : unit(parent.position, node.position)) ?? [0, 0, 1];
    let best = -1;
    let bestTips = -1;
    let bestCos = -2;
    for (const kid of kids) {
      const child = nodes[kid];
      if (child === undefined) continue;
      const out = unit(node.position, child.position) ?? incoming;
      const c = out[0] * incoming[0] + out[1] * incoming[1] + out[2] * incoming[2];
      const t = tips[kid] ?? 0;
      if (t > bestTips || (t === bestTips && c > bestCos)) {
        bestTips = t;
        bestCos = c;
        best = kid;
      }
    }
    continuation[i] = best;
  }
  // A node whose parent is a root starts a limb, as node 0's children do: the roots are the
  // anchors (node 0 of a tree; each plant's first node and the static anchors of a forest).
  const inner = (i: number): boolean => i >= 0 && (nodes[i]?.parent ?? -1) >= 0;
  const limb = new Array<number>(count).fill(0);
  const pathLengthM = new Map<number, number>();
  const last = new Map<number, number>();
  for (let i = 1; i < count; i += 1) {
    const node = nodes[i];
    if (node === undefined) continue;
    const p = node.parent;
    if (p < 0) {
      limb[i] = i;
      continue;
    }
    const base = inner(p) && continuation[p] === i ? (limb[p] ?? i) : i;
    limb[i] = base;
    pathLengthM.set(base, (pathLengthM.get(base) ?? 0) + (segmentM[i] ?? 0));
    // Topological order: a chain's far end is its highest index.
    last.set(base, i);
  }
  const treeBranch = continuation[0] ?? -1;
  // Every root's continuation is a whole-plant limb: the trunk of its tree.
  const trunks = new Set<number>();
  nodes.forEach((node, i) => {
    const c = continuation[i] ?? -1;
    if (node.parent < 0 && c >= 0) trunks.add(c);
  });
  // Branching order: the trunk is 0, a limb on it 1, a limb on that 2, ...
  const order = new Array<number>(count).fill(0);
  for (let i = 1; i < count; i += 1) {
    const base = limb[i] ?? i;
    if ((nodes[i]?.parent ?? -1) < 0) {
      order[i] = 0;
    } else if (i === base && !trunks.has(base)) {
      const attach = nodes[base]?.parent ?? 0;
      order[i] = inner(attach) ? (order[limb[attach] ?? 0] ?? 0) + 1 : 1;
    } else {
      order[i] = order[base] ?? 0;
    }
  }
  // A limb's length is its span: the farthest any joint of its subtree reaches from the joint
  // it hangs from. Beam scaling (f ~ D/L², D ~ L^1.4) holds for a whole branch, not a segment;
  // and a straight-line reach ignores the zigzag of an extracted chain.
  const branchLengthM = new Map<number, number>();
  for (const base of last.keys()) branchLengthM.set(base, 0);
  for (let i = 1; i < count; i += 1) {
    const here = nodes[i]?.position ?? [0, 0, 0];
    let cursor = i;
    while (inner(cursor)) {
      const base = limb[cursor] ?? cursor;
      const attach = nodes[base]?.parent ?? 0;
      const reach = distance(nodes[attach]?.position ?? here, here);
      if (reach > (branchLengthM.get(base) ?? 0)) branchLengthM.set(base, reach);
      cursor = attach;
    }
  }
  return {
    continuation,
    limb,
    order,
    segmentM,
    pathLengthM,
    branchLengthM,
    tips,
    reachM,
    treeBranch,
  };
}

/** Which oscillator every joint follows, and whether its limb is a twig. */
export interface LimbModes {
  /**
   * Per node: first node of the branch whose oscillator it follows — its limb's, or for a twig
   * the limb it hangs from's. The root is its own branch (index 0).
   */
  readonly branch: readonly number[];
  /** Per node: whether its limb is a twig: no bend of its own, leaf flutter only. */
  readonly twig: readonly boolean[];
}

/**
 * Splits limbs into modes and twigs. A limb is a **twig** — it rides the oscillator of the limb
 * it hangs from, bends not at all itself, and moves only by leaf flutter — when it is too short
 * to ring inside the tree's band (eq. 17 above {@link MODE_BAND}·f0) or deeper than
 * {@link MAX_BRANCH_ORDER}. A twig on the root joins the trunk's mode.
 */
export function limbModes(
  rig: MotionRig,
  structure: BranchStructure,
  treeFrequency: number,
): LimbModes {
  const nodes = rig.nodes;
  const count = nodes.length;
  const tree = structure.treeBranch;
  const ceiling = MODE_BAND * treeFrequency;
  const branch = new Array<number>(count).fill(0);
  const twig = new Array<boolean>(count).fill(false);
  for (let i = 1; i < count; i += 1) {
    const base = structure.limb[i] ?? i;
    if (
      base !== tree &&
      ((structure.order[base] ?? 0) > MAX_BRANCH_ORDER ||
        branchFrequencyHz(structure.branchLengthM.get(base) ?? 0) > ceiling)
    ) {
      twig[i] = true;
      const attach = nodes[base]?.parent ?? 0;
      branch[i] = attach > 0 ? (branch[attach] ?? tree) : tree;
    } else {
      branch[i] = base;
    }
  }
  return { branch, twig };
}

/** Coder's law. */
export function branchFrequencyHz(lengthM: number): number {
  const length = Math.max(Number.isFinite(lengthM) ? lengthM : 0, MIN_BRANCH_LENGTH_M);
  return CODER_COEFFICIENT_HZ * Math.pow(length, CODER_EXPONENT);
}

/** The pendulum law, `C/√H`. */
export function treeFrequencyHz(heightM: number): number {
  const height = Math.max(Number.isFinite(heightM) ? heightM : 0, 0.5);
  return PENDULUM_COEFFICIENT / Math.sqrt(height);
}

/**
 * A limb's total bend at the reference speed, radians: {@link BRANCH_BEND_REF_RAD} scaled by
 * `min(1, f0 / f)^0.305` ({@link RESPONSE_EXPONENT}), so its tip deflection falls as `1/f²`.
 */
export function limbBendRad(frequencyHz: number, treeFrequency: number): number {
  const ratio = frequencyHz > 0 ? Math.min(1, treeFrequency / frequencyHz) : 1;
  return BRANCH_BEND_REF_RAD * Math.pow(ratio, RESPONSE_EXPONENT);
}

/** Damping of a limb carrying `tips` of the largest limb's `maxTips` leaf tips. Quantised. */
export function limbDamping(tips: number, maxTips: number): number {
  const share = maxTips > 0 ? Math.sqrt(Math.min(1, Math.max(0, tips / maxTips))) : 0;
  const level = Math.round(share * DAMPING_LEVELS) / DAMPING_LEVELS;
  return LIMB_DAMPING_MIN + (LIMB_DAMPING_MAX - LIMB_DAMPING_MIN) * level;
}

/** Rounds to a fixed number of significant decimals, so both languages emit the same JSON. */
export function roundTo(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/**
 * Derives an `allometric` sidecar from a rig alone. Deterministic; mirrors
 * `tools/captures/motion_params.py`.
 */
export function deriveMotionSidecar(rig: MotionRig, options: DeriveOptions = {}): MotionSidecar {
  const structure = branchStructure(rig);
  const nodes = rig.nodes;
  const root = nodes[0];
  let top = 0;
  for (const node of nodes) top = Math.max(top, node.position[2] - (root?.position[2] ?? 0));
  const treeHeightM = options.treeHeightM ?? top;
  const treeHz = treeFrequencyHz(treeHeightM);
  const modes = limbModes(rig, structure, treeHz);
  let maxTips = 0;
  for (let i = 1; i < nodes.length; i += 1) {
    const base = modes.branch[i] ?? i;
    if (base === structure.treeBranch) continue;
    maxTips = Math.max(maxTips, structure.tips[base] ?? 0);
  }
  const branch: number[] = [];
  const mode: number[] = [];
  const share: number[] = [];
  const frequencyHz: number[] = [];
  const damping: number[] = [];
  const gainRad: number[] = [];
  const flutterM: number[] = [];
  nodes.forEach((_node, i) => {
    if (i === 0) {
      branch.push(0);
      mode.push(0);
      share.push(0);
      frequencyHz.push(roundTo(treeHz, 4));
      damping.push(TREE_DAMPING_SUMMER);
      gainRad.push(0);
      flutterM.push(0);
      return;
    }
    const base = modes.branch[i] ?? i;
    const isTree = base === structure.treeBranch;
    // A mode rings no lower than the tree it hangs from (Rodriguez et al. 2008).
    const frequency = isTree
      ? treeHz
      : Math.max(treeHz, branchFrequencyHz(structure.branchLengthM.get(base) ?? 0));
    // A twig rides its limb: same oscillator, no bend of its own.
    const path = structure.pathLengthM.get(base) ?? 0;
    const s = modes.twig[i] === true || !(path > 0) ? 0 : (structure.segmentM[i] ?? 0) / path;
    const bend = isTree ? TREE_BEND_REF_RAD : limbBendRad(frequency, treeHz);
    branch.push(base);
    mode.push(isTree ? 0 : 1);
    share.push(roundTo(s, 6));
    frequencyHz.push(roundTo(frequency, 4));
    damping.push(
      roundTo(isTree ? TREE_DAMPING_SUMMER : limbDamping(structure.tips[base] ?? 0, maxTips), 4),
    );
    gainRad.push(roundTo(bend * s, 7));
    const weight = Math.exp(-(structure.reachM[i] ?? 0) / FLUTTER_REACH_M);
    flutterM.push(weight < FLUTTER_CUT ? 0 : roundTo(FLUTTER_REF_M * weight, 6));
  });
  return {
    format: MOTION_SIDECAR_FORMAT,
    version: MOTION_SIDECAR_VERSION,
    motionEvidence: "allometric",
    rigChecksum: rig.canonicalChecksum,
    nodeCount: nodes.length,
    seed: options.seed ?? 1,
    treeHeightM: roundTo(treeHeightM, 4),
    leafSizeM: options.leafSizeM ?? DEFAULT_LEAF_SIZE_M,
    referenceSpeedMps: REFERENCE_SPEED_MPS,
    wind: options.wind ?? defaultSidecarWind(treeHeightM),
    seasons: {
      winter: {
        dampingScale: roundTo(TREE_DAMPING_WINTER / TREE_DAMPING_SUMMER, 4),
        branchFrequencyScale: LEAFLESS_FREQUENCY_SCALE,
        flutterScale: 0,
      },
    },
    nodes: { branch, mode, share, frequencyHz, damping, gainRad, flutterM },
    provenance: ALLOMETRIC_PROVENANCE,
    generator: options.generator ?? "@twin/world deriveMotionSidecar",
  };
}

// ---------------------------------------------------------------------------------------------
// Parsing and validation
// ---------------------------------------------------------------------------------------------

const NODE_COLUMNS = [
  "branch",
  "mode",
  "share",
  "frequencyHz",
  "damping",
  "gainRad",
  "flutterM",
] as const;

/** Every problem with a sidecar, against the rig it claims to describe. Empty means valid. */
export function validateMotionSidecar(sidecar: MotionSidecar, rig?: MotionRig): string[] {
  const issues: string[] = [];
  // Read loosely: this is what stands between parsed JSON and the typed value.
  const raw = sidecar as unknown as Record<string, unknown>;
  if (raw.format !== MOTION_SIDECAR_FORMAT) issues.push(`format must be ${MOTION_SIDECAR_FORMAT}`);
  if (raw.version !== MOTION_SIDECAR_VERSION)
    issues.push(`version ${JSON.stringify(raw.version)} is not ${MOTION_SIDECAR_VERSION}`);
  if (!(MOTION_EVIDENCE_LADDER as readonly unknown[]).includes(raw.motionEvidence))
    issues.push(`motionEvidence ${JSON.stringify(raw.motionEvidence)} is not on the ladder`);
  if (!Number.isInteger(sidecar.seed)) issues.push("seed must be an integer");
  for (const key of ["treeHeightM", "leafSizeM", "referenceSpeedMps"] as const) {
    const value = sidecar[key];
    if (!(Number.isFinite(value) && value > 0)) issues.push(`${key} must be > 0`);
  }
  const lengthScale = (raw.wind as { lengthScaleM?: unknown } | undefined)?.lengthScaleM;
  if (lengthScale !== undefined && !(typeof lengthScale === "number" && lengthScale > 0))
    issues.push("wind.lengthScaleM must be > 0 when present");
  const count = sidecar.nodeCount;
  for (const column of NODE_COLUMNS) {
    const values = sidecar.nodes[column] as readonly unknown[] | undefined;
    if (!Array.isArray(values) || values.length !== count) {
      issues.push(`nodes.${column} must have ${count} entries`);
      continue;
    }
    if (!values.every((v) => typeof v === "number" && Number.isFinite(v)))
      issues.push(`nodes.${column} must be finite numbers`);
  }
  if (issues.length === 0) {
    sidecar.nodes.branch.forEach((b, i) => {
      if (!Number.isInteger(b) || b < 0 || b > i)
        issues.push(`node ${i}: branch ${b} is not an earlier node`);
    });
    sidecar.nodes.frequencyHz.forEach((f, i) => {
      if (!(f > 0)) issues.push(`node ${i}: frequencyHz must be > 0`);
    });
    sidecar.nodes.damping.forEach((z, i) => {
      if (!(z > 0 && z < 1)) issues.push(`node ${i}: damping must be in (0, 1)`);
    });
  }
  const plants = (raw as { plants?: unknown }).plants;
  if (plants !== undefined) {
    if (!Array.isArray(plants)) {
      issues.push("plants, when present, must be an array");
    } else {
      (plants as SidecarPlant[]).forEach((plant, k) => {
        const start = plant.nodeStart;
        const end = plant.nodeEnd;
        if (
          !Number.isInteger(start) ||
          !Number.isInteger(end) ||
          start < 0 ||
          end <= start ||
          end > count
        )
          issues.push(`plants[${k}]: node range [${start}, ${end}) is not inside the sidecar`);
        if (!(Number.isFinite(plant.heightM) && plant.heightM > 0))
          issues.push(`plants[${k}]: heightM must be > 0`);
        if (!(Number.isFinite(plant.lengthScaleM) && plant.lengthScaleM > 0))
          issues.push(`plants[${k}]: lengthScaleM must be > 0`);
        const along = plant.turbulence?.along;
        const across = plant.turbulence?.across;
        if (!(Number.isFinite(along) && Number.isFinite(across)))
          issues.push(`plants[${k}]: turbulence must be two numbers`);
        const own = rig?.plants?.[k];
        if (
          rig !== undefined &&
          (own?.nodeStart !== start || own.nodeEnd !== end || own.id !== plant.id)
        )
          issues.push(`plants[${k}]: does not match the rig's plant ${k}`);
      });
      if (rig !== undefined && (rig.plants?.length ?? 0) !== plants.length)
        issues.push(`the rig has ${rig.plants?.length ?? 0} plants, the sidecar ${plants.length}`);
    }
  }
  if (rig !== undefined) {
    if (rig.nodes.length !== count)
      issues.push(`rig has ${rig.nodes.length} nodes, sidecar ${count}`);
    if (rig.canonicalChecksum !== sidecar.rigChecksum)
      issues.push("rigChecksum does not match the rig's canonicalChecksum");
  }
  return issues;
}

/** Parses sidecar JSON. Throws with every problem listed rather than returning a half-trusted one. */
export function parseMotionSidecar(text: string, rig?: MotionRig): MotionSidecar {
  const raw = JSON.parse(text) as MotionSidecar;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw))
    throw new Error("motion sidecar: the top level must be an object");
  const issues = validateMotionSidecar(raw, rig);
  if (issues.length > 0) throw new Error(`invalid motion sidecar:\n  ${issues.join("\n  ")}`);
  return raw;
}
