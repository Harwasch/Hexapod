/**
 * The PlayCanvas back-end of the scan renderer (ScanRendererHost): SuperSplat's own engine
 * (MIT). Each tile is decoded in a worker into the columns PlayCanvas's `GSplatData` takes
 * (playcanvasTile.worker.ts), already in PlayCanvas's Morton order, made a `GSplatResource`
 * within the main thread's frame budget (tileWork.ts), and drawn by PlayCanvas's unified
 * splat renderer, which sorts every tile's splats together. PlayCanvas has no level of detail
 * over resources given this way, so it draws what it is given: streamed at the budget itself.
 *
 * Drawn on demand. PlayCanvas runs its own update loop (streaming, the hand-off of sort
 * results) but only renders when the host draws a frame (`autoRender` off: the frame must be
 * drawn from the globe's camera, in step with it). The loop says when it has something new --
 * a sort result to apply, streamed detail -- with `frame:request`, PlayCanvas's own hook for
 * apps that render on demand, and the host draws a frame for it; `frame:ready` says whether a
 * rendered frame showed every change sorted, which is when a new tile counts as drawn
 * (handover.ts). Once nothing loads or sorts the loop itself pauses, and anything that could
 * give it work (a frame drawn, a sort landing) starts it again.
 *
 * Per-splat data rides as extra resource streams, in the resource's own Morton order, bound to
 * each tile by the checksum of its positions (the worker digests them as it decodes; `order`
 * says where each of the tile's own splats went):
 *
 * - `splatInstance` (R32U): the instance id (scanInstances.ts), for hide and highlight and for
 *   rigid motion;
 * - `splatSkin` (R32U) and `splatWeights` (RGBA32U): the skin id and the `skin.bin` row
 *   (scanMotion.ts), only on tiles `skin.json` lists with skinned splats -- written within the
 *   frame budget (`hooks.work`), as a tile's resource is built; `splatWeights2` (RGBA32U) the
 *   second texel of a wide row (a file with a skin of more than 16 handles).
 *
 * A work-buffer modifier reads them when PlayCanvas copies a tile into the buffer it sorts and
 * draws from: the scene-object colour rule (`modifySplatColor`), and the motion -- skin
 * handles and rigid poses from shared tables -- on the centre (`modifySplatCenter`) and, through
 * the motion's linear part, on the rotation and scales (`modifySplatRotationScale`: the
 * covariance `J·Σ·Jᵀ`). A change re-copies only the tiles that hold what changed; nothing is
 * decoded or uploaded again.
 *
 * Sorting: PlayCanvas sorts on the CPU from each resource's `centers`, which the modifier does
 * not see. A tile holding a rigidly moved object has its centres moved the same way (and its
 * `centersVersion` bumped, which has PlayCanvas re-send them and sort again), at most every
 * `SORT_REFRESH_MS` while it moves -- the host is told when the held-back one is due
 * (`frameDueBy`), so the last pose is sorted even once nothing moves -- and at once when it
 * comes to rest; a skin's sway is sorted at rest, as CesiumJS sorts it. A split object is its
 * own entity, sorted where it is placed.
 *
 * An inferred layer's tile (`loadLayer`, scanLayers.ts) is an entity like any other, sorted
 * with the scan's, but bound to none of its ids, skins or picks: its own modifier draws its
 * look (`PLAYCANVAS_LAYER_MODIFIER`, layerLook.ts), and as the eye moves past its view cones'
 * step (`eyeMoved`) a uniform is set again, which has PlayCanvas copy it -- a few thousand
 * splats -- into the work buffer again in that frame.
 *
 * Confidence and Data age (cellModifier.ts) are one more step of the same modifier, on every
 * tile of the scan: the shared cell look (lib/cellLook.ts) reading the view's texture, at one
 * level of the grid per tile -- picked from the eye as each frame is drawn, the tile copied
 * again only when its level steps. A layer's tiles go grey under either view (no evidence).
 * GLSL only, as the motion: the WebGPU trial does not draw them (lib/viewModes.ts says so).
 *
 * Sunlight (sunHook.ts) is the one look drawn every frame instead: PlayCanvas's own per-frame
 * hook in the user material's `gsplatModifyVS`, lit by the sun the relight hands over
 * (state/sunReport.ts) as uniforms, which copy nothing. The copy pass only leaves it one word per
 * splat (the `hexapodSun` stream of the work buffer, added with the first sun): whether the
 * splat is lit (a layer's is not), and under Confidence or Data age the look drawn for a black
 * splat, with which the hook keeps CesiumJS's order -- the sun before the look. GLSL only too.
 *
 * Selecting in the scene (cesium/sceneSelect) reads the tiles drawn now in each tile's own
 * order (`pickTiles`): the Morton order put back, so a pick's index is the index the tile's
 * ids in `instances.json` are listed by. The copies are made only when a pick first asks for
 * them (a scan with objects), not for every tile of every scan.
 *
 * Two entry points, one renderer. `createBackend` is the default: PlayCanvas's `Application`,
 * which always makes a WebGL2 device. `createWebgpuBackend` is the WebGPU trial
 * (docs/WEBGPU_TRIAL.md): the device is made first and asynchronously, WebGPU preferred,
 * PlayCanvas's own WebGL2 fallback after it, and the app is an `AppBase` with only what this
 * renderer uses (a camera, gsplats, the gsplat and texture asset handlers). On WebGPU four
 * things differ: a modifier is applied only where it has a WGSL form (`WorkBufferModifier`:
 * hide and highlight have one, the motion not yet), so the renderer moves nothing there and
 * says so (no `setMotion`) -- the host draws a scan with objects or motion with WebGL2 instead;
 * PlayCanvas sorts on the GPU in the frame that draws, so no sort result arrives later to ask
 * for the frame that confirms a new tile is drawn -- the renderer asks for that one frame
 * itself (`frameWanted`); and a device can be lost for good (a driver reset, a GPU process
 * crash), which the host answers by drawing with WebGL2 instead (`hooks.deviceLost`).
 */

import * as pc from "playcanvas";

import { CELL_TEXTURE_WIDTH } from "@/lib/cellGrid";
import { CELL_LOOK_GLSL, type CellLookView } from "@/lib/cellLook";
import { NO_OWNER, objectAnchor } from "@/lib/compositor";
import { tileInstanceIds, type InstancesDoc } from "@/lib/instances";
import { createLogger } from "@/lib/log";
import { tileSkin, type SkinDoc } from "@/lib/skin";
import type { PickTile } from "@/lib/splatPick";
import { reportScanSun, watchScanSun, type ScanSun } from "@/state/sunReport";
import type { TileNode } from "@/view/tiles";

import { INSTANCE_TEXTURE_WIDTH } from "../splatInstances";
import { CELL_TILE_GLSL, cellFillOf, CellTiles, type CellTileSample } from "./cellModifier";
import {
  coneEye,
  eyeMoved,
  LAYER_LOOK_GLSL,
  layerEye,
  layerUniforms,
  type LayerCones,
  type LayerLook,
} from "./layerLook";
import { splatMinPixelSize } from "./quality";
import {
  idsInResourceOrder,
  SCAN_INSTANCE_RULE_GLSL,
  SCAN_INSTANCE_RULE_WGSL,
  type InstanceStyle,
} from "./scanInstances";
import {
  MOTION_TEXTURE_WIDTH,
  movedCenters,
  SCAN_MOTION_GLSL,
  type ScanMotion,
} from "./scanMotion";
import { reportScanCellView, watchScanCellView } from "./scanCellView";
import {
  lentRuleKit,
  PLAYCANVAS_FOOTPRINT,
  reportScanRule,
  RULE_OWNER_WIDTH,
  RULE_WEIGHT_GLSL,
  watchScanRule,
  type ScanRuleView,
} from "./scanRule";
import { countOverlayLoopTick } from "./stats";
import { SUN_HOOK_VS, SUN_STREAM, SUN_WORD_GLSL } from "./sunHook";
import type { RuleDates } from "./ruleDates";
import type { BackendHooks, GraphicsApi, ScanBackend, ScanPose, SplatRendererKind } from "./types";

const log = createLogger("playcanvas");

/** No rule: no dates beside the scan (ruleDates.ts). */
const NO_DATES = {
  dates: [],
  plane: [0, 0, 0, 0] as const,
  cellM: 1,
  owners: { data: new Uint8Array(0), height: 0 },
  scanOwns: [],
};

/** Where an object not placed yet is weighed: outside every plane, so it is not drawn. */
const UNPLACED = 1e9;

interface Decoded {
  id: number;
  count?: number;
  /** In Morton order already (`order`): playcanvasTile.worker.ts. */
  properties?: Record<string, Float32Array>;
  /** The tile's centre in the scan's frame, its own frame's offset included: the positions are
   *  relative to it (playcanvasTile.worker.ts). */
  origin?: [number, number, number];
  /** `checksumPositions` of the tile's positions in the scan's frame, before centring. */
  checksum?: string;
  /** Slot `i` holds the tile's own splat `order[i]`. */
  order?: Uint32Array;
  error?: string;
}

/** The per-splat streams. */
const INSTANCE_STREAM = "splatInstance";
const SKIN_STREAM = "splatSkin";
const WEIGHTS_STREAM = "splatWeights";
/** A wide row's second texel (a file with a skin of more than 16 handles). */
const WEIGHTS2_STREAM = "splatWeights2";

/** Shortest time between two re-sorts of a moving object's tiles. */
export const SORT_REFRESH_MS = 100;

/**
 * The work-buffer modifier (PlayCanvas's `gsplatModifyVS`) for a tile with ids, skin weights,
 * or both: the motion on the centre and the covariance, the scene-object rule on the colour;
 * with `cell`, then the Confidence / Data age look (`CELL_TILE_GLSL`), as CesiumJS runs its
 * cell part after the objects' highlight. A tile with neither ids nor a skin under a cell view
 * gets the look alone: the objects' dim is not put on splats no object was found in. With
 * `rule`, last, a per-cell view rule's weight on the opacity (scanRule.ts), from the spread the
 * splat has once the motion moved it; under the rule a split object (`uRuleSource.y`) keeps its
 * own colours, as CesiumJS draws it outside the rule's look.
 */
export function playcanvasModifierGlsl(
  ids: boolean,
  skin: boolean,
  wide = false,
  cell = false,
  sun = false,
  rule = false,
): string {
  const id = ids ? "loadSplatInstance().r" : "0u";
  const second = wide ? "loadSplatWeights2()" : "uvec4(0u)";
  const skinStep = skin
    ? `hexapodSkinMotion(uSkinHandles, loadSplatSkin().r, loadSplatWeights(), ${second}, uMotionParams, uMotionExtra.x, center, delta, hexapodLinear);`
    : "";
  const rigidStep = ids
    ? `hexapodRigidMotion(uRigidSlots, uRigidPoses, ${id}, uMotionParams, center, delta, hexapodLinear);`
    : "";
  // Under a rule nothing returns early: the rule's weight comes last.
  const objectStep = rule
    ? `if (uInstanceParams.x > 0.5) {
        color = hexapodInstanceColor(${id}, color);
    }`
    : `if (uInstanceParams.x < 0.5) {
        return;
    }
    color = hexapodInstanceColor(${id}, color);`;
  // With `sun`, the word the frame's sunlight hook reads (sunHook.ts): a scan splat, lit; under
  // a cell view, with the look drawn for a black splat, so that the hook relights the look of
  // the relit splat (`A + (drawn - A) · factor`, every branch of the look being affine in the
  // colour's scale) -- CesiumJS's order, the sun (3) before the look (5).
  const sunStep =
    sun && !cell ? "writeHexapodSun(uvec4(hexapodSunWord(vec3(0.0)), 0u, 0u, 0u));\n    " : "";
  const cellArgs = "uCellGrid, uCellRule, uCellLevel, uCellTile";
  const blackStep = sun
    ? `writeHexapodSun(uvec4(hexapodSunWord(hexapodCellTile(center, vec4(0.0, 0.0, 0.0, color.a), ${cellArgs}).rgb), 0u, 0u, 0u));
    `
    : "";
  const look = `${blackStep}color = hexapodCellTile(center, color, ${cellArgs});`;
  // A split object under a rule is drawn in its own colours (and, lit, as the photograph is).
  const ruledLook = rule
    ? `if (uRuleSource.y < 0.5) {
        ${look}
    }${
      sun
        ? ` else {
        writeHexapodSun(uvec4(0u, 0u, 0u, 0u));
    }`
        : ""
    }`
    : look;
  const colorStep = !cell
    ? ids || skin
      ? objectStep
      : ""
    : `${
        ids || skin
          ? `if (uInstanceParams.x > 0.5) {
        color = hexapodInstanceColor(${id}, color);
    }
    `
          : ""
      }${ruledLook}`;
  const ruleStep = rule ? "\n    color.a *= hexapodRuleWeight(center);" : "";
  const ruleSpread = rule
    ? `
    hexapodRuleRotation = rotation;
    hexapodRuleScale = scale;`
    : "";
  const cellUniforms = cell
    ? `uniform highp sampler2D uCellGrid;
uniform vec4 uCellRule;
uniform vec4 uCellLevel;
uniform vec4 uCellTile;
${CELL_TILE_GLSL}`
    : "";
  return `
${sun ? SUN_WORD_GLSL : ""}
uniform highp sampler2D uInstanceState;
uniform vec4 uInstanceParams;
uniform vec4 uInstanceTint;
uniform vec4 uInstanceDim;
uniform highp sampler2D uSkinHandles;
uniform highp usampler2D uRigidSlots;
uniform highp sampler2D uRigidPoses;
uniform vec4 uMotionParams;
uniform vec4 uMotionExtra;
${SCAN_INSTANCE_RULE_GLSL}
${SCAN_MOTION_GLSL}
${cellUniforms}
${rule ? RULE_WEIGHT_GLSL : ""}
mat3 hexapodLinear = mat3(0.0);
void modifySplatCenter(inout vec3 center) {
    hexapodLinear = mat3(0.0);
    if (uMotionParams.x < 0.5 && uMotionParams.z < 0.5) {
        return;
    }
    vec3 delta = vec3(0.0);
    ${skinStep}
    ${rigidStep}
    center += delta;
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
    if (uMotionExtra.y > 0.5) {
        hexapodCovariance(mat3(1.0) + hexapodLinear, rotation, scale);
    }${ruleSpread}
}
void modifySplatColor(vec3 center, inout vec4 color) {
    ${sunStep}${colorStep}${ruleStep}
}
`;
}

/**
 * Hide and highlight in WGSL, for PlayCanvas on WebGPU: PlayCanvas's WGSL signatures for the
 * three functions (`gsplatModifyVS`), the colour passed by pointer, and the id stream's
 * `loadSplatInstance()` returning a `vec4u` (an R32U stream reads as `texture_2d<u32>`). The
 * motion has no WGSL port yet: it is GLSL only (`playcanvasModifierGlsl`).
 */
export const PLAYCANVAS_INSTANCE_WGSL = `
uniform uInstanceParams: vec4f;
uniform uInstanceTint: vec4f;
uniform uInstanceDim: vec4f;
var uInstanceState: texture_2d<f32>;
${SCAN_INSTANCE_RULE_WGSL}
fn modifySplatCenter(center: ptr<function, vec3f>) {
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    if (uniform.uInstanceParams.x < 0.5) {
        return;
    }
    *color = hexapodInstanceColor(loadSplatInstance().r, *color);
}
`;

/**
 * A work-buffer modifier in the languages it has: PlayCanvas takes its device's
 * (`GSplatComponent.setWorkBufferModifier`), and one without WGSL is not applied on WebGPU --
 * the tile is drawn as it was decoded.
 */
export interface WorkBufferModifier {
  readonly glsl: string;
  readonly wgsl?: string;
}

const MODIFIERS = new Map<string, WorkBufferModifier>();

/**
 * The modifier for a tile with ids, skin weights, a cell view, the sunlight's word (`sun`:
 * every tile writes it once the work buffer has it), a per-cell view rule, or any of them:
 * always GLSL (the colour rule, the motion, the cell look, the word, the rule's weight); WGSL
 * only for ids alone, and then hide and highlight alone -- a WebGPU renderer moves nothing and
 * draws no cell view, sunlight or rule (see the file comment). One object per kind, so a tile's
 * modifier is compared by identity.
 */
export function playcanvasModifier(
  ids: boolean,
  skin: boolean,
  wide = false,
  cell = false,
  sun = false,
  rule = false,
): WorkBufferModifier {
  const key = `${String(ids)}|${String(skin)}|${String(skin && wide)}|${String(cell)}|${String(sun)}|${String(rule)}`;
  let modifier = MODIFIERS.get(key);
  if (!modifier) {
    modifier = {
      glsl: playcanvasModifierGlsl(ids, skin, skin && wide, cell, sun, rule),
      ...(ids && !skin && !cell && !sun && !rule ? { wgsl: PLAYCANVAS_INSTANCE_WGSL } : {}),
    };
    MODIFIERS.set(key, modifier);
  }
  return modifier;
}

/**
 * The work-buffer modifier of an inferred layer's tile (scanLayers.ts): its look on the colour
 * -- Highlight and the fade of its view cones (layerLook.ts) -- and, while Confidence or Data
 * age is drawn (`uCellFill`), grey first: a layer has no capture evidence, so it is never given
 * a measured colour (cellModifier.ts). With `rule`, last, a per-cell view rule's weight
 * (scanRule.ts): a layer owns a cell only where no date saw it. GLSL only, as the motion: the
 * WebGPU trial draws a scan with inferred layers with WebGL2.
 */
function layerModifierGlsl(sun: boolean, rule = false): string {
  return `
${sun ? SUN_WORD_GLSL : ""}
${rule ? RULE_WEIGHT_GLSL : ""}
uniform vec4 uCellFill;
uniform mat4 uLayerFrame;
uniform vec4 uLayerTint;
uniform vec4 uLayerPattern;
uniform highp sampler2D uLayerCones;
uniform vec4 uLayerConeGrid;
uniform vec4 uLayerConeDims;
uniform vec4 uLayerConeEye;
uniform vec4 uInstanceParams;
uniform vec4 uInstanceDim;
${LAYER_LOOK_GLSL}
${CELL_LOOK_GLSL}
void modifySplatCenter(inout vec3 center) {
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {${
    rule
      ? `
    hexapodRuleRotation = rotation;
    hexapodRuleScale = scale;`
      : ""
  }
}
void modifySplatColor(vec3 center, inout vec4 color) {
    ${sun ? "writeHexapodSun(uvec4(HEXAPOD_SUN_SKIP, 0u, 0u, 0u));\n    " : ""}if (uCellFill.x > 0.5) {
        color = hexapodCellGrey(color);
    }
    color = hexapodLayerColor(center, color, uLayerFrame, uLayerTint, uLayerPattern, uLayerCones,
                              uLayerConeGrid, uLayerConeDims, uLayerConeEye,
                              hexapodLayerDim(uInstanceParams, uInstanceDim));${
                                rule ? "\n    color.a *= hexapodRuleWeight(center);" : ""
                              }
}
`;
}

export const PLAYCANVAS_LAYER_MODIFIER: WorkBufferModifier = { glsl: layerModifierGlsl(false) };

/**
 * The layer's modifier once the work buffer carries the sunlight's word: the skip bit first,
 * so the frame's hook leaves the layer as drawn -- it is not the capture (sunHook.ts).
 */
export const PLAYCANVAS_LAYER_SUN_MODIFIER: WorkBufferModifier = { glsl: layerModifierGlsl(true) };

/** The layer's modifiers under a per-cell view rule, without and with the sunlight's word. */
const LAYER_RULE_MODIFIERS: readonly WorkBufferModifier[] = [
  { glsl: layerModifierGlsl(false, true) },
  { glsl: layerModifierGlsl(true, true) },
];

/** The decoded columns a pick copy is made from (lib/splatLayout.ts names). */
type PickColumns = Readonly<Record<string, Float32Array>>;

/**
 * A tile's splats for picking (lib/splatPick.ts), in the tile's own order: the resource's
 * columns are in Morton order -- slot `k` holds the tile's splat `order[k]` -- so each is put
 * back at `order[k]`, and moved off the tile's centre (`origin`) into the scan's frame. A pick
 * then returns the index `instances.json` lists the tile's ids by. Null when a column is
 * missing.
 */
export function pickTileOf(
  columns: PickColumns,
  order: Uint32Array,
  origin: readonly [number, number, number],
  checksum: string,
): PickTile | null {
  const { x, y, z, scale_0: s0, scale_1: s1, scale_2: s2, opacity: alpha } = columns;
  if (!x || !y || !z || !s0 || !s1 || !s2 || !alpha) return null;
  const count = order.length;
  const [ox, oy, oz] = origin;
  const positions = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const opacity = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const i = order[k] ?? 0;
    positions[i * 3] = (x[k] ?? 0) + ox;
    positions[i * 3 + 1] = (y[k] ?? 0) + oy;
    positions[i * 3 + 2] = (z[k] ?? 0) + oz;
    radii[i] = Math.max(s0[k] ?? 0, s1[k] ?? 0, s2[k] ?? 0);
    opacity[i] = alpha[k] ?? 0;
  }
  return { checksum, count, positions, radii, opacity };
}

/** `pick` moved by `matrix` (a split object's placement in the scan frame). */
function placedPickTile(pick: PickTile, matrix: pc.Mat4): PickTile {
  const positions = new Float32Array(pick.positions.length);
  const point = new pc.Vec3();
  for (let i = 0; i < pick.count; i++) {
    point.set(
      pick.positions[i * 3] ?? 0,
      pick.positions[i * 3 + 1] ?? 0,
      pick.positions[i * 3 + 2] ?? 0,
    );
    matrix.transformPoint(point, point);
    positions[i * 3] = point.x;
    positions[i * 3 + 1] = point.y;
    positions[i * 3 + 2] = point.z;
  }
  return { ...pick, positions };
}

/** What binds a tile to the scan's objects: its digest, where each original splat went, and
 *  what was written from which document. */
interface TileBinding {
  checksum: string;
  /** Resource slot `i` holds the tile's splat `order[i]` (the worker's Morton reorder). */
  order: Uint32Array;
  resource: Resource;
  /** The decoded columns (Morton order; the resource's own data, not a copy). */
  columns: PickColumns;
  /** Where the entity sits (the tile's centre): resource centres are relative to it. */
  origin: [number, number, number];
  /** The doc the ids were written from, whether the file lists the tile, and the ids. */
  doc: InstancesDoc | null;
  matched: boolean;
  /** The tile's ids in resource order, when the file lists it. */
  ids: Uint32Array | null;
  idSet: ReadonlySet<number>;
  /** The skin doc the skin streams are (being) written from, and the skins the tile holds. */
  skinDoc: SkinDoc | null;
  skinned: boolean;
  /** Its weights take two texels a splat (`splatWeights2` written). */
  skinWide: boolean;
  skinSet: ReadonlySet<number>;
  /** The modifier on it now. */
  modifier: WorkBufferModifier | null;
  /** The tile's own centres, kept while a motion has moved them for the sorter. */
  restCenters: Float32Array | null;
  /** The tile's splats for picking, in its own order: made on first need (`pickTiles`). */
  pick: PickTile | null | undefined;
  /** Where `place` last put the tile (a split object at its pose), or null where decoded. */
  placement: pc.Mat4 | null;
  /** `pick` at `placement`: made on first need, so a moving object costs no copy a frame. */
  placedPick: PickTile | null;
}

/** An inferred layer's tile as the renderer keeps it: its look and the eye its fade is from. */
interface LayerState {
  look: LayerLook;
  eye: Float32Array<ArrayBuffer>;
  lastEye: number[] | null;
  centre: number[];
}

/** Frames a disposed tile's GPU resource outlives its entity: PlayCanvas's unified renderer
 *  drops a removed entity from its placements in the next frame it renders, and destroying
 *  the resource first left a placement with none ("Cannot read properties of null (reading
 *  'hasCenters')"). Counted in frames rendered, not loop ticks: a removal is only processed
 *  by a render -- so while one is waiting, the renderer asks the host for those frames
 *  (`frameWanted`): a tile evicted at rest (an off-screen one dropped from the cache at the
 *  end of a plan, a load that landed after it was abandoned) had no frames coming, its
 *  resource was never freed, and PlayCanvas's loop, which pauses only once nothing is
 *  waiting, ticked every display frame for as long as the page stayed open. */
const DESTROY_AFTER_FRAMES = 3;

/**
 * When PlayCanvas cannot say (no `frame:ready`, which it fires for every rendered frame with
 * splats in it), a new entity counts as drawn this many rendered frames and milliseconds after
 * it was added: the unified renderer copies it into its work buffer and sorts in a worker
 * first.
 */
const SETTLE_FRAMES = 4;
const SETTLE_MS = 120;

/** Loop ticks in a row with nothing to load, sort or show before PlayCanvas's loop pauses. */
export const QUIET_TICKS_BEFORE_PAUSE = 10;

/** Tiles decoded at once: a quarter of the cores, one to three. */
function workerCount(): number {
  const cores = navigator.hardwareConcurrency || 4;
  return Math.min(3, Math.max(1, Math.floor(cores / 4)));
}

/** The `frame:ready` handler's arguments (gsplat/system.d.ts EVENT_FRAMEREADY). */
type FrameReady = (camera: unknown, layer: unknown, ready: boolean, loadingCount: number) => void;

/** The resource members used here that the typings do not show. */
type Resource = pc.GSplatResource & {
  format: { addExtraStreams(streams: { name: string; format: number }[]): void };
  getTexture(name: string): pc.Texture | null;
  centers: Float32Array | null;
  centersVersion: number;
};

/** The overlay's canvas: transparent, premultiplied, no multisampling (splats are smooth). */
const DEVICE_OPTIONS = {
  alpha: true,
  antialias: false,
  premultipliedAlpha: true,
  powerPreference: "high-performance",
} as const;

/** The device options for `hooks`: the drawn frame kept readable when a harness asks. */
function deviceOptions(hooks: BackendHooks) {
  return { ...DEVICE_OPTIONS, preserveDrawingBuffer: hooks.preserveDrawingBuffer === true };
}

/** PlayCanvas on WebGL2, the default: its `Application`, which always makes a WebGL2 device. */
export function createBackend(
  canvas: HTMLCanvasElement,
  budget: number,
  hooks: BackendHooks,
): Promise<ScanBackend<pc.Entity>> {
  const app = new pc.Application(canvas, { graphicsDeviceOptions: deviceOptions(hooks) });
  return Promise.resolve(assemble(app, budget, hooks, { name: "playcanvas", note: null }));
}

/**
 * Why WebGPU was not used when PlayCanvas fell back to WebGL2 by itself (it logs the reason
 * and moves on): the browser has none, the page is not a secure context, or no adapter or
 * device came (PlayCanvas also declines PowerVR GPUs' WebGPU).
 */
export function whyNoWebgpu(
  gpu: unknown = typeof navigator === "undefined"
    ? undefined
    : (navigator as { gpu?: unknown }).gpu,
  secure: boolean = typeof window === "undefined" || window.isSecureContext,
): string {
  if (!gpu) return secure ? "this browser has no WebGPU" : "WebGPU needs a secure (https) page";
  return "no WebGPU adapter or device";
}

/**
 * PlayCanvas on WebGPU, the trial (docs/WEBGPU_TRIAL.md). `createGraphicsDevice` tries WebGPU
 * and then WebGL2 on the same canvas -- WebGPU takes the canvas only once its adapter and
 * device exist, so a WebGL2 fallback still can -- and after that, silently, a Null device that
 * draws nothing. A Null device is refused here (the host then draws with `createBackend` on a
 * fresh canvas); a WebGL2 one is used, and says why (`apiNote`).
 */
export async function createWebgpuBackend(
  canvas: HTMLCanvasElement,
  budget: number,
  hooks: BackendHooks,
): Promise<ScanBackend<pc.Entity>> {
  const device = (await pc.createGraphicsDevice(canvas, {
    ...deviceOptions(hooks),
    deviceTypes: [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2],
    // Not for a headset: an XR-compatible adapter can be another GPU than the display's.
    xrCompatible: false,
  })) as pc.GraphicsDevice;
  if (device.isNull || (!device.isWebGPU && !device.isWebGL2)) {
    device.destroy();
    throw new Error("PlayCanvas started neither WebGPU nor WebGL2 (its Null device draws nothing)");
  }
  // `Application` would make a WebGL2 device of its own: the app is assembled around this one,
  // with only what the renderer uses: a camera, gsplats, and the handlers a scan's streamed
  // package loads through (its octree and SOG chunks are gsplat assets; each chunk's WebP
  // planes are texture assets -- without that handler a native scan loads nothing).
  const app = new pc.AppBase(canvas);
  const options = new pc.AppOptions();
  options.graphicsDevice = device;
  options.componentSystems = [pc.CameraComponentSystem, pc.GSplatComponentSystem];
  options.resourceHandlers = [pc.GSplatHandler, pc.TextureHandler];
  app.init(options);
  return assemble(app, budget, hooks, {
    name: "playcanvas-webgpu",
    note: device.isWebGPU ? null : `WebGPU unavailable: ${whyNoWebgpu()}`,
  });
}

/** What a lost WebGPU device says (`GPUDeviceLostInfo`). */
interface LostInfo {
  reason: string;
  message: string;
}

/** The `lost` promise of a WebGPU device (`GPUDevice.lost`); null on WebGL2. */
function webgpuLost(device: pc.GraphicsDevice): Promise<LostInfo> | null {
  const wgpu = (device as unknown as { wgpu?: { lost?: Promise<LostInfo> } }).wgpu;
  return wgpu?.lost ?? null;
}

/** Everything but making the app and its device: the same renderer on either API. */
function assemble(
  app: pc.AppBase,
  budget: number,
  hooks: BackendHooks,
  identity: { name: SplatRendererKind; note: string | null },
): ScanBackend<pc.Entity> {
  const device = app.graphicsDevice;
  const api: GraphicsApi = device.isWebGPU ? "webgpu" : "webgl2";
  // On WebGPU PlayCanvas sorts on the GPU, in the frame that draws (gsplat-params.js,
  // `_resolveRenderer`): no sort result comes back later to ask for the frame that confirms a
  // new tile is drawn, so this renderer asks for it itself (`render`).
  const sortsOnGpu = device.isWebGPU;
  /** The motion modifier is GLSL only (`playcanvasModifier`): on WebGPU nothing moves. */
  const moves = api === "webgl2";
  /** The scan's tiles under a cell view, and the levels they read: GLSL only, as the motion. */
  const cells = moves ? new CellTiles<pc.Entity>() : null;
  let cellTexture: pc.Texture | null = null;
  /** `(1, 0, 0, 0)` while a cell view is drawn: what has a layer's tiles drawn grey. */
  const cellFill = new Float32Array(4);
  /** The modifier on each entity now (a tile with no binding has one only for a cell view). */
  const modifiers = new WeakMap<pc.Entity, WorkBufferModifier | null>();
  /**
   * A per-cell view rule (scanRule.ts), GLSL only as the motion: the rule drawn now, or null;
   * its owner texture and plane; and the scan's own cell view, which Data age under the rule
   * stands in for while it is on.
   */
  let rule: ScanRuleView | null = null;
  let ruleOwners: pc.Texture | null = null;
  const rulePlane = new Float32Array(4);
  const ruleParams = new Float32Array(4);
  let ownCellView: CellLookView | null = null;
  /** The rule's other dates, streamed into this app (ruleDates.ts, loaded with the first). */
  let ruleDates: RuleDates<pc.Entity> | null = null;
  let ruleDatesLoading = false;
  /** The budget the host draws to (`setBudget`), which a rule's dates share. */
  let drawBudget = budget;
  /** The tileset each entity came from: whose source it is under a rule. */
  const fromUrl = new WeakMap<pc.Entity, string>();
  /** A natively streamed scan's one entity: the rule's `native` source. */
  const natives = new WeakSet<pc.Entity>();
  /**
   * Where each placed entity's own origin is in the scan's frame: for a split object, the
   * origin its package declares at its pose (lib/compositor.ts `objectAnchor`), by which it is
   * weighed whole.
   */
  const anchors = new WeakMap<pc.Entity, [number, number]>();
  const ruleSources = new WeakMap<pc.Entity, Float32Array>();
  /**
   * Sunlight (state/sunReport.ts, sunHook.ts), GLSL only as the motion: the sun to light by now,
   * whether the work buffer carries the word the copy pass writes for it -- added with the
   * first sun and kept, a stream not being taken back -- and whether the frame's hook is in.
   */
  let sun: ScanSun | null = null;
  let sunStream = false;
  let sunHooked = false;
  let sunSet: ScanSun | null = null;
  let sunFrames = 0;
  /** PlayCanvas's own final-draw chunk, put back when the hook comes out. */
  let ownModifyVS: string | undefined;
  const sunValue = new Float32Array(4);
  const sunlightValue = new Float32Array(4);
  /** The device is gone for good: nothing more is drawn, and the host replaces the renderer. */
  let lost = false;
  /** The renderer is letting go of its device itself: that loss is not news. */
  let destroying = false;
  // PlayCanvas answers a lost WebGPU device by making another on the same canvas. A loss on a
  // phone is mostly memory pressure or a GPU process restart, which a new WebGPU device meets
  // again; the trial's answer is the API every device here has drawn with, WebGL2, on a fresh
  // canvas (the host's), so a lost device costs the scan's tiles once, not the view.
  void webgpuLost(device)?.then((info) => {
    if (destroying) return;
    lost = true;
    hooks.deviceLost?.(`WebGPU device lost: ${info.message || info.reason}`);
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.scene.gsplat.splatBudget = budget;
  // Full-precision work buffer rather than the compact one (quantised transforms): the tiles
  // are already the detail the view asked for. The smallest splat kept is set per resolution,
  // and to none from afar (`splatMinPixelSize`, in render).
  (app.scene.gsplat as unknown as { dataFormat: string }).dataFormat = "large";
  const camera = new pc.Entity("scan-camera");
  camera.addComponent("camera", { clearColor: new pc.Color(0, 0, 0, 0) });
  app.root.addChild(camera);
  // PlayCanvas updates in its own loop (streaming, sorting) but draws only from `render`, in
  // the same frame and from the same pose as the globe under it. Drawn in its own loop, it ran
  // before the host set the camera: a pose behind, so the scan slid on the map as the view
  // moved, and a resize cleared the canvas a frame before anything was drawn on it (a black
  // flash where the world is clipped away under the scan).
  app.autoRender = false;

  // The loop pauses when it has nothing to do (see the file comment): its next animation frame
  // is simply not asked for, and `resumeLoop` asks for it again.
  let looping = true;
  let quietTicks = 0;
  /** The last rendered frame showed every change sorted, with nothing left loading. */
  let settled = false;
  const tickAgain = app.requestAnimationFrame.bind(app);
  app.requestAnimationFrame = () => {
    if (looping) tickAgain();
  };
  const resumeLoop = (): void => {
    quietTicks = 0;
    if (looping) return;
    looping = true;
    tickAgain();
  };
  app.start();

  const workers = Array.from(
    { length: workerCount() },
    () => new Worker(new URL("./playcanvasTile.worker.ts", import.meta.url), { type: "module" }),
  );
  const waiting = new Map<number, (message: Decoded) => void>();
  let nextId = 1;
  let turn = 0;
  for (const worker of workers) {
    worker.onmessage = (event: MessageEvent<Decoded>) => {
      waiting.get(event.data.id)?.(event.data);
      waiting.delete(event.data.id);
    };
  }
  /** A tile decoded off the main thread, `offset` its frame in the scan's (`TileNode.offset`). */
  const decode = (url: string, offset?: [number, number, number]): Promise<Decoded> =>
    new Promise((resolve) => {
      const id = nextId++;
      waiting.set(id, resolve);
      const worker = workers[turn++ % workers.length];
      worker?.postMessage({ id, url, maxSh: hooks.maxShDegree, ...(offset ? { offset } : {}) });
    });

  const dataTexture = (
    name: string,
    width: number,
    height: number,
    format: number,
    data?: Uint8Array | Uint32Array | Float32Array,
  ): pc.Texture => {
    const texture = new pc.Texture(device, {
      name,
      width,
      height,
      format,
      mipmaps: false,
      minFilter: pc.FILTER_NEAREST,
      magFilter: pc.FILTER_NEAREST,
      addressU: pc.ADDRESS_CLAMP_TO_EDGE,
      addressV: pc.ADDRESS_CLAMP_TO_EDGE,
    });
    if (data) {
      (texture.lock() as typeof data).set(data);
      texture.unlock();
    }
    return texture;
  };
  /** Stand-ins for tables not there yet: every sampler the GLSL modifier declares is bound. */
  let empty: { state: pc.Texture; float: pc.Texture; uint: pc.Texture } | null = null;
  const stand = (): { state: pc.Texture; float: pc.Texture; uint: pc.Texture } =>
    (empty ??= {
      state: dataTexture("hexapodEmptyState", 1, 1, pc.PIXELFORMAT_RGBA8, new Uint8Array(4)),
      float: dataTexture("hexapodEmptyFloat", 1, 1, pc.PIXELFORMAT_RGBA32F, new Float32Array(4)),
      uint: dataTexture("hexapodEmptyUint", 1, 1, pc.PIXELFORMAT_RGBA32U, new Uint32Array(4)),
    });

  const resources = new WeakMap<pc.Entity, pc.GSplatResource>();
  /** Tiles loaded and not yet disposed, with what binds them to the scan's objects. */
  const tiles = new Map<pc.Entity, TileBinding>();
  /** Where each entity was decoded to sit (its tile's centre), for `place`. */
  const origins = new WeakMap<pc.Entity, [number, number, number]>();
  /** Tiles on screen now (added, not removed): what a pick reads. */
  const shown = new Set<pc.Entity>();
  let style: InstanceStyle | null = null;
  let motion: ScanMotion | null = null;
  let stateTexture: pc.Texture | null = null;
  let handlesTexture: pc.Texture | null = null;
  let slotsTexture: pc.Texture | null = null;
  let posesTexture: pc.Texture | null = null;
  const params = new Float32Array(4);
  const tint = new Float32Array(4);
  const dim = new Float32Array(4);
  const motionParams = new Float32Array(4);
  const motionExtra = new Float32Array([0, 1, 0, 0]);
  let redrawn = 0;
  /** Tiles whose centres must follow a motion for the sorter, and when they last did. */
  const sortDirty = new Set<pc.Entity>();
  let lastSortAt = -Infinity;

  /** The instances whose ids tiles carry: hide and highlight's, or the motion's. */
  const instancesDoc = (): InstancesDoc | null => style?.doc ?? motion?.instances ?? null;

  const bindIds = (tile: TileBinding): void => {
    const doc = instancesDoc();
    if (!doc || tile.doc === doc) return;
    const ids = tileInstanceIds(doc, tile.checksum);
    const listed = ids?.length === tile.order.length ? ids : undefined;
    const resource = tile.resource;
    resource.format.addExtraStreams([{ name: INSTANCE_STREAM, format: pc.PIXELFORMAT_R32U }]);
    const texture = resource.getTexture(INSTANCE_STREAM);
    if (!texture) return;
    const ordered = idsInResourceOrder(listed, tile.order, texture.lock() as Uint32Array);
    // The texture is padded past the tile's splats: what the tile holds is the first `count`.
    const own = ordered.subarray(0, tile.order.length);
    tile.ids = listed ? Uint32Array.from(own) : null;
    tile.idSet = new Set(own);
    texture.unlock();
    tile.doc = doc;
    tile.matched = listed !== undefined;
  };

  /** Writes the tile's skin and weight streams from `doc`: main-thread work (`hooks.work`). */
  const writeSkin = (tile: TileBinding, doc: SkinDoc): void => {
    const found = tileSkin(doc, tile.checksum);
    const skinSet = new Set<number>();
    const fits = found?.skins.length === tile.order.length;
    if (fits) for (const id of found.skins) if (id !== 0) skinSet.add(id);
    tile.skinSet = skinSet;
    const resource = tile.resource;
    if (!fits || skinSet.size === 0) {
      // Listed before, not now: its skins are zeroed (a stream cannot be taken back).
      const skins = tile.skinned ? resource.getTexture(SKIN_STREAM) : null;
      if (skins) {
        (skins.lock() as Uint32Array).fill(0);
        skins.unlock();
      }
      return;
    }
    const second = found.words2;
    resource.format.addExtraStreams([
      { name: SKIN_STREAM, format: pc.PIXELFORMAT_R32U },
      { name: WEIGHTS_STREAM, format: pc.PIXELFORMAT_RGBA32U },
      ...(second ? [{ name: WEIGHTS2_STREAM, format: pc.PIXELFORMAT_RGBA32U }] : []),
    ]);
    const skins = resource.getTexture(SKIN_STREAM);
    const weights = resource.getTexture(WEIGHTS_STREAM);
    const weights2 = second ? resource.getTexture(WEIGHTS2_STREAM) : undefined;
    if (!skins || !weights || (second && !weights2)) return;
    const skinOut = skins.lock() as Uint32Array;
    const wordOut = weights.lock() as Uint32Array;
    const word2Out = weights2 ? (weights2.lock() as Uint32Array) : undefined;
    skinOut.fill(0);
    wordOut.fill(0);
    word2Out?.fill(0);
    for (let i = 0; i < tile.order.length; i += 1) {
      const from = tile.order[i] ?? 0;
      skinOut[i] = found.skins[from] ?? 0;
      for (let k = 0; k < 4; k += 1) wordOut[i * 4 + k] = found.words[from * 4 + k] ?? 0;
      if (word2Out && second)
        for (let k = 0; k < 4; k += 1) word2Out[i * 4 + k] = second[from * 4 + k] ?? 0;
    }
    skins.unlock();
    weights.unlock();
    weights2?.unlock();
    tile.skinned = true;
    tile.skinWide = second !== undefined;
  };

  /**
   * Which source of the rule `entity` is (`uRuleSource`): the rule names its tileset (a natively
   * streamed scan is `native`), and a split object is weighed whole by its centre. A tileset the
   * rule does not name is no source: its splats are not drawn.
   */
  const ruleSourceOf = (entity: pc.Entity): Float32Array => {
    let out = ruleSources.get(entity);
    if (!out) ruleSources.set(entity, (out = new Float32Array(4)));
    const url = fromUrl.get(entity);
    const found = !rule
      ? undefined
      : natives.has(entity)
        ? { source: rule.native }
        : url === undefined
          ? undefined
          : rule.sources.get(url);
    // An object not placed yet has no anchor: outside every plane, it is not drawn.
    const [x, y] = anchors.get(entity) ?? [UNPLACED, UNPLACED];
    if (found?.whole) out.set([found.source, 1, x, y]);
    else out.set([found?.source ?? NO_OWNER, 0, 0, 0]);
    return out;
  };

  /** Whether the rule tints `entity` as invented (Highlight invented: an inferred source). */
  const ruleTinted = (entity: pc.Entity): boolean => {
    const source = ruleSourceOf(entity)[0] ?? NO_OWNER;
    return rule !== null && source < 31 && ((rule.tinted >> source) & 1) === 1;
  };

  /** The rule's uniforms on `component`, while one is drawn. */
  const setRuleUniforms = (component: pc.GSplatComponent, entity: pc.Entity): void => {
    if (!rule) return;
    component.setParameter("uRuleOwners", ruleOwners ?? stand().state);
    component.setParameter("uRulePlane", rulePlane);
    component.setParameter("uRuleParams", ruleParams);
    component.setParameter("uRuleSource", ruleSourceOf(entity));
  };

  /** Sets every uniform the modifier reads; setting any marks the tile for a new copy. */
  const setUniforms = (component: pc.GSplatComponent, entity: pc.Entity): void => {
    const cell = cells?.uniforms(entity);
    if (cells && cell) {
      component.setParameter("uCellGrid", cellTexture ?? stand().state);
      component.setParameter("uCellRule", cells.rule);
      component.setParameter("uCellLevel", cell.level);
      component.setParameter("uCellTile", cell.tile);
    }
    component.setParameter("uInstanceState", stateTexture ?? stand().state);
    component.setParameter("uInstanceTint", tint);
    component.setParameter("uInstanceDim", dim);
    if (moves) {
      component.setParameter("uSkinHandles", handlesTexture ?? stand().float);
      component.setParameter("uRigidSlots", slotsTexture ?? stand().uint);
      component.setParameter("uRigidPoses", posesTexture ?? stand().float);
      component.setParameter("uMotionExtra", motionExtra);
      component.setParameter("uMotionParams", motionParams);
    }
    setRuleUniforms(component, entity);
    // Last: setting a parameter marks the tile for a new copy into the work buffer.
    component.setParameter("uInstanceParams", params);
  };

  /**
   * Puts the modifier the tile's streams and the cell view call for on it, with the current
   * uniforms; none when it needs none (a cell view ended on a tile with neither ids nor a skin).
   */
  const applyModifier = (entity: pc.Entity): void => {
    const component = entity.gsplat;
    if (!component) return;
    const tile = tiles.get(entity);
    const hasIds = (tile?.doc ?? null) !== null;
    const skinned = tile?.skinned ?? false;
    const cell = cells?.drawing === true && cells.has(entity);
    // Once the work buffer carries the sunlight's word, every scan tile's copy writes it.
    const lit = sunStream && cells?.has(entity) === true;
    // Under a rule every tile is weighed by it, the scan's own and the other dates'.
    const ruled = rule !== null;
    const modifier =
      hasIds || skinned || cell || lit || ruled
        ? playcanvasModifier(hasIds, skinned, tile?.skinWide ?? false, cell, lit, ruled)
        : null;
    if (modifier !== (modifiers.get(entity) ?? null)) {
      modifiers.set(entity, modifier);
      if (tile) tile.modifier = modifier;
      component.setWorkBufferModifier(modifier);
    }
    if (modifier) setUniforms(component, entity);
  };

  /**
   * Writes what the tile needs for the current style and motion and puts the modifier on. The
   * skin streams -- a pass over every splat of the tile -- are written within the frame's
   * budget (`hooks.work`): a new skin document (the wind switched on) rebinds every tile. A
   * tile being built is already a job of that budget (`inWork`), and writes them at once.
   */
  const bind = (entity: pc.Entity, tile: TileBinding, inWork = false): void => {
    if (!entity.gsplat) return;
    bindIds(tile);
    const skinDoc = motion?.skin ?? null;
    if (skinDoc && tile.skinDoc !== skinDoc) {
      // Claimed now, so a motion handed meanwhile does not queue it again.
      tile.skinDoc = skinDoc;
      if (inWork) {
        writeSkin(tile, skinDoc);
      } else {
        void hooks.work
          .run(() => {
            if (tiles.get(entity) !== tile || tile.skinDoc !== skinDoc) return;
            writeSkin(tile, skinDoc);
            applyModifier(entity);
            hooks.frameWanted();
          })
          .catch(() => undefined);
      }
    }
    applyModifier(entity);
  };

  const setInstances = (next: InstanceStyle | null): void => {
    style = next;
    if (next) {
      if (stateTexture?.height !== next.rows) {
        stateTexture?.destroy();
        stateTexture = dataTexture(
          "instanceState",
          INSTANCE_TEXTURE_WIDTH,
          next.rows,
          pc.PIXELFORMAT_RGBA8,
        );
      }
      (stateTexture.lock() as Uint8Array).set(next.state);
      stateTexture.unlock();
      params.set(next.params);
      tint.set(next.tint);
      dim.set(next.dim);
    } else {
      params[0] = 0;
    }
    for (const [entity, tile] of tiles) bind(entity, tile);
    // A layer's tiles are dimmed with the rest while an object is highlighted.
    for (const entity of layers.keys()) {
      entity.gsplat?.setParameter("uInstanceDim", dim);
      entity.gsplat?.setParameter("uInstanceParams", params);
    }
  };

  /** Uploads `data` into `texture`, or into a new texture when its size changed. */
  const table = (
    texture: pc.Texture | null,
    name: string,
    rows: number,
    format: number,
    data: Float32Array | Uint32Array,
  ): { texture: pc.Texture; replaced: boolean } => {
    if (texture?.height === rows) {
      (texture.lock() as typeof data).set(data);
      texture.unlock();
      return { texture, replaced: false };
    }
    texture?.destroy();
    return {
      texture: dataTexture(name, MOTION_TEXTURE_WIDTH, rows, format, data),
      replaced: true,
    };
  };

  const setMotion = (next: ScanMotion | null): void => {
    const before = motion;
    motion = next;
    let replaced = false;
    if (next) {
      const handles = table(
        handlesTexture,
        "hexapodSkinHandles",
        next.handleRows,
        pc.PIXELFORMAT_RGBA32F,
        next.handles,
      );
      const slots = table(
        slotsTexture,
        "hexapodRigidSlots",
        next.slotRows,
        pc.PIXELFORMAT_RGBA32U,
        next.slots,
      );
      const poses = table(
        posesTexture,
        "hexapodRigidPoses",
        next.poseRows,
        pc.PIXELFORMAT_RGBA32F,
        next.poses,
      );
      handlesTexture = handles.texture;
      slotsTexture = slots.texture;
      posesTexture = poses.texture;
      replaced = handles.replaced || slots.replaced || poses.replaced;
      motionParams.set(next.params);
      motionExtra.set(next.extra);
    } else {
      motionParams.fill(0);
    }
    const covarianceChanged = (before?.extra[1] ?? 1) !== (next?.extra[1] ?? 1);
    for (const [entity, tile] of tiles) {
      // A new table, a new document, or motion switched on or off: every tile binds again.
      const rebind =
        replaced ||
        next === null ||
        before === null ||
        (next.skin !== null && tile.skinDoc !== next.skin) ||
        (next.instances !== null && tile.doc === null);
      const touched =
        rebind ||
        covarianceChanged ||
        [...next.changedSkins].some((id) => tile.skinSet.has(id)) ||
        [...next.changedIds].some((id) => tile.idSet.has(id));
      if (!touched) continue;
      if (rebind) bind(entity, tile);
      else if (tile.modifier !== null) entity.gsplat?.setParameter("uMotionParams", motionParams);
      if (tile.modifier !== null) redrawn += 1;
      if (tile.ids) sortDirty.add(entity);
    }
  };

  /** Moves the centres of tiles holding rigidly moved objects, for the sorter. */
  const refreshSort = (now: number): void => {
    if (sortDirty.size === 0) return;
    const resting = motion === null || motion.rigidByLeaf.size === 0;
    // Held back while it moves: `frameDueBy` has the host draw the frame it is due in.
    if (!resting && now - lastSortAt < SORT_REFRESH_MS) return;
    lastSortAt = now;
    for (const entity of sortDirty) {
      const tile = tiles.get(entity);
      const resource = tile?.resource;
      if (!tile?.ids || !resource?.centers) continue;
      if (tile.restCenters === null) {
        if (resting) continue;
        tile.restCenters = Float32Array.from(resource.centers);
      }
      const moved = movedCenters(
        tile.restCenters,
        tile.origin,
        tile.ids,
        motion?.rigidByLeaf ?? new Map<number, Float64Array>(),
        resource.centers,
      );
      if (!moved) tile.restCenters = null;
      resource.centersVersion += 1;
    }
    sortDirty.clear();
  };

  /** Frames rendered so far, the last one that showed every change sorted (`frame:ready`),
   *  and the frame count when each entity was added. */
  let framesDrawn = 0;
  let lastReadyFrame = -1;
  let frameReadySeen = false;
  const addedAt = new WeakMap<pc.Entity, number>();
  /** Added and not yet confirmed drawn by a ready frame two frames on (`isDrawn`). */
  const unconfirmed = new Set<pc.Entity>();
  const doomed: { resource: pc.GSplatResource; at: number }[] = [];
  const target = new pc.Vec3();
  const up = new pc.Vec3();
  /** Scratch for `place` (split objects only). */
  let placing: { matrix: pc.Mat4; position: pc.Vec3; rotation: pc.Quat } | null = null;
  let size = { width: 0, height: 0, pixelRatio: 0 };

  /**
   * The tiles of inferred layers (scanLayers.ts): bound to none of the scan's objects, each
   * with its layer's look and the eye it was last worked out from. `lastEye` and `centre` (the
   * tile's origin) are in the layer's frame, where `eyeMoved` measures (`layerEye`).
   */
  const layers = new Map<pc.Entity, LayerState>();
  /** Each layer's view cones, uploaded once whatever number of its tiles are drawn. */
  const coneTextures = new Map<LayerCones, pc.Texture>();
  /** The eye of the last frame drawn (the scan's frame): a layer added later starts from it. */
  let lastPoseEye: [number, number, number] | null = null;

  /**
   * Every uniform a layer's modifier reads, from its look -- Highlight's, or the rule's tint of
   * an invented source (Highlight invented) -- and the rule's while one is drawn.
   */
  const layerParameters = (entity: pc.Entity, state: LayerState): void => {
    const component = entity.gsplat;
    if (!component) return;
    const look =
      !state.look.highlight && ruleTinted(entity) ? { ...state.look, highlight: true } : state.look;
    const values = layerUniforms(look);
    component.setParameter("uLayerFrame", values.frame);
    component.setParameter("uLayerTint", values.tint);
    component.setParameter("uLayerPattern", values.pattern);
    component.setParameter("uLayerCones", coneTexture(look.cones));
    component.setParameter("uLayerConeGrid", values.grid);
    component.setParameter("uLayerConeDims", values.dims);
    // The objects' dim while one is highlighted: a layer belongs to none (layerLook.ts).
    component.setParameter("uInstanceDim", dim);
    component.setParameter("uInstanceParams", params);
    component.setParameter("uCellFill", cellFill);
    setRuleUniforms(component, entity);
    // Last: setting a parameter has the tile copied into the work buffer again.
    state.lastEye = lastPoseEye ? layerEye(state.look, lastPoseEye) : null;
    component.setParameter("uLayerConeEye", coneEye(look, lastPoseEye ?? [0, 0, 0], state.eye));
  };

  /** The cones' texture, made on first need; a stand-in for a layer without cones. */
  const coneTexture = (cones: LayerCones | null): pc.Texture => {
    if (!cones) return stand().state;
    let texture = coneTextures.get(cones);
    if (!texture) {
      texture = dataTexture(
        "hexapodLayerCones",
        cones.data.length / 4 / cones.height,
        cones.height,
        pc.PIXELFORMAT_RGBA8,
        cones.data,
      );
      coneTextures.set(cones, texture);
    }
    return texture;
  };

  /** Frees the cones' texture once no layer tile draws with it. */
  const releaseCones = (cones: LayerCones | null): void => {
    if (!cones) return;
    for (const state of layers.values()) if (state.look.cones === cones) return;
    coneTextures.get(cones)?.destroy();
    coneTextures.delete(cones);
  };

  /**
   * Draws `view` -- Confidence or Data age, its texture and the grid's rule -- on every tile of
   * the scan, or the photo again with null (cellModifier.ts); a layer's tiles grey while one is
   * drawn. Each tile is copied again once, in the frame this asks for.
   */
  const setCellView = (view: CellLookView | null): void => {
    if (!cells) return;
    const before = cells.view;
    if (!cells.setView(view)) return;
    // Photo frees the grid's texture: every tile is pointed at a stand-in before it goes.
    const freed = view ? null : cellTexture;
    if (!view) cellTexture = null;
    if (view && (view.rows !== before?.rows || !cellTexture)) {
      cellTexture?.destroy();
      cellTexture = dataTexture(
        "hexapodCellGrid",
        CELL_TEXTURE_WIDTH,
        view.rows.height,
        pc.PIXELFORMAT_RG8,
        view.rows.data,
      );
    }
    cellFillOf(view, cellFill);
    for (const entity of cells.meshes()) {
      applyModifier(entity);
      if (freed) entity.gsplat?.setParameter("uCellGrid", stand().state);
    }
    freed?.destroy();
    for (const entity of layers.keys()) entity.gsplat?.setParameter("uCellFill", cellFill);
    hooks.frameWanted();
  };

  /**
   * A layer's modifier: with the skip bit once the work buffer carries the sunlight's word, and
   * the rule's weight while a rule is drawn.
   */
  const layerModifier = (): WorkBufferModifier =>
    rule
      ? (LAYER_RULE_MODIFIERS[sunStream ? 1 : 0] ?? PLAYCANVAS_LAYER_MODIFIER)
      : sunStream
        ? PLAYCANVAS_LAYER_SUN_MODIFIER
        : PLAYCANVAS_LAYER_MODIFIER;

  /**
   * Draws `view` -- a per-cell view rule (scanRule.ts) -- on every tile, or none with null:
   * each tile's modifier gains or loses the rule's weight, and its source is read again. Data
   * age under the rule is the rule's own view (each cell by its owner's age), drawn in place of
   * the scan's own while the rule is on. Each tile is copied again once, in the frame this asks
   * for.
   */
  const setRule = (next: ScanRuleView | null): void => {
    if (!cells) return;
    const before = rule;
    rule = next;
    if (next && next.owners !== before?.owners) {
      const { data, height } = next.owners;
      if (ruleOwners?.height === height) {
        (ruleOwners.lock() as Uint8Array).set(data);
        ruleOwners.unlock();
      } else {
        ruleOwners?.destroy();
        ruleOwners = dataTexture(
          "hexapodRuleOwners",
          RULE_OWNER_WIDTH,
          height,
          pc.PIXELFORMAT_R8,
          data,
        );
      }
    }
    if (next) {
      rulePlane.set(next.plane);
      // The seam's reach in what PlayCanvas draws: its footprint is wider than CesiumJS's.
      ruleParams.set([next.enabled ? 1 : 0, next.cellM, next.reach * PLAYCANVAS_FOOTPRINT, 0]);
    }
    setCellView(next?.age ?? ownCellView);
    for (const entity of cells.meshes()) applyModifier(entity);
    for (const [entity, state] of layers) {
      entity.gsplat?.setWorkBufferModifier(layerModifier());
      layerParameters(entity, state);
    }
    syncRuleDates();
    hooks.frameWanted();
  };

  /**
   * Streams the dates the rule names beside the scan (ruleDates.ts), its code fetched with the
   * first rule that names one; drops them all with the rule.
   */
  const syncRuleDates = (): void => {
    const wanted = rule?.dates ?? [];
    if (ruleDates || wanted.length === 0) {
      ruleDates?.setRule(rule ?? NO_DATES);
      return;
    }
    const kit = lentRuleKit();
    if (ruleDatesLoading || !kit) return;
    ruleDatesLoading = true;
    void import("./ruleDates")
      .then(({ RuleDates }) => {
        if (destroying) return;
        ruleDates = new RuleDates<pc.Entity>(
          {
            loadDate,
            loadLayer: (url, tile, signal) =>
              backend.loadLayer?.(url, tile, signal) ?? Promise.reject(new Error("no layers")),
            setLayerLook: (entity, look) => backend.setLayerLook?.(entity, look),
            add: (entity) => backend.add(entity),
            remove: (entity) => backend.remove(entity),
            dispose: (entity) => backend.dispose(entity),
            place: (entity, matrix) => backend.place?.(entity, matrix),
          },
          {
            arrived: () => hooks.frameWanted(),
            changed: () => hooks.frameWanted(),
            // The budget the host draws to now, which every date shares: frames slowed by the
            // dates lower it for all of them.
            budget: () => drawBudget,
          },
          kit,
        );
        ruleDates.setRule(rule ?? NO_DATES);
        hooks.frameWanted();
      })
      .catch((error: unknown) => {
        // The code did not arrive: the other dates' cells draw nothing, as the rule fails closed.
        log.warn("the rule's other dates did not load; their cells draw nothing", {
          error: error instanceof Error ? error.message : String(error),
        });
      })
      .finally(() => {
        ruleDatesLoading = false;
      });
  };

  /**
   * The sun to light the scan by (state/sunReport.ts), or none. The first one adds the word to the
   * work buffer and has every tile's copy write it -- each tile copied again once, in the frame
   * this asks for; the hook and its uniforms follow in `render` (`syncSun`).
   */
  const setSun = (next: ScanSun | null): void => {
    sun = next;
    if (next && !sunStream && cells) {
      app.scene.gsplat.format.addExtraStreams([{ name: SUN_STREAM, format: pc.PIXELFORMAT_R32U }]);
      sunStream = true;
      for (const entity of cells.meshes()) applyModifier(entity);
      for (const entity of layers.keys()) {
        entity.gsplat?.setWorkBufferModifier(layerModifier());
        // Setting a parameter has the tile copied into the work buffer again.
        entity.gsplat?.setParameter("uCellFill", cellFill);
      }
    }
    hooks.frameWanted();
  };

  /**
   * Puts the frame's hook into the user material while there is a sun (and the word it reads),
   * takes it out when there is none, and sets the sun's uniforms when they change: PlayCanvas
   * copies the material onto its draw material when its version moves (`update`), and a uniform
   * copies nothing into the work buffer.
   */
  const syncSun = (): void => {
    const material = app.scene.gsplat.material;
    const want = sun !== null && sunStream;
    let changed = false;
    if (want !== sunHooked) {
      const chunks = material.getShaderChunks(pc.SHADERLANGUAGE_GLSL);
      if (want) {
        ownModifyVS = chunks.get("gsplatModifyVS") as string | undefined;
        chunks.set("gsplatModifyVS", SUN_HOOK_VS);
      } else if (ownModifyVS !== undefined) {
        chunks.set("gsplatModifyVS", ownModifyVS);
      } else {
        chunks.delete("gsplatModifyVS");
      }
      sunHooked = want;
      sunSet = null;
      changed = true;
    }
    if (want && sun && sun !== sunSet) {
      sunValue.set([sun.dir[0], sun.dir[1], sun.dir[2], sun.sunUp]);
      sunlightValue.set([sun.strength, sun.ambient, sun.flat, 0]);
      material.setParameter("uHexapodSun", sunValue);
      material.setParameter("uHexapodSunlight", sunlightValue);
      sunSet = sun;
      changed = true;
    }
    if (changed) material.update();
    if (sunHooked) sunFrames += 1;
  };

  /**
   * Where `entity`'s own origin is once placed (`matrix`'s translation; none where decoded): a
   * split object moved to a pose is weighed whole by the cell its anchor is in now, so it is
   * copied again.
   */
  const placedAnchor = (entity: pc.Entity, matrix: readonly number[] | null): void => {
    if (matrix) anchors.set(entity, objectAnchor(matrix));
    else anchors.delete(entity);
    const url = fromUrl.get(entity);
    if (rule && url !== undefined && rule.sources.get(url)?.whole)
      entity.gsplat?.setParameter("uRuleSource", ruleSourceOf(entity));
  };

  /** A tile's own splats in the scan's frame, for the cell view's reading (`CellTiles`). */
  const cellSample = (entity: pc.Entity): CellTileSample | null => {
    const tile = tiles.get(entity);
    const { x, y, z } = tile?.columns ?? {};
    if (!tile || !x || !y || !z) return null;
    const [ox, oy, oz] = tile.origin;
    return {
      count: tile.order.length,
      at: (i, out) => {
        out[0] = (x[i] ?? 0) + ox;
        out[1] = (y[i] ?? 0) + oy;
        out[2] = (z[i] ?? 0) + oz;
      },
    };
  };

  /**
   * A decoded tile of the tileset at `url` made into PlayCanvas's resource and entity:
   * main-thread work. `kind` says what it is: one of the scan's own tiles (bound to its objects
   * and picks), an inferred layer's (scanLayers.ts), or another date's under a per-cell view
   * rule (ruleDates.ts: drawn as a scan tile -- the cell views and the sun -- but bound to none
   * of this scan's objects, motion or picks, as CesiumJS's companions are not).
   */
  const build = (
    url: string,
    tile: TileNode,
    decoded: Decoded,
    kind: "scan" | "layer" | "date" = "scan",
  ): pc.Entity => {
    const layer = kind === "layer";
    const { properties, count, order } = decoded;
    if (!properties || count === undefined || !order) {
      throw new Error(decoded.error ?? "The tile could not be decoded.");
    }
    const data = new pc.GSplatData([
      {
        name: "vertex",
        count,
        properties: Object.entries(properties).map(([name, storage]) => ({
          type: "float",
          name,
          storage,
          byteSize: 4,
        })),
      },
    ]);
    // glTF KHR_gaussian_splatting's convention: linear scale, opacity after the sigmoid.
    data.activated = true;
    // Already in Morton order (the worker's `mortonOrder`), as `reorderData` would leave it.
    const resource = new pc.GSplatResource(device, data) as Resource;
    const entity = new pc.Entity(tile.uri);
    const origin = decoded.origin ?? [0, 0, 0];
    entity.setLocalPosition(...origin);
    origins.set(entity, origin);
    fromUrl.set(entity, url);
    entity.addComponent("gsplat", { resource });
    resources.set(entity, resource);
    // Every tile of the scan reads its cell, whether or not it carries ids.
    if (!layer) cells?.add(entity, origin);
    // A layer's tile carries none of the scan's ids, skins or picks: its look is its own. Nor
    // does another date's.
    if (decoded.checksum !== undefined && kind === "scan") {
      const binding: TileBinding = {
        checksum: decoded.checksum,
        order,
        resource,
        columns: properties,
        origin,
        doc: null,
        matched: false,
        ids: null,
        idSet: new Set(),
        skinDoc: null,
        skinned: false,
        skinWide: false,
        skinSet: new Set(),
        modifier: null,
        restCenters: null,
        pick: undefined,
        placement: null,
        placedPick: null,
      };
      tiles.set(entity, binding);
      if (style || motion) {
        bind(entity, binding, true);
        if (binding.ids && motion && motion.rigidByLeaf.size > 0) sortDirty.add(entity);
      }
    }
    if (!layer && (cells?.drawing || sunStream || rule)) applyModifier(entity);
    return entity;
  };

  /** A tile of another date under a rule (ruleDates.ts): a scan tile bound to none of this one. */
  const loadDate = async (
    tilesetUrl: string,
    tile: TileNode,
    signal?: AbortSignal,
  ): Promise<pc.Entity> => {
    const decoded = await decode(new URL(tile.uri, tilesetUrl).toString());
    signal?.throwIfAborted();
    return hooks.work.run(() => {
      signal?.throwIfAborted();
      return build(tilesetUrl, tile, decoded, "date");
    });
  };

  const backend: ScanBackend<pc.Entity> = {
    name: identity.name,
    api,
    apiNote: identity.note,
    loadFactor: 1,
    load: async (tilesetUrl: string, tile: TileNode, signal?: AbortSignal) => {
      const decoded = await decode(new URL(tile.uri, tilesetUrl).toString(), tile.offset);
      signal?.throwIfAborted();
      // Building the resource packs every splat into textures on the main thread: within the
      // frame's budget, a little at a time while the camera moves.
      const entity = await hooks.work.run(() => {
        signal?.throwIfAborted();
        return build(tilesetUrl, tile, decoded);
      });
      return entity;
    },
    loadLayer: async (tilesetUrl: string, tile: TileNode, signal?: AbortSignal) => {
      const decoded = await decode(new URL(tile.uri, tilesetUrl).toString(), tile.offset);
      signal?.throwIfAborted();
      return hooks.work.run(() => {
        signal?.throwIfAborted();
        return build(tilesetUrl, tile, decoded, "layer");
      });
    },
    setLayerLook: (entity, look) => {
      const component = entity.gsplat;
      if (!component) return;
      const before = layers.get(entity);
      const origin = origins.get(entity) ?? [0, 0, 0];
      const state = before ?? {
        look,
        eye: new Float32Array(4),
        lastEye: null,
        centre: [...origin],
      };
      const previous = before?.look.cones ?? null;
      state.look = look;
      layers.set(entity, state);
      if (!before) component.setWorkBufferModifier(layerModifier());
      layerParameters(entity, state);
      if (previous !== look.cones) releaseCones(previous);
    },
    add: (entity) => {
      app.root.addChild(entity);
      shown.add(entity);
      addedAt.set(entity, framesDrawn);
      unconfirmed.add(entity);
    },
    // Drawn once a rendered frame after the one that took it in showed every change sorted;
    // without `frame:ready`, after a few rendered frames. The same on WebGPU, where the frame
    // that takes a tile in can report ready before PlayCanvas's update has made the tile part
    // of what it draws (its streaming update runs in its own loop, after the frame): one more
    // ready frame is the proof, and `render` asks for it.
    isDrawn: (entity, sinceMs) => {
      const added = addedAt.get(entity) ?? framesDrawn;
      if (frameReadySeen) return lastReadyFrame >= added + 2;
      return framesDrawn - added >= SETTLE_FRAMES && sinceMs >= SETTLE_MS;
    },
    remove: (entity) => {
      unconfirmed.delete(entity);
      shown.delete(entity);
      if (entity.parent) entity.parent.removeChild(entity);
    },
    dispose: (entity) => {
      unconfirmed.delete(entity);
      shown.delete(entity);
      tiles.delete(entity);
      cells?.delete(entity);
      sortDirty.delete(entity);
      const layer = layers.get(entity);
      if (layer) {
        layers.delete(entity);
        releaseCones(layer.look.cones);
      }
      const resource = resources.get(entity);
      entity.destroy();
      if (resource) {
        doomed.push({ resource, at: framesDrawn });
        // The renders that let it go (DESTROY_AFTER_FRAMES): asked for, since a dispose at
        // rest has none coming. Within a frame the host takes it as one more frame.
        hooks.frameWanted();
      }
    },
    place: (entity, matrix) => {
      const origin = origins.get(entity) ?? [0, 0, 0];
      const tile = tiles.get(entity);
      if (matrix === null) {
        if (tile) {
          tile.placement = null;
          tile.placedPick = null;
        }
        entity.setLocalPosition(...origin);
        entity.setLocalRotation(pc.Quat.IDENTITY);
        placedAnchor(entity, null);
        return;
      }
      placing ??= { matrix: new pc.Mat4(), position: new pc.Vec3(), rotation: new pc.Quat() };
      const { matrix: placement, position, rotation } = placing;
      placement.set(Array.from(matrix));
      // Only the matrix is kept: a pick moves the tile's splats there when it asks.
      if (tile) {
        (tile.placement ??= new pc.Mat4()).copy(placement);
        tile.placedPick = null;
      }
      placement.transformPoint(position.set(...origin), position);
      rotation.setFromMat4(placement);
      entity.setLocalPosition(position);
      entity.setLocalRotation(rotation);
      placedAnchor(entity, matrix);
    },
    render: (pose: ScanPose) => {
      // A lost device draws nothing; the host is already replacing this renderer.
      if (lost) return;
      if (
        pose.width !== size.width ||
        pose.height !== size.height ||
        pose.pixelRatio !== size.pixelRatio
      ) {
        size = { width: pose.width, height: pose.height, pixelRatio: pose.pixelRatio };
        device.maxPixelRatio = pose.pixelRatio;
        app.resizeCanvas(pose.width, pose.height);
      }
      // Half a CSS pixel up close, nothing from afar (quality.ts): a uniform, set as it changes.
      const minPixelSize = splatMinPixelSize(pose.pixelRatio, pose.farView === true);
      if (app.scene.gsplat.minPixelSize !== minPixelSize) {
        app.scene.gsplat.minPixelSize = minPixelSize;
      }
      const component = camera.camera;
      if (component) {
        component.fov = (pose.fovy * 180) / Math.PI;
        component.nearClip = pose.near;
        component.farClip = pose.far;
        component.aspectRatioMode = pc.ASPECT_AUTO;
      }
      camera.setPosition(...pose.eye);
      target.set(
        pose.eye[0] + pose.direction[0],
        pose.eye[1] + pose.direction[1],
        pose.eye[2] + pose.direction[2],
      );
      camera.lookAt(target, up.set(...pose.up));
      refreshSort(performance.now());
      // A layer's fade follows the eye: its tiles are copied into the work buffer again, in
      // this frame, once the eye has moved enough to show (layerLook.ts `eyeMoved`).
      lastPoseEye = pose.eye;
      for (const [entity, layer] of layers) {
        if (!layer.look.cones) continue;
        const eye = layerEye(layer.look, pose.eye);
        if (!eyeMoved(layer.lastEye, eye, layer.centre)) continue;
        layer.lastEye = eye;
        entity.gsplat?.setParameter("uLayerConeEye", coneEye(layer.look, pose.eye, layer.eye));
      }
      // A tile whose cell level stepped is copied again in this frame (cellModifier.ts).
      cells?.update(pose, (entity, cell) => {
        entity.gsplat?.setParameter("uCellLevel", cell.level);
        entity.gsplat?.setParameter("uCellTile", cell.tile);
      });
      // The rule's other dates, planned from the pose this frame is drawn from.
      ruleDates?.plan(pose);
      syncSun();
      framesDrawn += 1;
      app.render();
      while (doomed[0] && framesDrawn - doomed[0].at >= DESTROY_AFTER_FRAMES) {
        doomed.shift()?.resource.destroy();
      }
      // A resource still waiting for its renders gets them: at most DESTROY_AFTER_FRAMES
      // frames after the last dispose, and then the loop can pause (`frameupdate`).
      if (doomed.length > 0) hooks.frameWanted();
      // A frame drawn may have started a sort or a load: the loop watches for it.
      resumeLoop();
      // Sorted on the GPU, this frame was ready and a tile added lately still needs a ready
      // frame two on to count as drawn: nothing else will ask for that frame (no sort result
      // comes back), so this one does -- one frame per batch of tiles, none at rest.
      for (const entity of unconfirmed) {
        if (lastReadyFrame >= (addedAt.get(entity) ?? 0) + 2) unconfirmed.delete(entity);
      }
      if (sortsOnGpu && lastReadyFrame === framesDrawn && unconfirmed.size > 0) {
        hooks.frameWanted();
      }
    },
    // A moving object's re-sort held back by SORT_REFRESH_MS: due then, whether or not
    // anything else changes by that time.
    frameDueBy: () => {
      if (lost) return null;
      const dates = ruleDates?.dueBy() ?? null;
      const sort = sortDirty.size > 0 ? lastSortAt + SORT_REFRESH_MS : null;
      return sort === null ? dates : dates === null ? sort : Math.min(sort, dates);
    },
    setBudget: (drawn) => {
      drawBudget = drawn;
      app.scene.gsplat.splatBudget = drawn;
    },
    setInstances,
    pickTiles: () => {
      const out: PickTile[] = [];
      for (const entity of shown) {
        const tile = tiles.get(entity);
        if (!tile) continue;
        tile.pick ??= pickTileOf(tile.columns, tile.order, tile.origin, tile.checksum);
        if (!tile.pick) continue;
        if (tile.placement) {
          tile.placedPick ??= placedPickTile(tile.pick, tile.placement);
          out.push(tile.placedPick);
        } else {
          out.push(tile.pick);
        }
      }
      return out;
    },
    instanceTiles: () => {
      let matched = 0;
      for (const tile of tiles.values()) if (tile.matched) matched += 1;
      return { tiles: tiles.size, matched };
    },
    ...(moves
      ? {
          setMotion,
          motionTiles: () => {
            let skinned = 0;
            for (const tile of tiles.values()) {
              if (tile.skinned && tile.skinSet.size > 0) skinned += 1;
            }
            return { skinned, redrawn };
          },
        }
      : {}),
    // PlayCanvas's own streamed level of detail -- what superspl.at runs: chunks of a few
    // hundred thousand splats per level as lossless WebP textures the browser decodes off the
    // main thread and uploads as they are, chosen against `splatBudget` from the camera.
    streamNative: async (url) => {
      const asset = new pc.Asset(url, "gsplat", { url });
      app.assets.add(asset);
      await new Promise<void>((resolve, reject) => {
        asset.once("load", () => resolve());
        asset.once("error", (error: unknown) => reject(new Error(String(error))));
        app.assets.load(asset);
      });
      const entity = new pc.Entity("scan");
      entity.addComponent("gsplat", { asset, unified: true });
      app.root.addChild(entity);
      // Streamed natively the scan is one entity: one cell level for all of it, from the
      // middle of its bounds (cellModifier.ts).
      const middle = (asset.resource as { aabb?: { center?: pc.Vec3 } } | null)?.aabb?.center;
      cells?.add(entity, middle ? [middle.x, middle.y, middle.z] : [0, 0, 0]);
      natives.add(entity);
      if (cells?.drawing || sunStream || rule) applyModifier(entity);
      resumeLoop();
      return {
        splats: () => app.scene.gsplat.splatBudget,
        stop: () => {
          cells?.delete(entity);
          entity.destroy();
          app.assets.remove(asset);
          asset.unload();
        },
      };
    },
    destroy: () => {
      for (const off of cellLinks) off();
      ruleDates?.stop();
      ruleDates = null;
      cellTexture?.destroy();
      cellTexture = null;
      ruleOwners?.destroy();
      ruleOwners = null;
      for (const worker of workers) worker.terminate();
      waiting.clear();
      tiles.clear();
      shown.clear();
      sortDirty.clear();
      layers.clear();
      for (const texture of coneTextures.values()) texture.destroy();
      coneTextures.clear();
      for (const texture of [
        stateTexture,
        handlesTexture,
        slotsTexture,
        posesTexture,
        empty?.state,
        empty?.float,
        empty?.uint,
      ]) {
        texture?.destroy();
      }
      stateTexture = null;
      handlesTexture = null;
      slotsTexture = null;
      posesTexture = null;
      empty = null;
      // PlayCanvas only lets go of its context (`gl = null`), and the browser frees a
      // context's memory whenever it collects the canvas -- with a new canvas and context per
      // session, a few sessions in a visit held several. Losing it frees it now, once
      // PlayCanvas has taken its own handlers off the canvas. A WebGPU device is destroyed by
      // PlayCanvas itself (`GPUDevice.destroy`), which frees its memory at once.
      destroying = true;
      const gl = (device as unknown as { gl?: WebGLRenderingContext | null }).gl;
      try {
        app.destroy();
      } catch (error) {
        // A lost WebGPU device can refuse what PlayCanvas does on the way out; nothing of it is
        // drawn again either way.
        if (!lost) throw error;
      }
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
  // A resource waiting to be destroyed keeps the loop going until the renders it asked for
  // (`render`, `dispose`) have let it go: a few frames, never for good.
  app.on("frameupdate", () => {
    countOverlayLoopTick();
    quietTicks += 1;
    if (quietTicks >= QUIET_TICKS_BEFORE_PAUSE && settled && doomed.length === 0) looping = false;
  });
  const gsplat = app.systems.gsplat as unknown as pc.EventHandler | undefined;
  // New streamed detail, or a sort result waiting to be applied: a frame shows it.
  gsplat?.on("frame:request", () => {
    quietTicks = 0;
    hooks.frameWanted();
  });
  gsplat?.on("frame:ready", ((_camera, _layer, ready, loadingCount) => {
    frameReadySeen = true;
    settled = ready && !loadingCount;
    if (ready) lastReadyFrame = framesDrawn;
  }) as FrameReady);
  // A sort that finished while the loop was paused: the loop hands it over (`frame:request`).
  // (Sorted on the CPU only: a GPU sort finishes in the frame that draws.)
  app.scene.on("gsplat:sorted", resumeLoop);
  // The scan's Confidence / Data age view, drawn as the view says (scanCellView.ts).
  const assetId = hooks.assetId;
  const cellLinks =
    cells && assetId !== undefined
      ? [
          watchScanCellView(assetId, (view) => {
            ownCellView = view;
            // Data age under a rule is the rule's own: the scan's comes back after it.
            setCellView(rule?.age ?? view);
          }),
          reportScanCellView(assetId, () => cells.reading(identity.name, cellSample)),
          // A per-cell view rule on the site's dates, drawn in this one sort (scanRule.ts).
          watchScanRule(assetId, setRule),
          reportScanRule(assetId, () => {
            const sources: Record<number, number> = {};
            if (rule)
              for (const entity of [...cells.meshes(), ...layers.keys()]) {
                const source = ruleSourceOf(entity)[0] ?? NO_OWNER;
                sources[source] = (sources[source] ?? 0) + 1;
              }
            return {
              renderer: identity.name,
              on: rule !== null,
              sources,
              dates: ruleDates?.status() ?? { dates: 0, tiles: 0, gaussians: 0, loading: 0 },
            };
          }),
          // The scan's sunlight, drawn by the frame's hook (sunHook.ts). Watching is what says
          // this renderer draws it (the badge's report), so only where it can: GLSL, as the
          // cell views.
          watchScanSun(assetId, setSun),
          reportScanSun(assetId, () => ({
            renderer: identity.name,
            installed: sunHooked,
            stream: sunStream,
            sun: sunHooked ? sunSet : null,
            frames: sunFrames,
          })),
        ]
      : [];
  return backend;
}
