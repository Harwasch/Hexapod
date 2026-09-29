/**
 * Moves a Gaussian splat tileset's gaussians by a motion rig, so a measured tree can move
 * without its measurement ever changing — one tile or a whole level-of-detail hierarchy.
 *
 * The shape of the thing: `apply(nodeTransforms)` takes the rig transforms some caller computed
 * and pushes one frame to the GPU. It owns no clock, no wind, no scene listener and no UI —
 * `LivingSurveyManager` drives it from `scene.preUpdate`. Everything arithmetical lives in
 * the pure modules beside it (`splatTexels.ts`, `splatFrames.ts`, `splatTiles.ts`,
 * `splatGpuMotion.ts`), which is why almost all of this is testable without a GPU.
 *
 * Two ways to put a frame on screen, chosen per snapshot (the reason for a CPU choice is in
 * `status.cpuReason`, and `setGpu` switches a live deformer between them):
 *
 * - **CPU** (without a `gpu` factory, and the fallback): recompute every displaced splat from
 *   the canonical copy and re-upload the attribute-texture rows they occupy. Linear in the
 *   splat count, and it needs the packed buffer the interception captured (`splatCapture.ts`).
 * - **GPU** (with a `gpu` texture factory, on the patched engine; the app's default): upload
 *   per-node transforms only and let the vertex shader apply them (`splatGpuMotion.ts`). Needs
 *   every selected tile to share one bake matrix, which every `splat_tiles.py` tileset does; a
 *   snapshot that does not falls back to the CPU path.
 *
 * Four rules it exists to enforce:
 *
 * **Canonical positions are never mutated.** At attach the baked positions are copied once into
 * buffers this class owns, and every frame recomputes `displaced = f(canonical, transforms)`
 * from that pristine copy. There is no read-modify-write anywhere, so error cannot accumulate
 * and zero wind returns the exact measured bytes. `tile.content.positions`,
 * `primitive._positions`, `snapshot.positions` and the glTF POSITION array are read-only to us
 * — `transformTile` rewrites the first of those in place, so a write there would make a
 * deformation permanent and compound it across every rebuild.
 *
 * **A gaussian's motion is a function of where it stands, never of its tile.** A multi-tile
 * snapshot is the selected tiles' splats concatenated, and which tiles are selected changes
 * with the camera. So identity, rig binding and flutter identity are all derived per tile from
 * that tile's own positions (`splatTiles.ts`) and concatenated in snapshot order; a merged
 * parent binds to the four nodes nearest its own centre exactly as a leaf gaussian does.
 *
 * **Refuse rather than mislead.** Before its first write it checks that the root frame really
 * is east-north-up, that the tree stands along local +Z, and — bit-exactly, per tile — that
 * these are the splats the rig was built for. On any failure it writes nothing, ever, and says
 * why. A tree that does not move is honest; a tree whose wrong splats move is not.
 *
 * **What is not a plant never moves.** A forest rig (`rig.plants`: a whole capture's plants in
 * one rig) binds every gaussian its plant binding marks static to a static anchor node, and this
 * class pins every such anchor at rest, frame after frame, whatever transform or flutter the
 * motion model hands it: on the CPU path a pinned splat gets the engine's own bytes back, on
 * the GPU path its node rows are exact zeros and the shader returns the fetched position. A
 * plant's gaussians are skinned to its own joints only (`skinSplatsToPlants`), so a crown
 * cannot drag the wall beside it. A forest rig without its binding is refused.
 *
 * **Rebuilds re-derive, they do not re-apply.** A snapshot rebuilds when the camera changes the
 * tile selection, and when `SiteManager.clampToGround` sets `tileset.modelMatrix` after an
 * asynchronous terrain sample. Each rebuild derives a new attachment from the new snapshot;
 * tiles that stayed selected under the same bake matrix reuse their binding.
 */

import {
  deformPositions,
  FLUTTER_STILL,
  flutterHash,
  IDENTITY_TRANSFORM,
  isForestRig,
  rigTileChecksums,
  SKIN_INFLUENCES,
  skinSlice,
  staticAnchors,
  type FlutterField,
  type MotionRig,
  type NodeTransform,
  type PlantBinding,
  type SplatSkin,
} from "@twin/world";

import { createLogger } from "@/lib/log";

import { digestSplatPositions, findSplatCapture } from "./splatCaptureRegistry";
import {
  GEODETIC_ALIGNMENT_MIN,
  geodeticFrameAlignment,
  resolveBakedPositions,
  treeUprightness,
  type TreeUprightness,
} from "./splatFrames";
import { evaluateSplatMotion, SplatGpuMotion, type MotionTextureFactory } from "./splatGpuMotion";
import { hasVertexMotionHook, type SplatPrimitive, type SplatTilesetLike } from "./splatInternals";
import {
  createStagingBuffer,
  fullRowRange,
  rowRangeForMovingNodes,
  rowRangeWordCount,
  rowSlice,
  splatTextureLayout,
  writeSplatPositions,
  type SplatRowRange,
  type SplatStagingBuffer,
  type SplatTextureLayout,
} from "./splatTexels";
import {
  aggregateBindings,
  bindTile,
  snapshotTiles,
  TileBindingCache,
  type SnapshotBinding,
  type TileBinding,
} from "./splatTiles";

const log = createLogger("splat-deformer");

/** What the deformer is doing. Only `ready` ever writes to the GPU. */
export type DeformerPhase =
  /** Nothing to do yet — the primitive, its texture or its capture has not appeared. */
  | "waiting"
  /** Attached and validated; `apply` writes. */
  | "ready"
  /** Permanently declined for this tileset. Never writes, never retries. */
  | "refused"
  /** `destroy()` has been called. */
  | "detached";

/** Why the deformer is waiting, or why it refused. */
export type DeformerReason =
  /** The tileset has no splat primitive yet. */
  | "no-primitive"
  /** No snapshot, positions or addressing parameters yet. */
  | "no-snapshot"
  /** The attribute texture does not exist for several frames after tile load. */
  | "no-texture"
  /** A selected tile has not been baked yet, so its local frame is unknown. */
  | "no-bake-transform"
  /** The interception did not see this snapshot's packed buffer (CPU path only). */
  | "no-capture"
  /** The snapshot's tile list cannot be read yet: a rebuild is in flight, or it does not add up. */
  | "tiles"
  /** The addressing parameters the primitive reports are not self-consistent. */
  | "layout"
  /** The root transform's Z column is not the geodetic normal. */
  | "frame"
  /** The tree's own points say local +Z is not up. */
  | "upright"
  /** A tile's splats are not ones the rig was built for. */
  | "checksum"
  /** A bake matrix is singular, or re-baking does not reproduce the engine's positions. */
  | "bake"
  /** A forest rig came without the plant binding that keeps what is not a plant still. */
  | "binding"
  /** A bug in this class. Reported rather than thrown, because it runs inside `preUpdate`. */
  | "internal";

/** How a frame reaches the screen. */
export type DeformerMotion = "cpu" | "gpu";

/**
 * Why an attachment is on the CPU path.
 *
 * - `no-factory`: the deformer was given no GPU texture factory — the caller chose the CPU path
 *   (a setting, a build flag) or this CesiumJS build does not export what the factory needs.
 * - `no-hook`: the primitive does not carry the engine patch's `vertexMotion` hook.
 * - `mixed-bake`: the snapshot's tiles do not share one bake matrix, which the shader needs.
 */
export type DeformerCpuReason = "no-factory" | "no-hook" | "mixed-bake";

/** Everything the manager and the HUD need to know, and nothing they have to guess. */
export interface DeformerStatus {
  readonly phase: DeformerPhase;
  readonly reason?: DeformerReason;
  readonly numSplats: number;
  /** Tiles the current snapshot aggregates. */
  readonly tiles: number;
  /** Tiles bound since construction: a tile that stays selected is bound once. */
  readonly tileBindings: number;
  /** Which path the current attachment writes through. */
  readonly motion: DeformerMotion;
  /** Why `motion` is `cpu`; absent on the GPU path. See {@link DeformerCpuReason}. */
  readonly cpuReason?: DeformerCpuReason;
  /** `primitive._snapshot.generation` the current attachment was derived from. */
  readonly generation: number;
  /** Times the snapshot rebuilt and the attachment was re-derived from a new base. */
  readonly rederivations: number;
  /** Frames that reached a texture upload. */
  readonly uploads: number;
  /** Rows and words uploaded by the most recent write. */
  readonly lastUploadRows: number;
  readonly lastUploadWords: number;
  /** Milliseconds the most recent attachment took to derive (binding included). */
  readonly lastDeriveMs: number;
  /** True while the screen shows displaced positions rather than the measured ones. */
  readonly displaced: boolean;
  /** Dot of the root transform's Z column with the geodetic normal; 1 is a perfect ENU frame. */
  readonly geodeticAlignment: number;
  /** Largest disagreement between our re-bake and the engine's baked positions, metres. */
  readonly bakeResidualM: number;
  /**
   * The digest observed over the un-baked positions: the single tile's, or — on a refusal —
   * the tile that failed. For diagnosing a refusal.
   */
  readonly observedChecksum?: string;
  readonly uprightness?: TreeUprightness;
  /** A forest rig's plants; 0 for a single-plant rig. */
  readonly plants: number;
  /** Splats of the current snapshot bound to a pinned static anchor. */
  readonly staticSplats: number;
  /** Tiles bound by the most recent derivation (tiles reused from the cache excluded). */
  readonly lastBoundTiles: number;
  /** Gaussians in those tiles. */
  readonly lastBoundSplats: number;
  /** Milliseconds those tiles took to prove and bind, all together. */
  readonly lastBindMs: number;
}

/**
 * Whether the static splats of the current frame are exactly their canonical bytes, as the
 * deformer last drew them: for tests and the harness. Nothing here is on a per-frame path.
 */
export interface StaticAudit {
  readonly motion: DeformerMotion;
  readonly staticSplats: number;
  /** Static splats drawn anywhere but their canonical bytes. Zero, or something is wrong. */
  readonly staticMoved: number;
  /** Static splats whose skin weights a node that is not pinned. Zero by construction. */
  readonly staticUnpinned: number;
  /** Plant splats drawn somewhere else than canonical: that they move at all. */
  readonly plantMoved: number;
  readonly plantSplats: number;
}

/**
 * A re-bake that disagrees with the engine by more than this is not a rounding difference —
 * the bake matrix is wrong, and deforming would move the tree somewhere it was never measured.
 */
export const BAKE_RESIDUAL_LIMIT_M = 1e-3;

/**
 * Fewest splats the upright test is decided on. A level-of-detail view from afar can be a
 * merged root of a hundred gaussians, too few for a crown/base spread ratio to mean anything;
 * identity is already proven per tile by then, so the test waits for a view that can answer it.
 */
export const UPRIGHT_MIN_SPLATS = 1000;

export interface SplatDeformerOptions {
  /** A `Cesium3DTileset` viewed through `splatTilesetOf`, or a test double. */
  readonly tileset: SplatTilesetLike;
  readonly rig: MotionRig;
  /**
   * Enables the GPU path where the engine carries the motion hook. `undefined` keeps every
   * frame on the CPU path.
   */
  readonly gpu?: MotionTextureFactory;
  /** A forest rig's plant binding (`plants.json`). Required when the rig lists plants. */
  readonly plantBinding?: PlantBinding;
}

/** The CPU path's per-snapshot buffers. */
interface CpuBuffers {
  /** Immutable. The engine's baked bytes, the exact rest pose. */
  readonly canonicalBaked: Float32Array;
  readonly staging: SplatStagingBuffer;
  /** Own `ArrayBuffer`s: `deformPositions` guards against `out === positions` but not aliasing. */
  readonly displacedLocal: Float32Array;
  readonly displacedBaked: Float32Array;
}

/** Everything derived from one snapshot generation. Replaced wholesale when the snapshot rebuilds. */
interface Attachment {
  readonly generation: number;
  /** The engine's own array, kept only to notice when it is swapped. Never written. */
  readonly enginePositions: Float32Array;
  readonly numSplats: number;
  /** Splats bound to a pinned static anchor. */
  readonly staticSplats: number;
  readonly layout: SplatTextureLayout;
  readonly binding: SnapshotBinding;
  readonly motion: DeformerMotion;
  readonly cpuReason: DeformerCpuReason | undefined;
  readonly cpu: CpuBuffers | undefined;
  readonly nodeMoves: Uint8Array;
  readonly geodeticAlignment: number;
}

export class SplatDeformer {
  readonly #tileset: SplatTilesetLike;
  readonly #rig: MotionRig;
  readonly #accepted: ReadonlySet<string>;
  readonly #plantBinding: PlantBinding | undefined;
  /** Per node, 1 for a static anchor, which is held at rest whatever the model says. */
  readonly #pinned: Uint8Array;
  readonly #forest: boolean;
  #lastBoundTiles = 0;
  #lastBoundSplats = 0;
  #lastBindMs = 0;
  readonly #cache = new TileBindingCache();
  #gpuFactory: MotionTextureFactory | undefined;
  #gpu: SplatGpuMotion | undefined;
  #attachment: Attachment | undefined;
  #phase: DeformerPhase = "waiting";
  #reason: DeformerReason | undefined = "no-primitive";
  #observedChecksum: string | undefined;
  #uprightness: TreeUprightness | undefined;
  #displaced = false;
  #lastRange: SplatRowRange | undefined;
  #uploads = 0;
  #rederivations = 0;
  #lastUploadRows = 0;
  #lastUploadWords = 0;
  #lastDeriveMs = 0;

  constructor(options: SplatDeformerOptions) {
    this.#tileset = options.tileset;
    this.#rig = options.rig;
    this.#accepted = rigTileChecksums(options.rig);
    this.#gpuFactory = options.gpu;
    this.#plantBinding = options.plantBinding;
    this.#forest = isForestRig(options.rig);
    this.#pinned = staticAnchors(options.rig);
  }

  get status(): DeformerStatus {
    const attachment = this.#attachment;
    return {
      phase: this.#phase,
      reason: this.#reason,
      numSplats: attachment?.numSplats ?? 0,
      tiles: attachment?.binding.tiles.length ?? 0,
      tileBindings: this.#cache.bound,
      motion: attachment?.motion ?? (this.#gpuFactory === undefined ? "cpu" : "gpu"),
      cpuReason:
        attachment !== undefined
          ? attachment.cpuReason
          : this.#gpuFactory === undefined
            ? "no-factory"
            : undefined,
      generation: attachment?.generation ?? -1,
      rederivations: this.#rederivations,
      uploads: this.#uploads,
      lastUploadRows: this.#lastUploadRows,
      lastUploadWords: this.#lastUploadWords,
      lastDeriveMs: this.#lastDeriveMs,
      displaced: this.#displaced,
      geodeticAlignment: attachment?.geodeticAlignment ?? Number.NaN,
      bakeResidualM: attachment?.binding.bakeResidualM ?? Number.NaN,
      observedChecksum: this.#observedChecksum,
      uprightness: this.#uprightness,
      plants: this.#rig.plants?.length ?? 0,
      staticSplats: attachment?.staticSplats ?? 0,
      lastBoundTiles: this.#lastBoundTiles,
      lastBoundSplats: this.#lastBoundSplats,
      lastBindMs: this.#lastBindMs,
    };
  }

  /**
   * Checks, splat by splat, that every static splat of the current snapshot was last drawn at
   * its canonical bytes: on the CPU path from the buffer the last write uploaded, on the GPU
   * path by evaluating the shader's transcription over the motion texture it was given.
   */
  staticAudit(): StaticAudit | undefined {
    const attachment = this.#attachment;
    const labels = attachment?.binding.plantLabels;
    if (attachment === undefined || labels === undefined) return undefined;
    const { skin, flutterKeys } = attachment.binding;
    const baked = attachment.enginePositions;
    let staticSplats = 0;
    let staticMoved = 0;
    let staticUnpinned = 0;
    let plantMoved = 0;
    let plantSplats = 0;
    const motionData = this.#gpu?.motionData;
    for (let i = 0; i < attachment.numSplats; i += 1) {
      const isStatic = (labels[i] ?? 0) === 0;
      if (isStatic) {
        staticSplats += 1;
        for (let k = 0; k < SKIN_INFLUENCES; k += 1) {
          const at = i * SKIN_INFLUENCES + k;
          if ((skin.weights[at] ?? 0) > 0 && (this.#pinned[skin.nodes[at] ?? 0] ?? 0) === 0)
            staticUnpinned += 1;
        }
      } else {
        plantSplats += 1;
      }
      let drawn: readonly [number, number, number];
      const rest: [number, number, number] = [
        baked[i * 3] ?? 0,
        baked[i * 3 + 1] ?? 0,
        baked[i * 3 + 2] ?? 0,
      ];
      if (attachment.motion === "cpu") {
        const out = attachment.cpu?.displacedBaked;
        drawn =
          out === undefined || !this.#displaced
            ? rest
            : [out[i * 3] ?? 0, out[i * 3 + 1] ?? 0, out[i * 3 + 2] ?? 0];
      } else {
        drawn =
          motionData === undefined || !this.#displaced
            ? rest
            : evaluateSplatMotion(
                motionData,
                skin.nodes.subarray(i * SKIN_INFLUENCES, (i + 1) * SKIN_INFLUENCES),
                skin.weights.subarray(i * SKIN_INFLUENCES, (i + 1) * SKIN_INFLUENCES),
                flutterHash(flutterKeys[i] ?? 0),
                rest,
              );
      }
      const same =
        Object.is(Math.fround(drawn[0]), rest[0]) &&
        Object.is(Math.fround(drawn[1]), rest[1]) &&
        Object.is(Math.fround(drawn[2]), rest[2]);
      if (isStatic && !same) staticMoved += 1;
      if (!isStatic && !same) plantMoved += 1;
    }
    return {
      motion: attachment.motion,
      staticSplats,
      staticMoved,
      staticUnpinned,
      plantMoved,
      plantSplats,
    };
  }

  /** The canonical local positions of the current snapshot, for metrics and tests. Immutable. */
  get canonicalPositions(): Float32Array | undefined {
    return this.#attachment?.binding.canonicalLocal;
  }

  /** Each splat's nearest rig node in the current snapshot (slot 0 of its skin). Immutable. */
  get assignment(): Uint16Array | undefined {
    return this.#attachment?.binding.assignment;
  }

  /** The nodes and blend weights each splat of the current snapshot follows. Immutable. */
  get skin(): SplatSkin | undefined {
    return this.#attachment?.binding.skin;
  }

  /** Each splat's flutter identity in the current snapshot. Treat as immutable. */
  get flutterKeys(): Uint32Array | undefined {
    return this.#attachment?.binding.flutterKeys;
  }

  /** The current snapshot's tiles and their bindings, for tests and diagnostics. */
  get snapshotBinding(): SnapshotBinding | undefined {
    return this.#attachment?.binding;
  }

  /** The GPU path's hook, when one exists. For tests and the harness. */
  get gpuMotion(): SplatGpuMotion | undefined {
    return this.#gpu;
  }

  /**
   * Pushes one frame.
   *
   * Returns the status rather than throwing: this runs inside `scene.preUpdate`, where an
   * exception takes the whole viewer down, and every failure here has a correct do-nothing
   * answer.
   */
  apply(
    transforms: readonly NodeTransform[],
    flutter: FlutterField = FLUTTER_STILL,
  ): DeformerStatus {
    if (this.#phase === "refused" || this.#phase === "detached") return this.status;
    try {
      const attachment = this.#syncAttachment();
      if (attachment === undefined) {
        // Whatever the GPU hook last held belongs to a snapshot we cannot vouch for.
        this.#gpu?.deactivate();
        return this.status;
      }
      this.#write(attachment, transforms, flutter);
    } catch (error) {
      // A refusal is a decision; an unexpected throw is a bug, and it must not take the scene
      // with it. Both end in "this tileset does not deform".
      this.#refuse("internal", `unexpected failure: ${String(error)}`);
    }
    return this.status;
  }

  /** Releases the buffers, uninstalls the GPU hook and stops writing. */
  destroy(): void {
    this.#attachment = undefined;
    this.#phase = "detached";
    this.#reason = undefined;
    this.#displaced = false;
    this.#lastRange = undefined;
    this.#gpu?.destroy();
    this.#gpu = undefined;
    this.#cache.clear();
  }

  /**
   * Switches the motion path: a GPU texture factory, or `undefined` for the CPU path.
   *
   * The measured pose is put back first, through the path being left, so nothing displaced is
   * left behind on it: on the CPU path that is the ordinary restoring write — every row the
   * last frame moved, rewritten from the canonical bytes, exactly as a drop to calm does it —
   * and on the GPU path it is uninstalling the hook, after which the engine rebuilds its draw
   * command without it and draws the attribute texture it never stopped holding. The next
   * `apply` derives a new attachment on the new path. Tile bindings do not depend on the path
   * and are kept, so a switch costs no re-binding.
   */
  setGpu(factory: MotionTextureFactory | undefined): void {
    if (factory === this.#gpuFactory) return;
    this.#gpuFactory = factory;
    if (this.#phase === "refused" || this.#phase === "detached") return;
    try {
      this.#restore();
    } catch (error) {
      // Never leave a switch half done: the hook still goes and the attachment is re-derived.
      log.warn("restoring the measured pose before a path switch failed", {
        error: String(error),
      });
    }
    this.#gpu?.destroy();
    this.#gpu = undefined;
    this.#attachment = undefined;
    this.#lastRange = undefined;
    this.#displaced = false;
    this.#phase = "waiting";
    this.#reason = "no-snapshot";
  }

  /**
   * Writes the measured pose through the current attachment, when it is displaced and its
   * snapshot is still the committed one. A snapshot that has rebuilt since is already at rest —
   * the engine packed its texture afresh — and the old rows must not be written into it.
   */
  #restore(): void {
    const attachment = this.#attachment;
    if (attachment === undefined || !this.#displaced) return;
    const primitive = this.#tileset.gaussianSplatPrimitive;
    if (primitive === undefined || !isCurrent(attachment, primitive)) return;
    const rest = new Array<NodeTransform>(this.#rig.nodes.length).fill(IDENTITY_TRANSFORM);
    this.#write(attachment, rest, FLUTTER_STILL);
  }

  #refuse(reason: DeformerReason, detail: string): void {
    this.#phase = "refused";
    this.#reason = reason;
    this.#attachment = undefined;
    this.#lastRange = undefined;
    this.#displaced = false;
    this.#gpu?.destroy();
    this.#gpu = undefined;
    log.warn("refusing to deform", { reason, detail });
  }

  #wait(reason: DeformerReason): undefined {
    this.#phase = "waiting";
    this.#reason = reason;
    return undefined;
  }

  /**
   * Returns the attachment for the primitive's current snapshot, deriving a new one when the
   * snapshot has rebuilt.
   *
   * `_snapshot.generation` is a monotonic counter bumped on every rebuild, and `_positions`
   * identity and `_numSplats` guard it. None of them says *which* splats these are; that is
   * the per-tile checksum's job, over positions un-baked back to the rig's frame, so it holds
   * across any model matrix — a re-baked tree still matches and is re-derived from its new
   * base, and a different tileset does not match and is refused for good.
   */
  #syncAttachment(): Attachment | undefined {
    const primitive = this.#tileset.gaussianSplatPrimitive;
    if (primitive === undefined) return this.#wait("no-primitive");

    const positions = primitive._positions;
    const numSplats = primitive._numSplats ?? 0;
    const rowMask = primitive._splatRowMask ?? 0;
    const rowShift = primitive._splatRowShift ?? 0;
    const generation = primitive._snapshot?.generation ?? -1;
    if (positions === undefined || numSplats <= 0 || rowShift <= 0) {
      return this.#wait("no-snapshot");
    }

    const current = this.#attachment;
    if (current !== undefined && isCurrent(current, primitive)) return current;

    const started = performance.now();
    const next = this.#derive(primitive, positions, numSplats, rowMask, rowShift, generation);
    if (next === undefined) return undefined;
    this.#lastDeriveMs = performance.now() - started;
    if (current !== undefined) this.#rederivations += 1;
    this.#attachment = next;
    // A new snapshot's texture holds the measured positions again, whatever we last wrote.
    this.#displaced = false;
    this.#lastRange = undefined;
    this.#phase = "ready";
    this.#reason = undefined;
    log.info("attached to splat primitive", {
      generation,
      numSplats,
      tiles: next.binding.tiles.length,
      motion: next.motion,
      rows: next.layout.height,
      bakeResidualM: next.binding.bakeResidualM,
      deriveMs: Math.round(this.#lastDeriveMs * 10) / 10,
    });
    return next;
  }

  #derive(
    primitive: SplatPrimitive,
    positions: Float32Array,
    numSplats: number,
    rowMask: number,
    rowShift: number,
    generation: number,
  ): Attachment | undefined {
    let layout: SplatTextureLayout;
    try {
      layout = splatTextureLayout(numSplats, rowMask, rowShift);
    } catch (error) {
      this.#refuse("layout", String(error));
      return undefined;
    }

    const rootTransform = primitive._rootTransform;
    if (rootTransform === undefined) return this.#wait("no-snapshot");
    const geodeticAlignment = geodeticFrameAlignment(rootTransform);
    if (!(geodeticAlignment >= GEODETIC_ALIGNMENT_MIN)) {
      this.#refuse(
        "frame",
        `root transform Z column is ${String(geodeticAlignment)} against the geodetic normal`,
      );
      return undefined;
    }

    if (this.#forest && this.#plantBinding === undefined) {
      this.#refuse("binding", "a forest rig needs its plant binding, and none was given");
      return undefined;
    }

    const listed = snapshotTiles(this.#tileset, primitive, positions, numSplats);
    if (listed.kind === "wait") {
      return this.#wait(listed.reason === "no-bake-transform" ? "no-bake-transform" : "tiles");
    }
    const tiles = listed.tiles;
    const first = tiles[0];
    const sharedBake =
      first !== undefined &&
      tiles.every((tile) => tile.bake.length === 16 && sameBake(tile.bake, first.bake));
    const cpuReason: DeformerCpuReason | undefined =
      this.#gpuFactory === undefined
        ? "no-factory"
        : !hasVertexMotionHook(primitive)
          ? "no-hook"
          : !sharedBake
            ? "mixed-bake"
            : undefined;
    const motion: DeformerMotion = cpuReason === undefined ? "gpu" : "cpu";

    // Cheap and first on the CPU path: without the packed buffer there is nothing to stage
    // into, and the binding below is work we would otherwise repeat every frame while waiting.
    let packed: Uint32Array | undefined;
    if (motion === "cpu") {
      const capture = findSplatCapture(numSplats, digestSplatPositions(positions, numSplats));
      if (capture === undefined) return this.#wait("no-capture");
      packed = capture.data;
    }

    const bindings: TileBinding[] = [];
    let boundTiles = 0;
    let boundSplats = 0;
    let bindMs = 0;
    for (const tile of tiles) {
      const cached = this.#cache.get(tile);
      if (cached !== undefined) {
        bindings.push(cached);
        continue;
      }
      const baked = positions.subarray(tile.start * 3, (tile.start + tile.count) * 3);
      const result = bindTile(
        baked,
        tile.bake,
        this.#rig,
        this.#accepted,
        BAKE_RESIDUAL_LIMIT_M,
        this.#plantBinding,
      );
      if (result.kind === "refused") {
        this.#observedChecksum = result.checksum;
        this.#refuse(result.reason, result.detail);
        return undefined;
      }
      this.#cache.set(tile, result.binding);
      bindings.push(result.binding);
      boundTiles += 1;
      boundSplats += tile.count;
      bindMs += result.binding.bindMs;
    }
    this.#lastBoundTiles = boundTiles;
    this.#lastBoundSplats = boundSplats;
    this.#lastBindMs = bindMs;
    const binding = aggregateBindings(tiles, bindings, numSplats);
    this.#observedChecksum = tiles.length === 1 ? bindings[0]?.checksum : undefined;

    // A forest is a scene, not a standing tree: the frame check above stands for "up", and the
    // scene step measured every plant's height against its own ground.
    if (!this.#forest && (this.#uprightness === undefined || !this.#uprightness.upright)) {
      const uprightness = treeUprightness(binding.canonicalLocal);
      // A single-tile rig is decided on its one tile, as it always was.
      if (numSplats >= UPRIGHT_MIN_SPLATS || this.#rig.tileChecksums === undefined) {
        this.#uprightness = uprightness;
        if (!uprightness.upright) {
          this.#refuse(
            "upright",
            `crown/base spread ratio ${String(uprightness.crownToBase)} does not read as a standing tree`,
          );
          return undefined;
        }
      }
    }

    const nodeMoves = new Uint8Array(this.#rig.nodes.length);
    if (motion === "gpu") {
      const gpu = this.#gpu ?? this.#createGpu();
      const bake = binding.commonBake;
      if (gpu === undefined || bake === undefined) return this.#wait("internal");
      gpu.install(primitive);
      if (!gpu.bind(generation, layout, binding.skin, binding.flutterKeys, bake)) {
        this.#refuse("bake", "the shared bake transform is singular");
        return undefined;
      }
    } else {
      // A snapshot the GPU path cannot take: make sure its hook draws nothing displaced.
      this.#gpu?.deactivate();
    }

    let staticSplats = 0;
    const labels = binding.plantLabels;
    if (labels !== undefined) for (const label of labels) if (label === 0) staticSplats += 1;
    return {
      generation,
      enginePositions: positions,
      numSplats,
      staticSplats,
      layout,
      binding,
      motion,
      cpuReason,
      cpu:
        motion === "cpu" && packed !== undefined
          ? {
              canonicalBaked: positions.slice(0, numSplats * 3),
              staging: createStagingBuffer(layout, packed),
              displacedLocal: new Float32Array(numSplats * 3),
              displacedBaked: new Float32Array(numSplats * 3),
            }
          : undefined,
      nodeMoves,
      geodeticAlignment,
    };
  }

  #createGpu(): SplatGpuMotion | undefined {
    const factory = this.#gpuFactory;
    if (factory === undefined) return undefined;
    this.#gpu = new SplatGpuMotion(factory, this.#rig.nodes.length);
    return this.#gpu;
  }

  #write(
    attachment: Attachment,
    transforms: readonly NodeTransform[],
    flutter: FlutterField,
  ): void {
    const moving = pinStaticNodes(
      attachment.nodeMoves,
      this.#pinned,
      markMovingNodes(attachment.nodeMoves, transforms, flutter),
    );

    // Nothing moves and nothing is displaced: an idle scene stays idle.
    if (!moving && !this.#displaced) return;

    if (attachment.motion === "gpu") {
      const gpu = this.#gpu;
      if (!gpu?.write(transforms, flutter, attachment.nodeMoves, moving)) {
        // No context yet: the first draw-command build hands it over. Nothing is displaced.
        this.#displaced = false;
        this.#phase = "waiting";
        this.#reason = "no-texture";
        return;
      }
      if (gpu.lastUploadWords > 0) this.#uploads += 1;
      this.#lastUploadRows = gpu.lastUploadWords > 0 ? 1 : 0;
      this.#lastUploadWords = gpu.lastUploadWords;
      this.#displaced = moving;
      this.#phase = "ready";
      this.#reason = undefined;
      return;
    }

    const cpu = attachment.cpu;
    if (cpu === undefined) return;
    const { binding } = attachment;
    deformPositions(
      binding.canonicalLocal,
      binding.assignment,
      transforms,
      cpu.displacedLocal,
      flutter,
      binding.flutterKeys,
      binding.skin,
    );
    // Per tile, each through its own bake matrix. Splats whose nodes are all at rest get the
    // engine's own bytes back, which is what makes calm exact.
    for (const tile of binding.tiles) {
      const from = tile.start * 3;
      const to = (tile.start + tile.count) * 3;
      resolveBakedPositions(
        cpu.displacedLocal.subarray(from, to),
        cpu.canonicalBaked.subarray(from, to),
        skinSlice(binding.skin, tile.start, tile.count),
        attachment.nodeMoves,
        tile.bake,
        cpu.displacedBaked.subarray(from, to),
      );
    }
    writeSplatPositions(cpu.staging, cpu.displacedBaked);

    // Returning to rest has to repaint whatever the last frame moved, not what this one does.
    const range = moving
      ? (rowRangeForMovingNodes(binding.skin, attachment.nodeMoves, attachment.layout) ??
        fullRowRange(attachment.layout))
      : (this.#lastRange ?? fullRowRange(attachment.layout));

    // Re-read every write: the engine destroys and recreates this texture on a dimension change,
    // and the draw command's uniform closed over the object, not over a getter.
    const texture = this.#tileset.gaussianSplatPrimitive?.gaussianSplatTexture;
    if (texture === undefined || texture.isDestroyed()) {
      this.#phase = "waiting";
      this.#reason = "no-texture";
      return;
    }

    texture.copyFrom({
      source: {
        width: attachment.layout.width,
        height: range.rowCount,
        arrayBufferView: rowSlice(cpu.staging, range),
      },
      xOffset: 0,
      yOffset: range.firstRow,
    });

    this.#uploads += 1;
    this.#lastUploadRows = range.rowCount;
    this.#lastUploadWords = rowRangeWordCount(attachment.layout, range);
    this.#lastRange = moving ? range : undefined;
    this.#displaced = moving;
    this.#phase = "ready";
    this.#reason = undefined;
  }
}

/**
 * Whether `attachment` was derived from the primitive's committed snapshot: same generation,
 * same engine array, same count and addressing.
 */
function isCurrent(attachment: Attachment, primitive: SplatPrimitive): boolean {
  return (
    attachment.generation === (primitive._snapshot?.generation ?? -1) &&
    attachment.enginePositions === primitive._positions &&
    attachment.numSplats === (primitive._numSplats ?? 0) &&
    attachment.layout.rowMask === (primitive._splatRowMask ?? 0) &&
    attachment.layout.rowShift === (primitive._splatRowShift ?? 0)
  );
}

function sameBake(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  for (let i = 0; i < 16; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * Holds every pinned node (`pinned[n] = 1`, a forest rig's static anchors) at rest in
 * `nodeMoves`, and says whether anything still moves. `moving` is what `markMovingNodes` said
 * before the pins: when nothing moved, nothing is re-read.
 */
export function pinStaticNodes(
  nodeMoves: Uint8Array,
  pinned: Uint8Array,
  moving: boolean,
): boolean {
  if (!moving) return false;
  let any = false;
  for (let n = 0; n < nodeMoves.length; n += 1) {
    if ((pinned[n] ?? 0) !== 0) nodeMoves[n] = 0;
    else if ((nodeMoves[n] ?? 0) !== 0) any = true;
  }
  return any;
}

/**
 * Flags the nodes whose splats are not at rest, and says whether any are.
 *
 * Exact comparison, not a tolerance: `deform` returns bit-exact identity at zero wind by
 * construction (`quatFromAxisAngle` returns the identity quaternion by value at angle 0), so
 * "nothing is moving" is a fact about the numbers rather than a judgement about them.
 *
 * A node counts as moving when its transform is not the rest transform **or** it has a non-zero
 * flutter amplitude. The second half matters because `resolveBakedPositions` copies the
 * canonical bytes for any splat whose node is flagged still — so a node that fluttered while
 * flagged still would have its shimmer silently thrown away between the deform and the upload.
 * The root's transform is identity at every wind, and the root is the base of the bole, so on a
 * rig whose trunk is thick enough not to flutter this changes nothing; on one where a thin node
 * happens to be the root, it is the difference between shimmering and not.
 */
export function markMovingNodes(
  nodeMoves: Uint8Array,
  transforms: readonly NodeTransform[],
  flutter: FlutterField = FLUTTER_STILL,
): boolean {
  let moving = false;
  for (let n = 0; n < nodeMoves.length; n += 1) {
    const transform = transforms[n];
    const rest =
      (transform === undefined ||
        (transform.rotation[0] === 0 &&
          transform.rotation[1] === 0 &&
          transform.rotation[2] === 0 &&
          transform.rotation[3] === 1 &&
          transform.translation[0] === 0 &&
          transform.translation[1] === 0 &&
          transform.translation[2] === 0)) &&
      (flutter.still || (flutter.amplitudeM[n] ?? 0) === 0);
    nodeMoves[n] = rest ? 0 : 1;
    if (!rest) moving = true;
  }
  return moving;
}
