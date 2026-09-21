/**
 * Rewrites the position lanes of a Gaussian splat tileset's attribute texture, so a measured
 * tree can move without its measurement ever changing.
 *
 * The shape of the thing: `apply(nodeTransforms)` takes the rig transforms some caller computed
 * and pushes one frame to the GPU. It owns no clock, no wind, no scene listener and no UI —
 * `LivingSurveyManager` (S4) drives it from `scene.preUpdate`. Everything arithmetical lives in
 * the pure modules beside it (`splatTexels.ts`, `splatFrames.ts`), which is why almost all of
 * this is testable without a GPU.
 *
 * Three rules it exists to enforce:
 *
 * **Canonical positions are never mutated.** At attach the baked positions are copied once into
 * buffers this class owns, and every frame recomputes `displaced = f(canonical, transforms)`
 * from that pristine copy. There is no read-modify-write anywhere, so error cannot accumulate
 * and zero wind returns the exact measured bytes. `tile.content.positions`,
 * `primitive._positions`, `snapshot.positions` and the glTF POSITION array are read-only to us
 * — `transformTile` rewrites the first of those in place, so a write there would make a
 * deformation permanent and compound it across every rebuild.
 *
 * **Refuse rather than mislead.** Before its first write it checks that the tileset is
 * single-tile, that the root frame really is east-north-up, that the tree stands along local +Z,
 * and — bit-exactly — that these are the splats the rig was built for. On any failure it writes
 * nothing, ever, and says why. A tree that does not move is honest; a tree whose wrong splats
 * move is not.
 *
 * **Rebuilds re-derive, they do not re-apply.** `SiteManager.clampToGround` sets
 * `tileset.modelMatrix` after an asynchronous terrain sample, so every site rebuilds its
 * snapshot seconds after load with every baked position changed. See `#syncAttachment` for how
 * that is told apart from the wrong tileset.
 */

import {
  assignSplatsToNodes,
  checksumPositions,
  deformPositions,
  FLUTTER_STILL,
  type FlutterField,
  type MotionRig,
  type NodeTransform,
} from "@twin/world";

import { createLogger } from "@/lib/log";

import { digestSplatPositions, findSplatCapture } from "./splatCaptureRegistry";
import {
  GEODETIC_ALIGNMENT_MIN,
  geodeticFrameAlignment,
  invertAffine,
  maxAbsDifference,
  resolveBakedPositions,
  transformPositions,
  treeUprightness,
  unbakePositions,
  type Mat4,
  type TreeUprightness,
} from "./splatFrames";
import {
  bakeTransformOf,
  isSingleTile,
  type SplatPrimitive,
  type SplatTilesetLike,
} from "./splatInternals";
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
  /** The tile has not been baked yet, so the local frame is unknown. */
  | "no-bake-transform"
  /** The interception did not see this snapshot's packed buffer. */
  | "no-capture"
  /** More than one tile: splat indices are not stable, so the rig cannot be trusted. */
  | "multi-tile"
  /** The addressing parameters the primitive reports are not self-consistent. */
  | "layout"
  /** The root transform's Z column is not the geodetic normal. */
  | "frame"
  /** The tree's own points say local +Z is not up. */
  | "upright"
  /** These are not the splats the rig was built for. */
  | "checksum"
  /** The bake matrix is singular, or re-baking does not reproduce the engine's positions. */
  | "bake"
  /** A bug in this class. Reported rather than thrown, because it runs inside `preUpdate`. */
  | "internal";

/** Everything the manager and the HUD need to know, and nothing they have to guess. */
export interface DeformerStatus {
  readonly phase: DeformerPhase;
  readonly reason?: DeformerReason;
  readonly numSplats: number;
  /** `primitive._snapshot.generation` the current attachment was derived from. */
  readonly generation: number;
  /** Times the snapshot rebuilt and the assignment was re-derived from a new base. */
  readonly rederivations: number;
  /** Frames that reached `copyFrom`. */
  readonly uploads: number;
  /** Rows and words uploaded by the most recent write. */
  readonly lastUploadRows: number;
  readonly lastUploadWords: number;
  /** True while the texture holds displaced positions rather than the measured ones. */
  readonly displaced: boolean;
  /** Dot of the root transform's Z column with the geodetic normal; 1 is a perfect ENU frame. */
  readonly geodeticAlignment: number;
  /** Largest disagreement between our re-bake and the engine's baked positions, metres. */
  readonly bakeResidualM: number;
  /** The digest actually observed over the un-baked positions, for diagnosing a refusal. */
  readonly observedChecksum?: string;
  readonly uprightness?: TreeUprightness;
}

/**
 * A re-bake that disagrees with the engine by more than this is not a rounding difference —
 * the bake matrix is wrong, and deforming would move the tree somewhere it was never measured.
 */
export const BAKE_RESIDUAL_LIMIT_M = 1e-3;

export interface SplatDeformerOptions {
  /** A `Cesium3DTileset` viewed through `splatTilesetOf`, or a test double. */
  readonly tileset: SplatTilesetLike;
  readonly rig: MotionRig;
}

/** Everything derived from one snapshot generation. Replaced wholesale when the snapshot rebuilds. */
interface Attachment {
  readonly generation: number;
  /** The engine's own array, kept only to notice when it is swapped. Never written. */
  readonly enginePositions: Float32Array;
  readonly numSplats: number;
  readonly layout: SplatTextureLayout;
  readonly bakeMatrix: Mat4;
  /** Immutable. The rig's frame, recovered from the baked positions. */
  readonly canonicalLocal: Float32Array;
  /** Immutable. The engine's baked bytes, the exact rest pose. */
  readonly canonicalBaked: Float32Array;
  readonly assignment: Uint16Array;
  readonly staging: SplatStagingBuffer;
  /** Own `ArrayBuffer`s: `deformPositions` guards against `out === positions` but not aliasing. */
  readonly displacedLocal: Float32Array;
  readonly displacedBaked: Float32Array;
  readonly nodeMoves: Uint8Array;
  readonly uprightness: TreeUprightness;
  readonly geodeticAlignment: number;
  readonly bakeResidualM: number;
}

export class SplatDeformer {
  readonly #tileset: SplatTilesetLike;
  readonly #rig: MotionRig;
  #attachment: Attachment | undefined;
  #phase: DeformerPhase = "waiting";
  #reason: DeformerReason | undefined = "no-primitive";
  #observedChecksum: string | undefined;
  #displaced = false;
  #lastRange: SplatRowRange | undefined;
  #uploads = 0;
  #rederivations = 0;
  #lastUploadRows = 0;
  #lastUploadWords = 0;

  constructor(options: SplatDeformerOptions) {
    this.#tileset = options.tileset;
    this.#rig = options.rig;
  }

  get status(): DeformerStatus {
    const attachment = this.#attachment;
    return {
      phase: this.#phase,
      reason: this.#reason,
      numSplats: attachment?.numSplats ?? 0,
      generation: attachment?.generation ?? -1,
      rederivations: this.#rederivations,
      uploads: this.#uploads,
      lastUploadRows: this.#lastUploadRows,
      lastUploadWords: this.#lastUploadWords,
      displaced: this.#displaced,
      geodeticAlignment: attachment?.geodeticAlignment ?? Number.NaN,
      bakeResidualM: attachment?.bakeResidualM ?? Number.NaN,
      observedChecksum: this.#observedChecksum,
      uprightness: attachment?.uprightness,
    };
  }

  /** The canonical local positions, for metrics and tests. Treat as immutable. */
  get canonicalPositions(): Float32Array | undefined {
    return this.#attachment?.canonicalLocal;
  }

  /** Which rig node each splat belongs to. Treat as immutable. */
  get assignment(): Uint16Array | undefined {
    return this.#attachment?.assignment;
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
      if (attachment === undefined) return this.status;
      this.#write(attachment, transforms, flutter);
    } catch (error) {
      // A refusal is a decision; an unexpected throw is a bug, and it must not take the scene
      // with it. Both end in "this tileset does not deform".
      this.#refuse("internal", `unexpected failure: ${String(error)}`);
    }
    return this.status;
  }

  /** Releases the staging buffers and stops writing. */
  destroy(): void {
    this.#attachment = undefined;
    this.#phase = "detached";
    this.#reason = undefined;
    this.#displaced = false;
    this.#lastRange = undefined;
  }

  #refuse(reason: DeformerReason, detail: string): void {
    this.#phase = "refused";
    this.#reason = reason;
    this.#attachment = undefined;
    this.#lastRange = undefined;
    log.warn("refusing to deform", { reason, detail });
  }

  #wait(reason: DeformerReason): undefined {
    this.#phase = "waiting";
    this.#reason = reason;
    return undefined;
  }

  /**
   * Returns the attachment for the primitive's current snapshot, building it when the snapshot
   * has rebuilt.
   *
   * **The signal that decides refuse against re-derive.** `_snapshot.generation` is a monotonic
   * counter bumped on every rebuild, and `_positions` identity and `_numSplats` guard it — but
   * none of them says *which* tree this is. A rebuild is expected and routine:
   * `clampToGround` changes the model matrix, `transformTile` re-bakes every position from the
   * pristine glTF attribute, and the numbers in `_positions` all change while the tree is
   * unchanged. So rebuild detection alone cannot tell "the survey moved onto the terrain" from
   * "a different tileset is in this variable".
   *
   * What tells them apart is that the bake is undone before the identity test. The baked
   * positions go back through `B⁻¹` and onto SPZ's 1/4096 m grid, which reproduces the capture's
   * own float32s exactly; `checksumPositions` over those is bit-identical across any model
   * matrix, because the model matrix is exactly what was removed. A re-baked tree still matches
   * and is re-derived from its new base; a different tileset does not match and is refused for
   * good.
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
    if (
      current?.generation === generation &&
      current.enginePositions === positions &&
      current.numSplats === numSplats &&
      current.layout.rowMask === rowMask &&
      current.layout.rowShift === rowShift
    ) {
      return current;
    }

    const next = this.#derive(primitive, positions, numSplats, rowMask, rowShift, generation);
    if (next === undefined) return undefined;
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
      rows: next.layout.height,
      bakeResidualM: next.bakeResidualM,
      crownToBase: next.uprightness.crownToBase,
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
    const single = isSingleTile(this.#tileset);
    if (single === undefined) return this.#wait("no-primitive");
    if (!single) {
      this.#refuse("multi-tile", "splat indices are only stable for a single-tile tileset");
      return undefined;
    }
    const selected = primitive.selectedTileLength;
    if (typeof selected === "number" && selected > 1) {
      this.#refuse("multi-tile", `snapshot aggregates ${selected} tiles`);
      return undefined;
    }

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
        `root transform Z column is ${geodeticAlignment} against the geodetic normal`,
      );
      return undefined;
    }

    const bakeMatrix = bakeTransformOf(this.#tileset);
    if (bakeMatrix === undefined) return this.#wait("no-bake-transform");
    const inverse = invertAffine(bakeMatrix);
    if (inverse === undefined) {
      this.#refuse("bake", "the tile's bake transform is singular");
      return undefined;
    }

    // Cheap and first: without the packed buffer there is nothing to stage into, and the work
    // below is a few linear passes we would otherwise repeat every frame while waiting.
    const capture = findSplatCapture(numSplats, digestSplatPositions(positions, numSplats));
    if (capture === undefined) return this.#wait("no-capture");

    const canonicalBaked = positions.slice(0, numSplats * 3);
    const canonicalLocal = unbakePositions(canonicalBaked, inverse);

    const checksum = checksumPositions(canonicalLocal);
    this.#observedChecksum = checksum;
    if (checksum !== this.#rig.canonicalChecksum) {
      this.#refuse(
        "checksum",
        `un-baked positions digest ${checksum}, rig expects ${this.#rig.canonicalChecksum}`,
      );
      return undefined;
    }

    const uprightness = treeUprightness(canonicalLocal);
    if (!uprightness.upright) {
      this.#refuse(
        "upright",
        `crown/base spread ratio ${uprightness.crownToBase} does not read as a standing tree`,
      );
      return undefined;
    }

    // Proves in one number that the bake matrix is right, that the un-bake recovered the true
    // source, and that our arithmetic matches the engine's. Expected to be exactly 0 on the
    // rigid fast path `transformTile` takes for a placed capture.
    const rebaked = transformPositions(canonicalLocal, bakeMatrix, new Float32Array(numSplats * 3));
    const bakeResidualM = maxAbsDifference(rebaked, canonicalBaked);
    if (!(bakeResidualM <= BAKE_RESIDUAL_LIMIT_M)) {
      this.#refuse("bake", `re-baking disagrees with the engine by ${bakeResidualM} m`);
      return undefined;
    }

    return {
      generation,
      enginePositions: positions,
      numSplats,
      layout,
      bakeMatrix: Array.from(bakeMatrix),
      canonicalLocal,
      canonicalBaked,
      assignment: assignSplatsToNodes(canonicalLocal, this.#rig),
      staging: createStagingBuffer(layout, capture.data),
      displacedLocal: new Float32Array(numSplats * 3),
      displacedBaked: new Float32Array(numSplats * 3),
      nodeMoves: new Uint8Array(this.#rig.nodes.length),
      uprightness,
      geodeticAlignment,
      bakeResidualM,
    };
  }

  #write(
    attachment: Attachment,
    transforms: readonly NodeTransform[],
    flutter: FlutterField,
  ): void {
    const moving = markMovingNodes(attachment.nodeMoves, transforms, flutter);

    // Nothing moves and nothing is displaced: an idle scene stays idle.
    if (!moving && !this.#displaced) return;

    deformPositions(
      attachment.canonicalLocal,
      attachment.assignment,
      transforms,
      attachment.displacedLocal,
      flutter,
    );
    resolveBakedPositions(
      attachment.displacedLocal,
      attachment.canonicalBaked,
      attachment.assignment,
      attachment.nodeMoves,
      attachment.bakeMatrix,
      attachment.displacedBaked,
    );
    writeSplatPositions(attachment.staging, attachment.displacedBaked);

    // Returning to rest has to repaint whatever the last frame moved, not what this one does.
    const range = moving
      ? (rowRangeForMovingNodes(attachment.assignment, attachment.nodeMoves, attachment.layout) ??
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
        arrayBufferView: rowSlice(attachment.staging, range),
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
