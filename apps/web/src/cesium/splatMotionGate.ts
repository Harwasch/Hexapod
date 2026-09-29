/**
 * Camera first, streaming second: what splat tilesets may spend on the main thread while the
 * camera moves.
 *
 * Measured on the Fort Clatsop site (22.6M gaussians, 514 tiles) during a 20 s orbit, with a
 * CPU profile: each time a new level of detail was selected, CesiumJS rebuilt the splat
 * primitive's snapshot -- re-aggregating every selected splat, regenerating and re-uploading
 * one texture of all of them -- about 265 ms of main-thread work for 1.25M splats, seven times;
 * and each arriving tile's SPZ was decoded on the main thread (spz-loader's WASM), tens of
 * milliseconds apiece. Both land in the frames of a gesture. Google Maps' rule is the one
 * wanted: the camera moves at once, over what is already drawn, and detail follows.
 *
 * So while the camera moves (and for `REST_MS` after, so two quick gestures are one):
 *
 * - **no snapshot rebuilds** (`primitive.holdRebuilds`, engine patch): the committed snapshot
 *   keeps drawing and its re-sort (in a worker) keeps it correct from every angle. A long
 *   continuous motion -- a walk, a flight -- still gets one every `MAX_HOLD_MS`, so detail
 *   is never frozen for good;
 * - **a trickle of decodes** (`frameState.splatDecodesAllowed`, engine patch): one tile every
 *   `DECODE_INTERVAL_MS`, so a burst of arrivals cannot stack decodes into one frame.
 *
 * What makes holding safe is the other half of the patch, `tileset.selectOffscreen`
 * (providers/tiles.ts sets it on splat tilesets): the snapshot holds the whole scan -- coarse
 * out of view -- rather than only what was in the frustum, so turning while it is held shows
 * blurry splats, never a hole.
 *
 * At rest nothing is capped, and the render is requested so a rebuild held back starts at
 * once in request-render mode.
 */

import { PrimitiveCollection, type Scene } from "cesium";

import type { Emitter } from "@/lib/emitter";

import { splatFrameStateOf, type SplatPrimitive } from "./splatInternals";
import type { SceneEvents } from "./types";

/** Motion stops counting this long after the camera's last change. */
export const REST_MS = 200;
/** A continuous motion lets a rebuild start this often, in a window of `RELEASE_MS`. */
export const MAX_HOLD_MS = 2500;
const RELEASE_MS = 120;
/** While moving, at most one tile's SPZ decode starts per this interval. */
export const DECODE_INTERVAL_MS = 250;

export interface GateState {
  /** Rebuilds are held this frame. */
  hold: boolean;
  /** Decodes that may start this frame; undefined for no cap. */
  decodes: number | undefined;
}

/**
 * The gate's decision for a frame at `now`, from when motion last ended (`restAt`, or null
 * while moving), when the current motion began, when a rebuild was last let through and
 * when a decode last started. Pure, so the timing is testable without a scene.
 */
export function gateAt(
  now: number,
  motion: { moving: boolean; endedAt: number; startedAt: number },
  last: { releaseAt: number; decodeAt: number },
): GateState & { release: boolean } {
  const active = motion.moving || now - motion.endedAt < REST_MS;
  if (!active) return { hold: false, decodes: undefined, release: false };
  const since = Math.max(motion.startedAt, last.releaseAt);
  const releasing = now - last.releaseAt < RELEASE_MS && last.releaseAt >= motion.startedAt;
  const release = !releasing && now - since >= MAX_HOLD_MS;
  const decodes = now - last.decodeAt >= DECODE_INTERVAL_MS ? 1 : 0;
  return { hold: !(releasing || release), decodes, release };
}

export class SplatMotionGate {
  private moving = false;
  private startedAt = 0;
  private endedAt = Number.NEGATIVE_INFINITY;
  private releaseAt = Number.NEGATIVE_INFINITY;
  private decodeAt = Number.NEGATIVE_INFINITY;
  private held = false;
  private readonly off: (() => void)[] = [];

  constructor(
    private readonly scene: Scene,
    events: Emitter<SceneEvents>,
  ) {
    this.off.push(
      events.on("motion", (moving) => {
        const now = performance.now();
        if (moving && !this.moving && now - this.endedAt >= REST_MS) this.startedAt = now;
        if (!moving && this.moving) this.endedAt = now;
        this.moving = moving;
      }),
      // preUpdate runs every widget tick, rendered or not, and before the tilesets update.
      scene.preUpdate.addEventListener(() => this.tick()),
    );
  }

  /** Whether splat rebuilds are held right now (for the debug overlay and tests). */
  get holding(): boolean {
    return this.held;
  }

  private tick(): void {
    const now = performance.now();
    const state = gateAt(
      now,
      { moving: this.moving, endedAt: this.endedAt, startedAt: this.startedAt },
      { releaseAt: this.releaseAt, decodeAt: this.decodeAt },
    );
    if (state.release) this.releaseAt = now;
    const frameState = splatFrameStateOf(this.scene);
    const before = frameState.splatDecodesAllowed;
    frameState.splatDecodesAllowed = state.decodes;
    // A decode granted this frame is spent (or not) by the loaders; either way the next one
    // waits a full interval, which is the trickle.
    if (state.decodes === 1) this.decodeAt = now;
    const primitives = splatPrimitives(this.scene.primitives);
    for (const primitive of primitives) primitive.holdRebuilds = state.hold;
    const wasHeld = this.held;
    this.held = state.hold;
    // Request-render mode renders only when asked: a rebuild or decode held back must be
    // asked for once it may run.
    if (
      primitives.length > 0 &&
      ((wasHeld && !state.hold) || (before !== undefined && state.decodes === undefined))
    ) {
      this.scene.requestRender();
    }
  }

  destroy(): void {
    for (const off of this.off) off();
    const frameState = splatFrameStateOf(this.scene);
    frameState.splatDecodesAllowed = undefined;
    for (const primitive of splatPrimitives(this.scene.primitives)) primitive.holdRebuilds = false;
  }
}

/** Every splat primitive under a collection, nested collections included. */
function splatPrimitives(collection: PrimitiveCollection): SplatPrimitive[] {
  const found: SplatPrimitive[] = [];
  const visit = (items: PrimitiveCollection): void => {
    for (let i = 0; i < items.length; i++) {
      const item: unknown = items.get(i);
      if (item instanceof PrimitiveCollection) visit(item);
      else {
        const primitive = (item as { gaussianSplatPrimitive?: SplatPrimitive } | undefined)
          ?.gaussianSplatPrimitive;
        if (primitive) found.push(primitive);
      }
    }
  };
  visit(collection);
  return found;
}
