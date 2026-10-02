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
 * wanted: the camera moves at once, over what is already drawn, and detail follows. (The
 * decodes have since moved to workers, splatDecoder.ts; the rebuilds remain.)
 *
 * So while the camera moves (and for `REST_MS` after, so two quick gestures are one), there
 * are **no snapshot rebuilds** (`primitive.holdRebuilds`, engine patch): the committed snapshot
 *   keeps drawing and its re-sort (in a worker) keeps it correct from every angle. A long
 *   continuous motion -- a walk, a flight -- still gets one every `MAX_HOLD_MS`, so detail
 *   is never frozen for good.
 *
 * What makes holding safe is the other half of the patch, `tileset.selectOffscreen`
 * (providers/tiles.ts sets it on splat tilesets): the snapshot holds the whole scan -- coarse
 * out of view -- rather than only what was in the frustum, so turning while it is held shows
 * blurry splats, never a hole.
 *
 * At rest nothing is held, and the render is requested so a rebuild held back starts at once
 * in request-render mode.
 */

import { PrimitiveCollection, type Scene } from "cesium";

import type { Emitter } from "@/lib/emitter";

import type { SplatPrimitive } from "./splatInternals";
import type { SceneEvents } from "./types";

/** Motion stops counting this long after the camera's last change. */
export const REST_MS = 200;
/** A continuous motion lets a rebuild start this often, in a window of `RELEASE_MS`. */
export const MAX_HOLD_MS = 2500;
const RELEASE_MS = 120;

export interface GateState {
  /** Rebuilds are held this frame. */
  hold: boolean;
}

/**
 * The gate's decision for a frame at `now`, from when motion last ended (`restAt`, or null
 * while moving), when the current motion began and when a rebuild was last let through. Pure, so the timing is testable without a scene.
 */
export function gateAt(
  now: number,
  motion: { moving: boolean; endedAt: number; startedAt: number },
  last: { releaseAt: number },
): GateState & { release: boolean } {
  const active = motion.moving || now - motion.endedAt < REST_MS;
  if (!active) return { hold: false, release: false };
  const since = Math.max(motion.startedAt, last.releaseAt);
  const releasing = now - last.releaseAt < RELEASE_MS && last.releaseAt >= motion.startedAt;
  const release = !releasing && now - since >= MAX_HOLD_MS;
  return { hold: !(releasing || release), release };
}

export class SplatMotionGate {
  private moving = false;
  private startedAt = 0;
  private endedAt = Number.NEGATIVE_INFINITY;
  private releaseAt = Number.NEGATIVE_INFINITY;
  private held = false;
  private readonly off: (() => void)[] = [];

  constructor(
    private readonly scene: Scene,
    events: Emitter<SceneEvents>,
    /** The interface is being used (uiActivity.ts): streaming holds back as for motion. */
    private readonly interfaceBusy: () => boolean = () => false,
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
      { releaseAt: this.releaseAt },
    );
    if (state.release) this.releaseAt = now;
    const primitives = splatPrimitives(this.scene.primitives);
    const hold = state.hold || this.interfaceBusy();
    for (const primitive of primitives) primitive.holdRebuilds = hold;
    const wasHeld = this.held;
    this.held = hold;
    // Request-render mode renders only when asked: a rebuild held back must be asked for
    // once it may run.
    if (primitives.length > 0 && wasHeld && !hold) this.scene.requestRender();
  }

  destroy(): void {
    for (const off of this.off) off();
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
