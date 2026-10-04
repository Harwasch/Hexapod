/**
 * A scan's split objects (docs/SCENE_OBJECTS.md §4 "Split objects", C4) drawn by a dedicated
 * splat renderer at their poses, as `cesium/splitObjects.ts` draws them under CesiumJS: each
 * object's one tile is loaded by the back-end like any scan tile (so its splats bind to the
 * scan's instance ids by checksum, and hide, highlight and rigid motion act on it), added beside
 * the scan's tiles, and placed every frame where its pose puts it.
 *
 * Frames: an object's tileset root is the scan's root `S` times `T(origin)`, its positions
 * relative to `origin`. The renderer draws in the scan's frame (`S`'s), so an object's tile
 * goes under `L · S⁻¹ · O` (`O` its root transform, `L = T(origin + t) R T(−origin)` its pose,
 * `poseLocalMatrix`): at rest exactly where it was measured. The pose is the store's
 * (`useSceneObjects.setPose`, what a driver or a person sets) over the one the scan declares.
 */

import { resolveLayerUrl } from "@/lib/inferred";
import { createLogger } from "@/lib/log";
import {
  poseLocalMatrix,
  splitObjectsOf,
  type ObjectPose,
  type SplitObjectRef,
} from "@/lib/sceneObjects";
import { useSceneObjects } from "@/state/sceneObjects";
import { parseTileset, type TileNode } from "@/view/tiles";

import { invertAffine, type Mat4 } from "../splatFrames";
import type { ScanBackend } from "./types";

const log = createLogger("scan-objects");

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** `a · b`, column-major 4x4. */
export function multiply4(a: Mat4, b: Mat4): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c += 1)
    for (let r = 0; r < 4; r += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += (a[k * 4 + r] ?? 0) * (b[c * 4 + k] ?? 0);
      out[c * 4 + r] = sum;
    }
  return out;
}

/** A tileset JSON's `root.transform` (column-major), or the identity. */
export function rootTransformOf(tileset: unknown): number[] {
  const transform = (tileset as { root?: { transform?: unknown } } | null)?.root?.transform;
  return Array.isArray(transform) &&
    transform.length === 16 &&
    transform.every((v) => typeof v === "number" && Number.isFinite(v))
    ? (transform as number[])
    : IDENTITY.slice();
}

/**
 * Where an object's tile is drawn in the scan's frame: `L · S⁻¹ · O` for the scan's root
 * transform `S`, the object's `O` and its pose `L` about `origin` (column-major 4x4).
 */
export function objectPlacement(
  scanRoot: Mat4,
  objectRoot: Mat4,
  ref: Pick<SplitObjectRef, "origin">,
  pose: ObjectPose,
): number[] {
  const inverse = invertAffine(scanRoot) ?? IDENTITY;
  return multiply4(poseLocalMatrix(ref.origin, pose), multiply4(inverse, objectRoot));
}

interface Placed<M> {
  ref: SplitObjectRef;
  mesh: M;
  base: number[];
  key: string;
}

/**
 * One session's split objects: loaded once, placed at their poses as each frame is drawn
 * (`tick`, which says whether any moved). `onLoaded` is told as each object is added: the
 * overlay draws only when something changes (overlayFrames.ts), and an object arriving is a
 * change no camera makes.
 */
export class ScanObjects<M> {
  readonly #placed: Placed<M>[] = [];
  #stopped = false;

  constructor(
    private readonly backend: Pick<ScanBackend<M>, "load" | "add" | "remove" | "dispose" | "place">,
    private readonly assetId: string | undefined,
    private readonly onLoaded: () => void = () => undefined,
  ) {}

  /** Objects drawn now. */
  get count(): number {
    return this.#placed.length;
  }

  /**
   * Loads the objects the scan at `tilesetUrl` declares (its root `extras`), its root
   * transform `scanRoot`. Objects that fail are logged and skipped.
   */
  async load(
    tilesetUrl: string,
    extras: unknown,
    scanRoot: Mat4,
    fetchJson: (url: string) => Promise<unknown> = async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`answered ${String(response.status)}`);
      return response.json();
    },
  ): Promise<void> {
    if (!this.backend.place) return;
    const refs = splitObjectsOf(extras);
    await Promise.all(
      refs.map(async (ref) => {
        try {
          const url = resolveLayerUrl(tilesetUrl, ref.uri);
          const json = await fetchJson(url);
          const root: TileNode = parseTileset(json).root;
          const mesh = await this.backend.load(url, root);
          if (this.#stopped) {
            this.backend.dispose(mesh);
            return;
          }
          const base = multiply4(invertAffine(scanRoot) ?? IDENTITY, rootTransformOf(json));
          this.backend.add(mesh);
          const placed: Placed<M> = { ref, mesh, base, key: "" };
          this.#placed.push(placed);
          this.#place(placed);
          this.onLoaded();
        } catch (error) {
          log.warn("split object did not load", {
            instance: ref.instance,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }),
    );
  }

  /** Places every object at its pose now; true when any moved. */
  tick(): boolean {
    let moved = false;
    for (const placed of this.#placed) moved = this.#place(placed) || moved;
    return moved;
  }

  stop(): void {
    this.#stopped = true;
    for (const { mesh } of this.#placed.splice(0)) {
      this.backend.remove(mesh);
      this.backend.dispose(mesh);
    }
  }

  #place(placed: Placed<M>): boolean {
    const pose =
      (this.assetId !== undefined
        ? useSceneObjects.getState().poses[this.assetId]?.[placed.ref.instance]
        : undefined) ?? placed.ref.pose;
    const key = `${pose.translation.join()}|${pose.rotation.join()}`;
    if (key === placed.key) return false;
    placed.key = key;
    this.backend.place?.(
      placed.mesh,
      multiply4(poseLocalMatrix(placed.ref.origin, pose), placed.base),
    );
    return true;
  }
}
