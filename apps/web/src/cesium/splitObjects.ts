/**
 * Draws a scan's split objects (lib/sceneObjects.ts) beside it: each its own tileset, placed
 * where the scan is and moved by its pose, shown while the scan is shown, faded by its own
 * view cones, and hidden and highlighted by the scan's instance ids -- its tile is bound in
 * the scan's `instances.json`, so the objects panel acts on it as on any other part of the
 * scan (`attachInstances` as a follower).
 *
 * World placement: an object's root transform is the scan's root `S` times `T(origin)`;
 * drawn under the model matrix `P · S · L · S⁻¹` (`L` = `poseLocalMatrix`, `P` the scan's
 * model matrix) it lands at `P · S · T(origin + t) · R`, its pose in the scan's frame
 * wherever the scan itself has been put. A pose set in the store (`useSceneObjects.setPose`,
 * a driver or a person) overrides the one the scan declares.
 */
import { Cesium3DTileset, Matrix4, type Scene } from "cesium";

import { resolveLayerUrl } from "@/lib/inferred";
import { createLogger } from "@/lib/log";
import {
  isRest,
  poseLocalMatrix,
  splitObjectsOf,
  type ObjectPose,
  type SplitObjectRef,
} from "@/lib/sceneObjects";
import { useSceneObjects } from "@/state/sceneObjects";

import { keepOffscreenSplats } from "./splatInternals";
import { attachInstances, setInstanceOffset } from "./splatInstances";
import { attachViewCones } from "./splatViewCones";

const log = createLogger("objects");

export type LoadObject = (url: string, parent: Cesium3DTileset) => Promise<Cesium3DTileset>;
export type AttachFollower = (tileset: Cesium3DTileset, assetId: string) => () => void;

async function loadObject(url: string, parent: Cesium3DTileset): Promise<Cesium3DTileset> {
  const tileset = await Cesium3DTileset.fromUrl(url, {
    maximumScreenSpaceError: parent.maximumScreenSpaceError,
    show: false,
    enableCollision: false,
  });
  keepOffscreenSplats(tileset);
  tileset.cullRequestsWhileMoving = false;
  return tileset;
}

/** The model matrix that draws an object at `pose` under a scan at `parentModel`. */
export function objectModelMatrix(
  parentModel: Matrix4,
  sceneRoot: Matrix4,
  ref: Pick<SplitObjectRef, "origin">,
  pose: ObjectPose,
  result = new Matrix4(),
): Matrix4 {
  if (isRest(pose)) return Matrix4.clone(parentModel, result);
  const local = Matrix4.fromArray(poseLocalMatrix(ref.origin, pose));
  const inverse = Matrix4.inverse(sceneRoot, new Matrix4());
  const m = Matrix4.multiply(parentModel, sceneRoot, new Matrix4());
  Matrix4.multiply(m, local, m);
  return Matrix4.multiply(m, inverse, result);
}

/** The ids an object's tileset carries (its instance and what it absorbed). */
function idsOf(tileset: Cesium3DTileset, ref: SplitObjectRef): number[] {
  const extras = (tileset.root as { extras?: { object?: { ids?: unknown } } } | undefined)?.extras;
  const ids = extras?.object?.ids;
  return Array.isArray(ids) && ids.every((v) => Number.isInteger(v))
    ? (ids as number[])
    : [ref.instance];
}

/** Loads `parent`'s split objects into `scene`; returns the disposer. */
export function attachSplitObjects(
  parent: Cesium3DTileset,
  scene: Pick<Scene, "primitives" | "preUpdate" | "requestRender">,
  assetId: string,
  load: LoadObject = loadObject,
  follow: AttachFollower = (tileset, id) =>
    attachInstances(tileset, scene, id, undefined, undefined, { follower: true }),
): () => void {
  const root = parent.root as { extras?: unknown; transform?: Matrix4 } | undefined;
  const refs = splitObjectsOf(root?.extras);
  const url = (parent as unknown as { resource?: { url?: string } }).resource?.url;
  if (!refs.length || !url) return () => undefined;
  const sceneRoot = root?.transform
    ? Matrix4.clone(root.transform)
    : Matrix4.clone(Matrix4.IDENTITY);
  const drawn: {
    ref: SplitObjectRef;
    ids: number[];
    tileset: Cesium3DTileset;
    offs: (() => void)[];
    key: string;
  }[] = [];
  let disposed = false;
  const poseFor = (ref: SplitObjectRef): ObjectPose =>
    useSceneObjects.getState().poses[assetId]?.[ref.instance] ?? ref.pose;
  const sync = (): void => {
    const shown = parent.show;
    for (const object of drawn) {
      const pose = poseFor(object.ref);
      const key = `${Matrix4.toArray(parent.modelMatrix).join()}|${pose.translation.join()}|${pose.rotation.join()}`;
      if (key !== object.key) {
        object.key = key;
        object.tileset.modelMatrix = objectModelMatrix(
          parent.modelMatrix,
          sceneRoot,
          object.ref,
          pose,
        );
        setInstanceOffset(
          assetId,
          object.ids,
          isRest(pose) ? undefined : Matrix4.fromArray(poseLocalMatrix(object.ref.origin, pose)),
        );
      }
      if (object.tileset.show !== shown) object.tileset.show = shown;
    }
  };
  const offUpdate = scene.preUpdate.addEventListener(sync);
  const offStore = useSceneObjects.subscribe((state, previous) => {
    if (state.poses[assetId] !== previous.poses[assetId]) scene.requestRender();
  });
  void Promise.allSettled(
    refs.map(async (ref) => {
      const tileset = await load(resolveLayerUrl(url, ref.uri), parent);
      if (disposed) {
        tileset.destroy();
        return;
      }
      scene.primitives.add(tileset);
      drawn.push({
        ref,
        ids: idsOf(tileset, ref),
        tileset,
        offs: [attachViewCones(tileset), follow(tileset, assetId)],
        key: "",
      });
      useSceneObjects.getState().setObjects(
        assetId,
        drawn.map((d) => d.ref),
      );
      sync();
      scene.requestRender();
    }),
  ).then((results) => {
    for (const result of results) {
      if (result.status === "rejected")
        log.warn("split object did not load", { asset: assetId, error: String(result.reason) });
    }
  });
  return () => {
    disposed = true;
    offUpdate();
    offStore();
    for (const { tileset, offs, ids } of drawn) {
      for (const off of offs) off();
      setInstanceOffset(assetId, ids, undefined);
      scene.primitives.remove(tileset);
    }
    drawn.length = 0;
    useSceneObjects.getState().setObjects(assetId, []);
  };
}
