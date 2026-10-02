import { create } from "zustand";

import type { ObjectPose, SplitObjectRef } from "@/lib/sceneObjects";

/** Split objects (lib/sceneObjects.ts) per asset, and where each is drawn now. */
interface SceneObjectsState {
  /** Per asset id: the objects its scan declares, once their tilesets loaded. */
  objects: Record<string, SplitObjectRef[]>;
  /**
   * Per asset id, per instance id: a pose set at runtime (a driver, a person dragging it).
   * Absent: the pose the scan declares (`SplitObjectRef.pose`, the rest pose by default).
   */
  poses: Record<string, Record<number, ObjectPose>>;
  setObjects: (assetId: string, objects: SplitObjectRef[]) => void;
  /** Moves instance `instance` of `assetId`'s scan; `null` puts it back to its declared pose. */
  setPose: (assetId: string, instance: number, pose: ObjectPose | null) => void;
}

export const useSceneObjects = create<SceneObjectsState>()((set) => ({
  objects: {},
  poses: {},
  setObjects: (assetId, objects) =>
    set((s) => {
      const next: Record<string, SplitObjectRef[]> = {};
      for (const [id, list] of Object.entries(s.objects)) if (id !== assetId) next[id] = list;
      if (objects.length > 0) next[assetId] = objects;
      return { objects: next };
    }),
  setPose: (assetId, instance, pose) =>
    set((s) => {
      const others = Object.entries(s.poses[assetId] ?? {}).filter(
        ([id]) => Number(id) !== instance,
      );
      const current: Record<number, ObjectPose> = Object.fromEntries(others);
      if (pose) current[instance] = pose;
      return { poses: { ...s.poses, [assetId]: current } };
    }),
}));
