import { createContext, useContext, useSyncExternalStore } from "react";

import type { CesiumSceneManager } from "./CesiumSceneManager";

/** Holds the one scene manager; components read it imperatively, never re-rendering Cesium. */
export class SceneRegistry {
  private manager: CesiumSceneManager | null = null;
  private readonly listeners = new Set<() => void>();

  get current(): CesiumSceneManager | null {
    return this.manager;
  }

  set(manager: CesiumSceneManager | null): void {
    this.manager = manager;
    for (const listener of this.listeners) listener();
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
}

export const sceneRegistry = new SceneRegistry();
export const SceneContext = createContext<SceneRegistry>(sceneRegistry);

/** The live scene manager or null before the viewer exists. Re-renders only when it changes. */
export function useScene(): CesiumSceneManager | null {
  const registry = useContext(SceneContext);
  return useSyncExternalStore(
    registry.subscribe,
    () => registry.current,
    () => null,
  );
}
