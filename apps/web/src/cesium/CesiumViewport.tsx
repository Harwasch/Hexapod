import { useContext, useEffect, useRef } from "react";

import { env } from "@/app/env";
import { createLogger, describeError } from "@/lib/log";
import { useViewer } from "@/state/viewer";

import { CesiumSceneManager } from "./CesiumSceneManager";
import { SceneContext } from "./SceneContext";

const log = createLogger("viewport");

/** Mounts the CesiumJS viewer exactly once. All other UI floats above this element. */
export function CesiumViewport() {
  const containerRef = useRef<HTMLDivElement>(null);
  const registry = useContext(SceneContext);
  const setStatus = useViewer((s) => s.setStatus);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || registry.current) return;
    setStatus("initializing");
    let manager: CesiumSceneManager | null = null;
    try {
      manager = new CesiumSceneManager(container, { ionToken: env.ionAccessToken });
      registry.set(manager);
      if (env.isDev) window.__twin = manager;
    } catch (error) {
      log.error("viewer failed to initialize", { error: describeError(error) });
      setStatus("error", describeError(error));
    }
    return () => {
      registry.set(null);
      manager?.destroy();
      if (env.isDev) window.__twin = undefined;
    };
  }, [registry, setStatus]);

  return (
    <div
      ref={containerRef}
      className="viewport"
      data-testid="cesium-viewport"
      aria-label="3D world"
      role="application"
    />
  );
}
