import type { CesiumSceneManager } from "./CesiumSceneManager";

/** Frames always asked for, then more while entity geometry is still being built. */
const MIN_FRAMES = 10;
const MAX_FRAMES = 300;

/**
 * Keeps asking for frames until the entities just changed are on screen. With rendering on
 * demand, Cesium builds clamped polygons and lines over several frames and asks for none of
 * them, so a new or removed outline would otherwise wait for the next camera move. Returns a
 * function that stops the loop; starting a new loop through the same `repaint` replaces it.
 */
export function repaintUntilDrawn(host: CesiumSceneManager): () => void {
  let frames = 0;
  let handle = 0;
  const step = () => {
    if (host.isDestroyed) return;
    host.scene.requestRender();
    frames += 1;
    if (frames < MAX_FRAMES && (frames < MIN_FRAMES || !host.viewer.dataSourceDisplay.ready))
      handle = requestAnimationFrame(step);
  };
  step();
  return () => cancelAnimationFrame(handle);
}
