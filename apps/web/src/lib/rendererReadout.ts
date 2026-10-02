/**
 * The splat renderer as the developer readouts say it: which engine draws the scan and with
 * which graphics API ("PlayCanvas · WebGPU", or "PlayCanvas · WebGL2 (WebGPU unavailable)"
 * when the WebGPU trial could not have it), how fast it drew during the latest camera motion,
 * and the one line on why the trial fell back. What the owner compares renderers by in person
 * (docs/WEBGPU_TRIAL.md); shown in the bottom bar and, for phones, where the bar is hidden,
 * under the renderer switch in Settings › Advanced.
 */

import { useEffect, useState } from "react";

import { useScene } from "@/cesium/SceneContext";
import type { ScanRendererStatus } from "@/cesium/scanView/ScanRendererHost";
import type { SplatRenderer } from "@/state/settings";

/** What each choice is called before a renderer draws with it. */
export const RENDERER_NAMES: Record<SplatRenderer, string> = {
  playcanvas: "PlayCanvas",
  "playcanvas-webgpu": "PlayCanvas WebGPU (beta)",
  spark: "Spark",
  cesium: "CesiumJS",
};

/** The engine alone, once it draws (the API is said beside it). */
const ENGINE: Record<SplatRenderer, string> = {
  playcanvas: "PlayCanvas",
  "playcanvas-webgpu": "PlayCanvas",
  spark: "Spark",
  cesium: "CesiumJS",
};

const API_NAMES = { webgl2: "WebGL2", webgpu: "WebGPU" } as const;

export interface RendererReadout {
  /** Engine and API: "PlayCanvas · WebGPU". */
  label: string;
  /** "58 fps · p95 21 ms · draw 2.3 ms" while moving, "last move …" after; null before any. */
  meter: string | null;
  /** Why the WebGPU trial draws with WebGL2, in one line, or null. */
  notice: string | null;
}

/** The readout for the renderer `chosen` and the overlay's status (null while none runs). */
export function rendererReadout(
  chosen: SplatRenderer,
  status: Pick<ScanRendererStatus, "kind" | "active" | "api" | "notice" | "meter"> | null,
): RendererReadout {
  if (chosen === "cesium" || !status?.active || status.kind !== chosen || status.api === null) {
    return { label: RENDERER_NAMES[chosen], meter: null, notice: null };
  }
  const fellBack = chosen === "playcanvas-webgpu" && status.api !== "webgpu";
  const label = `${ENGINE[chosen]} · ${API_NAMES[status.api]}${fellBack ? " (WebGPU unavailable)" : ""}`;
  const reading = status.meter;
  const meter = reading
    ? `${reading.live ? "" : "last move "}${String(Math.round(reading.fps))} fps · p95 ${String(Math.round(reading.p95Ms))} ms · draw ${reading.cpuMs.toFixed(1)} ms`
    : null;
  return { label, meter, notice: fellBack ? status.notice : null };
}

/** How often the readout reads the overlay's status (ms). */
const POLL_MS = 500;

/**
 * The dedicated splat renderer's status, read twice a second while `enabled`, else null. A
 * poll, not a subscription: the overlay draws on its own clock and must not wait on React.
 */
export function useScanRendererStatus(enabled: boolean): ScanRendererStatus | null {
  const scene = useScene();
  const [status, setStatus] = useState<ScanRendererStatus | null>(null);
  useEffect(() => {
    if (!scene || !enabled) {
      setStatus(null);
      return;
    }
    setStatus(scene.scanRendererStatus);
    const timer = setInterval(() => setStatus(scene.scanRendererStatus), POLL_MS);
    return () => clearInterval(timer);
  }, [scene, enabled]);
  return status;
}
