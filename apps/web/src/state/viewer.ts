import { create } from "zustand";

import type { ScaleBand } from "@twin/geo";

export type ViewerStatus = "idle" | "initializing" | "ready" | "error" | "context-lost";
export type TokenState = "unknown" | "custom" | "default" | "invalid";

export interface CameraPose {
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
  roll: number;
  /** Height above the terrain/surface under the camera when known. */
  altitude: number;
  scaleBand: ScaleBand;
  metersPerPixel: number;
}

export interface PerformanceSnapshot {
  fps: number;
  frameTimeMs: number;
  resolutionScale: number;
  devicePixelRatio: number;
  pendingRequests: number;
  tilesProcessing: number;
  siteScreenSpaceError: number | null;
  worldScreenSpaceError: number | null;
  adaptiveReason: string;
  moving: boolean;
  gpu: string | null;
  webgl2: boolean;
  tilesetMemoryMb: number;
  memoryBudgetMb: number;
  /** False while nothing is being rendered (request-render mode idles the GPU). */
  rendering: boolean;
  msaaSamples: number;
  /** "full" renders the preset as configured; "reduced" means the adaptive ladder cut something. */
  profile: "full" | "reduced";
  /** Motion-only frame statistics since the last site change, for comparing datasets. */
  benchmark: { motionFps: number | null; p95FrameMs: number | null; samples: number };
  /** Median main-thread time of a moving frame's update and render phases, and draw calls. */
  frameBudget: {
    updateMs: number | null;
    renderMs: number | null;
    commands: number | null;
    loadingUpdateMs: number | null;
    loadingFrames: number;
    steadyFrames: number;
  };
}

interface ViewerState {
  status: ViewerStatus;
  errorMessage: string | null;
  tokenState: TokenState;
  camera: CameraPose;
  performance: PerformanceSnapshot;
  worldLabel: string;
  activeTilesets: string[];
  setStatus: (status: ViewerStatus, errorMessage?: string | null) => void;
  setTokenState: (tokenState: TokenState) => void;
  setCamera: (camera: CameraPose) => void;
  setPerformance: (perf: Partial<PerformanceSnapshot>) => void;
  setWorldLabel: (worldLabel: string) => void;
  setActiveTilesets: (activeTilesets: string[]) => void;
}

export const initialCamera: CameraPose = {
  longitude: -110,
  latitude: 35,
  height: 18_000_000,
  heading: 0,
  pitch: -90,
  roll: 0,
  altitude: 18_000_000,
  scaleBand: "planet",
  metersPerPixel: Number.NaN,
};

export const useViewer = create<ViewerState>()((set) => ({
  status: "idle",
  errorMessage: null,
  tokenState: "unknown",
  camera: initialCamera,
  performance: {
    fps: 0,
    frameTimeMs: 0,
    resolutionScale: 1,
    devicePixelRatio: typeof window === "undefined" ? 1 : window.devicePixelRatio,
    pendingRequests: 0,
    tilesProcessing: 0,
    siteScreenSpaceError: null,
    worldScreenSpaceError: null,
    adaptiveReason: "idle",
    moving: false,
    gpu: null,
    webgl2: false,
    tilesetMemoryMb: 0,
    memoryBudgetMb: 0,
    rendering: false,
    msaaSamples: 4,
    profile: "full",
    benchmark: { motionFps: null, p95FrameMs: null, samples: 0 },
    frameBudget: {
      updateMs: null,
      renderMs: null,
      commands: null,
      loadingUpdateMs: null,
      loadingFrames: 0,
      steadyFrames: 0,
    },
  },
  worldLabel: "Open world",
  activeTilesets: [],
  setStatus: (status, errorMessage = null) => set({ status, errorMessage }),
  setTokenState: (tokenState) => set({ tokenState }),
  setCamera: (camera) => set({ camera }),
  setPerformance: (perf) => set((s) => ({ performance: { ...s.performance, ...perf } })),
  setWorldLabel: (worldLabel) => set({ worldLabel }),
  setActiveTilesets: (activeTilesets) => set({ activeTilesets }),
}));
