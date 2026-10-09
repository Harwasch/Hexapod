/** Local library records. No inference credentials belong in exported records. */
export type ProviderId =
  "runpod" | "local" | "modal" | "lambda" | "coreweave" | "aws" | "gcp" | "azure";
export type PerformanceMode = "quality" | "balanced" | "low-latency";
export interface WorldSettings {
  seed?: number;
  performance: PerformanceMode;
  resolution?: string;
}
export interface WorldProject {
  id: string;
  name: string;
  prompt: string;
  modelId: string;
  providerId: ProviderId;
  createdAt: number;
  updatedAt: number;
  assetIds: string[];
  characterIds: string[];
  settings: WorldSettings;
  templateId?: string;
  parentSceneId?: string;
  thumbnailAssetId?: string;
  game?: {
    objective: string;
    events: { atSeconds: number; prompt: string }[];
  };
  referencePreparation?: {
    method: "last-video-frame" | "image-synthesis";
    sourceAssetIds: string[];
  };
  /** Explicit interaction preview only; never represents generated model output. */
  previewOnly?: boolean;
}
export type World = WorldProject;
export type ResumeKind = "exact" | "approximate" | "visual";
export interface WorldScene {
  id: string;
  name: string;
  projectId: string;
  createdAt: number;
  modelId: string;
  prompt: string;
  resumeKind: ResumeKind;
  thumbnailAssetId?: string;
  assetIds: string[];
  events: ControlEvent[];
  seed?: number;
  snapshot?: Record<string, unknown>;
}
export type Scene = WorldScene;
export interface Character {
  id: string;
  name: string;
  description: string;
  assetIds: string[];
  createdAt: number;
  updatedAt: number;
  supportedModels?: string[];
  identityMethod?: "reference-images" | "prompt" | "lora" | "unverified";
}
export interface Replay {
  id: string;
  name: string;
  projectId: string;
  assetId: string;
  createdAt: number;
  durationMs: number;
  modelId: string;
  events: ControlEvent[];
  previewOnly?: boolean;
}
export interface MediaAsset {
  id: string;
  name: string;
  kind: "image" | "video" | "audio" | "snapshot" | "reconstruction";
  mimeType: string;
  size: number;
  createdAt: number;
}
export interface BenchmarkResult {
  id: string;
  name: string;
  projectId: string;
  modelId: string;
  providerId: ProviderId;
  createdAt: number;
  durationMs: number;
  frameCount: number;
  measuredFPS?: number;
  latencyMs?: number;
  estimatedCostUSD?: number;
  gpuName?: string;
  ratings?: Partial<
    Record<
      | "visualFidelity"
      | "actionAdherence"
      | "temporalStability"
      | "spatialConsistency"
      | "characterConsistency",
      number
    >
  >;
  notes?: string;
  events: ControlEvent[];
}
export interface Reconstruction {
  id: string;
  name: string;
  projectId: string;
  createdAt: number;
  status: "queued" | "running" | "completed" | "failed";
  format: "ply" | "spz" | "glb";
  assetId?: string;
  jobId?: string;
  error?: string;
  sourceAssetIds: string[];
  sourceReplayId?: string;
  progress?: number;
  stage?: string;
  artifacts?: { format: "ply" | "spz" | "glb" | "gltf" | "point-cloud"; url: string }[];
}
export interface ModelCapabilities {
  input: {
    text: boolean;
    image: boolean;
    multiImage: boolean;
    video: boolean;
    audio: boolean;
    requiredImage?: boolean;
  };
  control: {
    wasd: boolean;
    mouseLook: boolean;
    camera6DoF: boolean;
    gamepad: boolean;
    discreteActions: boolean;
    continuousActions: boolean;
    semanticActions: boolean;
    promptDuringRollout: boolean;
    promptSwitching: boolean;
    timedEvents: boolean;
    characterReference: boolean;
  };
  output: { video: boolean; audio: boolean; depth: boolean; cameraPose: boolean };
  persistence: { nativeMemory: boolean; snapshotRestore: boolean; deterministicSeed: boolean };
  runtime: {
    resolutionOptions: string[];
    realtime: boolean;
    expectedFPS?: number;
    expectedVRAMGB?: number;
    qualityOptions?: PerformanceMode[];
    interactionMode?: string;
    controlMode?: "native" | "prompt-adapted";
  };
  customization: { lora: boolean; fineTune: boolean; adapters: boolean };
}
export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  family?: string;
  repository?: string;
  license: string;
  status: "adapter-ready" | "research" | "preview";
  capabilities: ModelCapabilities;
  nativeActions: string[];
  evidence: string[];
  caveat: string;
}
export interface ProviderInfo {
  id: ProviderId;
  name: string;
  configured: boolean;
  canProvision: boolean;
  gpuTypeId?: string | null;
  message: string;
}
export interface ActionBinding {
  id: string;
  key: string;
  label: string;
  type: "native" | "prompt" | "semantic";
  action?: string;
  prompt?: string;
  experimental?: boolean;
  gamepadButton?: number;
}
export interface ControlEvent {
  id: string;
  timestampMs: number;
  type: "native" | "prompt" | "semantic" | "pause" | "resume";
  action?: string;
  prompt?: string;
  values?: Record<string, number | string | boolean>;
}
export interface LibraryRecords {
  projects: WorldProject;
  scenes: WorldScene;
  characters: Character;
  replays: Replay;
  assets: MediaAsset;
  benchmarks: BenchmarkResult;
  reconstructions: Reconstruction;
}
export type LibraryStore = keyof LibraryRecords;
export interface AppSettings {
  apiBaseUrl: string;
  defaultProvider: ProviderId;
  idleTimeoutMinutes: number;
  retainWorker: boolean;
}
export const DEFAULT_SETTINGS: AppSettings = {
  apiBaseUrl: "",
  defaultProvider: "runpod",
  idleTimeoutMinutes: 10,
  retainWorker: false,
};
export function newId(): string {
  return crypto.randomUUID();
}
