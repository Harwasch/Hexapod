import type { Attribution, Footprint, Representation } from "@twin/contracts";

import type { AssetRuntime } from "@/state/sites";
import type { LayerRuntime, LoadState } from "@/state/layers";
import type { Measurement } from "@/state/measurements";
import type { Selection } from "@/state/selection";
import type { ToastTone } from "@/state/toasts";
import type { CameraPose, PerformanceSnapshot, TokenState, ViewerStatus } from "@/state/viewer";

export type { LoadState };

/** Events the Cesium subsystem raises; a bridge component mirrors them into React stores. */
export interface SceneEvents extends Record<string, unknown> {
  status: { status: ViewerStatus; message?: string };
  token: TokenState;
  camera: CameraPose;
  /** True while the camera moves; the UI drops expensive blur effects for the duration. */
  motion: boolean;
  performance: Partial<PerformanceSnapshot>;
  layer: { id: string; patch: Partial<LayerRuntime> };
  asset: { id: string; patch: Partial<AssetRuntime> };
  "site-near": string | null;
  "site-active": string | null;
  representation: { siteId: string; representation: Representation };
  selection: Selection | null;
  hover: string | null;
  measurement: Measurement;
  "measurement-mode": null;
  toast: { tone: ToastTone; title: string; body?: string; id?: string; sticky?: boolean };
  world: string;
  tilesets: string[];
  explore: boolean;
  "mission-select": { kind: "zone" | "machine"; id: string };
  /** The operator reshaped an area on the map; the store takes the new outline. */
  "area-edit": { zoneId: string; footprint: Footprint };
  /** True while a handle is held (the camera is paused for the duration). */
  "area-edit-drag": boolean;
  /** Esc while editing an area. */
  "area-edit-end": string;
  "area-candidate-hover": string | null;
  /** True while the next map click is being waited for as the ground of a plan. */
  "ground-pick-mode": boolean;
  "area-candidate-pick": string;
}

export interface GeocodeResult {
  label: string;
  /** Bounding box in degrees, or a point. */
  destination:
    | { kind: "rectangle"; west: number; south: number; east: number; north: number }
    | { kind: "point"; longitude: number; latitude: number };
  attribution?: string;
}

export interface Geocoder {
  readonly name: string;
  readonly attribution: string;
  search(query: string, signal?: AbortSignal): Promise<GeocodeResult[]>;
}

export interface CreditInfo {
  attribution: Attribution[];
}
