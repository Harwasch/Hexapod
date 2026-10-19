import { useEffect } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { create } from "zustand";

import type { Footprint, LandArea } from "@twin/contracts";
import { boundsOf } from "@twin/geo";

import { ApiError, api, unwrap } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { describeError } from "@/lib/log";
import { fetchOsmContaining, OSM_ATTRIBUTION } from "@/missions/osm";
import { useLand, type LandMode } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";

import { createCorridor, createLand, reviseLand } from "../api";
import { drawingReferences } from "../boundarySources";

/**
 * The steps of outlining and saving land, shared by the cards in the left dock and the draw
 * bar in the strip. One ticket counts operations, so a stale save or corridor build that
 * resolves after the person moved on never overwrites what they are doing now.
 */
let operation = 0;

interface FlowState {
  busy: boolean;
  setBusy: (busy: boolean) => void;
}

export const useLandFlow = create<FlowState>((set) => ({
  busy: false,
  setBusy: (busy) => set({ busy }),
}));

export function landError(error: unknown): string {
  return error instanceof ApiError && error.fieldErrors.length
    ? error.fieldErrors.join(". ")
    : describeError(error);
}

export function areaLabel(squareMetres: number): string {
  if (squareMetres < 4046.8564224) return `${Math.round(squareMetres).toLocaleString()} m²`;
  const acres = squareMetres / 4046.8564224;
  const shown = acres.toLocaleString(undefined, { maximumFractionDigits: acres < 10 ? 2 : 1 });
  return `${shown} ${shown === "1" ? "acre" : "acres"}`;
}

/** Frames a boundary with room on the left for the land cards. */
export function frameBoundary(scene: CesiumSceneManager | null, boundary: Footprint): void {
  const b = boundsOf(boundary);
  const dx = Math.max(0.0001, b.east - b.west);
  const dy = Math.max(0.0001, b.north - b.south);
  scene?.camera.flyToRectangle(
    Math.max(-180, b.west - dx * 0.9),
    Math.max(-90, b.south - dy * 0.4),
    Math.min(180, b.east + dx * 0.45),
    Math.min(90, b.north + dy * 0.35),
  );
}

/** Starts an outline (or returns to browsing), clearing whatever else held the pointer. */
export function beginLand(scene: CesiumSceneManager | null, next: LandMode): void {
  operation += 1;
  useLandFlow.getState().setBusy(false);
  const ui = useUi.getState();
  ui.setMeasureMode(null);
  ui.setExploreMode(false);
  scene?.areas.cancelPick();
  scene?.areas.edit(null);
  const land = useLand.getState();
  land.begin(next);
  if (next === "draw" || next === "corridor") {
    const guides = useLandContext.getState().layers["drawing-guides"];
    land.setTraceSources(drawingReferences(guides?.drawingGuideSources ?? []));
  }
}

/** Closes the outline being drawn into a draft (or combines it with the draft). */
export async function finishSketch(): Promise<void> {
  const ticket = ++operation;
  const state = useLand.getState();
  const flow = useLandFlow.getState();
  const { mode, points, draft, active, boundaryOperation } = state;
  state.setError(null);
  if (mode === "draw") {
    const first = points[0];
    if (points.length < 3 || !first) return;
    const boundary: Footprint = { type: "Polygon", coordinates: [[...points, first]] };
    if (boundaryOperation && draft) {
      flow.setBusy(true);
      try {
        const result = await unwrap(
          api.POST("/api/v1/land/operations", {
            body: { operation: boundaryOperation, left: draft.boundary, right: boundary },
          }),
        );
        if (operation !== ticket) return;
        const references = drawingReferences([
          ...(draft.source.references ?? []),
          ...useLand.getState().traceSources,
        ]);
        state.updateBoundary(result.boundary);
        const updated = useLand.getState().draft;
        if (updated) state.updateDraft({ source: { ...updated.source, references } });
        state.begin("browse");
        state.setBoundaryOperation(null);
      } catch (error) {
        if (operation === ticket) state.setError(landError(error));
      } finally {
        if (operation === ticket) flow.setBusy(false);
      }
      return;
    }
    state.propose({
      name: active?.name ?? "Untitled land",
      description: active?.description ?? "",
      boundary,
      source: {
        method: "drawn",
        label: "Drawn on the map",
        meaning: "study-area",
        references: state.traceSources,
      },
    });
    return;
  }
  if (mode !== "corridor") return;
  const { corridorWidth: width, corridorUnit: unit } = state;
  flow.setBusy(true);
  try {
    const result = await createCorridor(points, unit === "ft" ? width * 0.3048 : width);
    if (operation !== ticket) return;
    state.propose({
      name: active?.name ?? "Land corridor",
      description: active?.description ?? "",
      boundary: result.boundary,
      source: {
        method: "corridor",
        label: `${width} ${unit} total width, centered on the drawn line`,
        meaning: "study-area",
        references: state.traceSources,
      },
    });
  } catch (error) {
    if (operation === ticket) state.setError(landError(error));
  } finally {
    if (operation === ticket) flow.setBusy(false);
  }
}

/** Saves the draft as new land, or as a revision of the land it was drawn from. */
export async function saveDraft(queryClient: QueryClient): Promise<void> {
  const state = useLand.getState();
  const { draft, active } = state;
  if (!draft) return;
  state.begin("browse");
  const ticket = ++operation;
  const flow = useLandFlow.getState();
  flow.setBusy(true);
  state.setError(null);
  try {
    const saved = active
      ? await reviseLand(active.id, {
          ...draft,
          expectedRevision: active.revision,
          note: "Boundary updated in the land workspace",
        })
      : await createLand(draft);
    if (operation === ticket) useLand.getState().select(saved);
    await queryClient.invalidateQueries({ queryKey: ["land-areas"] });
    await queryClient.invalidateQueries({ queryKey: ["land-history", saved.id] });
  } catch (error) {
    if (operation === ticket) useLand.getState().setError(landError(error));
  } finally {
    if (operation === ticket) flow.setBusy(false);
  }
}

/** Gives up any pending build or save, so its result is ignored when it lands. */
export function abandonOperation(): void {
  operation += 1;
  useLandFlow.getState().setBusy(false);
}

export function exportBoundary(
  land: { name: string; boundary: Footprint; source: unknown },
  revision?: number,
): void {
  const feature = {
    type: "Feature",
    geometry: land.boundary,
    properties: { name: land.name, source: land.source, revision },
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(feature, null, 2)], { type: "application/geo+json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "land-boundary.geojson";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Opens saved land and frames it. */
export function openLand(scene: CesiumSceneManager | null, land: LandArea): void {
  useLand.getState().select(land);
  frameBoundary(scene, land.boundary);
}

/**
 * While picking a mapped feature, the first click asks OpenStreetMap for the area it falls in
 * and proposes it as the draft.
 */
export function useMappedFeaturePick(enabled: boolean): void {
  const mode = useLand((s) => s.mode);
  const points = useLand((s) => s.points);
  useEffect(() => {
    if (!enabled || mode !== "pick" || points.length !== 1) return;
    const point = points[0];
    if (!point) return;
    let cancelled = false;
    void fetchOsmContaining({ longitude: point[0], latitude: point[1] })
      .then((feature) => {
        if (cancelled) return;
        if (!feature) {
          useLand.getState().begin("browse");
          useLand
            .getState()
            .setError("No mapped area found there. Draw the land instead, or import a file.");
          return;
        }
        useLand.getState().propose({
          name: feature.name,
          description: "",
          boundary: feature.footprint,
          source: {
            method: "mapped-feature",
            meaning: "physical-feature",
            label: "OpenStreetMap feature",
            attribution: OSM_ATTRIBUTION,
            url: "https://www.openstreetmap.org/copyright",
          },
        });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        useLand.getState().begin("browse");
        useLand.getState().setError(`Map data could not be reached. ${describeError(error)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, mode, points]);
}
