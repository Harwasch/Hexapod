import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { create } from "zustand";

import type { components, LandArea } from "@twin/contracts";
import { footprintContains } from "@twin/geo";

import { api, unwrap } from "@/api/client";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { useSelection } from "@/state/selection";

import { useLandAnalysis } from "../../useLandAnalysis";
import {
  bandOf,
  defaultVegetationBand,
  OVERVIEW_SPECS,
  pickRaster,
  type OverviewData,
  type Raster,
  type RasterClass,
} from "./overviewData";

type Detail = components["schemas"]["InvestigationDetail"];
type Sample = components["schemas"]["RasterSample"];

/** What the map can show for the Overview, one at a time. */
export type OverviewLayer = "elevation" | "slope" | "cover" | "vegetation";

export interface OverviewView {
  layer: OverviewLayer;
  /** A particular band (a vegetation month); the layer's usual band otherwise. */
  band?: number;
}

export interface ShownLayer {
  view: OverviewView;
  raster: Raster;
  band: number;
}

interface OverviewMapState {
  /** The chosen layer per land: absent means the default, null means nothing on the map. */
  views: Record<string, OverviewView | null>;
  opacity: number;
  setView: (landId: string, view: OverviewView | null) => void;
  setOpacity: (opacity: number) => void;
}

/** Remembers, for this session, what the Overview had on the map and how strongly. */
export const useOverviewMap = create<OverviewMapState>((set) => ({
  views: {},
  opacity: 0.8,
  setView: (landId, view) => set((state) => ({ views: { ...state.views, [landId]: view } })),
  setOpacity: (opacity) => set({ opacity }),
}));

function rasterIdOf(detail: Detail | null): string | null {
  for (const artifact of detail?.artifacts ?? [])
    if (artifact.output.kind === "raster") return artifact.output.rasterId;
  return null;
}

/**
 * The three analyses behind the Overview (terrain, land cover, greenness) and the newest saved
 * raster of each. Nothing here starts an analysis.
 */
export function useOverview(land: LandArea) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const terrain = useLandAnalysis(land, OVERVIEW_SPECS.terrain.title);
  const cover = useLandAnalysis(land, OVERVIEW_SPECS.cover.title);
  const vegetation = useLandAnalysis(land, OVERVIEW_SPECS.vegetation.title);
  const list = useQuery({
    queryKey: ["land-rasters", scope, "list", land.id],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/rasters", {
          params: { path: { land_id: land.id }, query: { limit: 50 } },
        }),
      ),
    enabled: ready,
    retry: false,
  });
  const rows = useMemo(() => (Array.isArray(list.data) ? list.data : []), [list.data]);
  const ids = {
    terrain: rasterIdOf(terrain.detail),
    cover: rasterIdOf(cover.detail),
    vegetation: rasterIdOf(vegetation.detail),
  };
  const data: OverviewData = {
    terrain: pickRaster(rows, OVERVIEW_SPECS.terrain.dataset, ids.terrain, land.revision),
    cover: pickRaster(rows, OVERVIEW_SPECS.cover.dataset, ids.cover, land.revision),
    vegetation: pickRaster(rows, OVERVIEW_SPECS.vegetation.dataset, ids.vegetation, land.revision),
  };
  // A run that just finished names a raster the list has not seen yet: fetch the list again.
  const missing = [ids.terrain, ids.cover, ids.vegetation]
    .filter((id): id is string => Boolean(id) && !rows.some((row) => row.id === id))
    .join(",");
  const listed = list.isSuccess;
  const refetch = list.refetch;
  useEffect(() => {
    if (missing && listed) void refetch();
  }, [missing, listed, refetch]);
  return {
    analyses: { terrain, cover, vegetation },
    data,
    loading:
      terrain.loading || cover.loading || vegetation.loading || (ready && list.isPending) || !ready,
    /** A run just finished and its raster is still on its way. */
    settling: Boolean(missing) && !list.isError,
    error: terrain.error ?? list.error ?? null,
    retry: () => {
      void list.refetch();
    },
  };
}

function hasSamples(raster: Raster | null, band: number): boolean {
  return Boolean(bandOf(raster, band)?.validCells);
}

function defaultView(data: OverviewData): OverviewView | null {
  if (hasSamples(data.terrain, 1)) return { layer: "elevation" };
  if (data.cover && hasSamples(data.cover, 1)) return { layer: "cover" };
  if (data.vegetation) return { layer: "vegetation" };
  return null;
}

/** The raster and band a view stands for, if this land has it. */
export function layerOf(view: OverviewView, data: OverviewData): ShownLayer | null {
  if (view.layer === "elevation" || view.layer === "slope") {
    if (!data.terrain) return null;
    return { view, raster: data.terrain, band: view.layer === "elevation" ? 1 : 2 };
  }
  if (view.layer === "cover") return data.cover ? { view, raster: data.cover, band: 1 } : null;
  if (!data.vegetation) return null;
  const band = view.band ?? defaultVegetationBand(data.vegetation);
  return bandOf(data.vegetation, band) ? { view, raster: data.vegetation, band } : null;
}

/** What the map should show: the chosen layer, the default one, or (when hidden) nothing. */
export function resolveView(
  chosen: OverviewView | null | undefined,
  data: OverviewData,
): ShownLayer | null {
  if (chosen === null) return null;
  const picked = chosen ? layerOf(chosen, data) : null;
  if (picked) return picked;
  const fallback = defaultView(data);
  return fallback ? layerOf(fallback, data) : null;
}

/**
 * Puts one raster on the map while the lens is open and takes it off when the lens closes or
 * shows something else. Returns the raster layer id when something is drawn.
 */
export function useOverviewRasterLayer(shown: ShownLayer | null): string | null {
  const raster = shown?.raster ?? null;
  const band = shown?.band ?? null;
  const definition = band == null ? null : bandOf(raster, band);
  const id = raster?.id ?? null;
  const drawable = Boolean(definition?.validCells);
  const categorical = definition?.palette === "categorical";
  const bounds = raster?.metadata.bounds.join(",") ?? "";
  const attribution = raster
    ? [...new Set(raster.metadata.sources.map((source) => source.attribution))].join(" · ")
    : "";
  const layer = useMemo(
    () =>
      id && band != null && drawable
        ? { id, band, categorical, attribution, bounds: bounds.split(",").map(Number) }
        : null,
    [id, band, drawable, categorical, attribution, bounds],
  );
  useEffect(() => {
    if (!layer) return;
    useLandContext.getState().setRaster({ ...layer, opacity: useOverviewMap.getState().opacity });
    return () => useLandContext.getState().removeRaster(layer.id);
  }, [layer]);
  return layer?.id ?? null;
}

/** Changes how strongly the shown raster covers the map, and remembers it. */
export function setOverviewOpacity(rasterId: string | null, opacity: number): void {
  useOverviewMap.getState().setOpacity(opacity);
  if (!rasterId) return;
  const current = useLandContext.getState().rasters[rasterId];
  if (current) useLandContext.getState().setRaster({ ...current, opacity });
}

export interface SpotReading {
  /** When the spot was picked ("What's here"), to tell one pick from the next. */
  at: number;
  position: [number, number];
  elevation: number | null;
  slope: number | null;
  cover: RasterClass | null;
  interpretation: string | null;
  loading: boolean;
  error: string | null;
}

function valueAt(raster: Raster, sample: Sample | undefined, band: number): number | null {
  const index = raster.metadata.bands.findIndex((item) => item.index === band);
  return index < 0 ? null : (sample?.values[index] ?? null);
}

/**
 * Reads the saved rasters at the spot picked with "What's here" on the map, when it is on this
 * land. The map menu already picks spots; this lens only listens.
 */
export function useSpotReading(land: LandArea, data: OverviewData): SpotReading | null {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const selection = useSelection((s) => s.selection);
  const point =
    selection &&
    footprintContains(land.boundary, {
      longitude: selection.longitude,
      latitude: selection.latitude,
    })
      ? selection
      : null;
  const sources = [data.terrain, data.cover].filter((item): item is Raster => Boolean(item));
  const query = useQuery({
    queryKey: [
      "land-rasters",
      scope,
      "sample",
      sources.map((item) => item.id).join(","),
      point?.longitude,
      point?.latitude,
    ],
    queryFn: () =>
      Promise.all(
        sources.map((item) =>
          unwrap(
            api.GET("/api/v1/land/rasters/{raster_id}/sample", {
              params: {
                path: { raster_id: item.id },
                query: { longitude: point?.longitude ?? 0, latitude: point?.latitude ?? 0 },
              },
            }),
          ),
        ),
      ),
    enabled: ready && Boolean(point) && sources.length > 0,
    retry: false,
    staleTime: Infinity,
  });
  if (!point || !sources.length) return null;
  const samples = query.data;
  const sampleOf = (raster: Raster | null) =>
    raster ? samples?.[sources.indexOf(raster)] : undefined;
  const terrainSample = sampleOf(data.terrain);
  const coverSample = sampleOf(data.cover);
  const code = data.cover ? valueAt(data.cover, coverSample, 1) : null;
  return {
    at: point.at,
    position: [point.longitude, point.latitude],
    elevation: data.terrain ? valueAt(data.terrain, terrainSample, 1) : null,
    slope: data.terrain ? valueAt(data.terrain, terrainSample, 2) : null,
    cover:
      code == null
        ? null
        : (bandOf(data.cover, 1)?.classes?.find((item) => item.code === code) ?? null),
    interpretation: samples?.[0]?.interpretation ?? null,
    loading: query.isFetching,
    error: query.error ? query.error.message : null,
  };
}
