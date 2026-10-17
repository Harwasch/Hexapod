import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { queryKeys } from "@/api/queries";

import { useKnowledgeGrid } from "@/features/knowledge/api";
import {
  cellAt,
  cellRing,
  decodeGrid,
  densityAt,
  densityImage,
  fogLifted,
  seedOf,
  type DecodedGrid,
} from "@/features/knowledge/grid";
import { useKnowledge } from "@/state/knowledge";
import { useLand } from "@/state/land";
import { useSettings } from "@/state/settings";

import { GroundGridLayer } from "./GroundGridLayer";
import { useScene } from "./SceneContext";

/**
 * The Fog of Knowing on the map (`GroundGridLayer`, material "fog"), for the selected Land.
 *
 * While it shows, a plain click on the ground inside the Land opens that cell's card (the
 * selection asks for 'ground-tap' events only for as long as the fog is up; otherwise a click
 * on the ground still does nothing). New evidence arriving (an inspection saved, a raster run
 * finished: `invalidateLandDerived`; a capture registered as a site) brings a new grid; where fog went, it dissolves, and
 * nothing else moves: the fog thickening slowly with age is not animated, and nothing renders
 * at rest. Changing or leaving the Land clears the card and the fog.
 */
export function KnowledgeFogBridge() {
  const host = useScene();
  const landId = useLand((s) => s.active?.id ?? null);
  const visible = useKnowledge((s) => s.visible);
  const selected = useKnowledge((s) => s.selected);
  const pending = useKnowledge((s) => s.pending);
  const query = useKnowledgeGrid(landId, visible);
  const client = useQueryClient();
  const layerRef = useRef<GroundGridLayer | null>(null);
  const shownRef = useRef<{ landId: string; grid: DecodedGrid } | null>(null);

  useEffect(() => {
    if (!host) return;
    const layer = new GroundGridLayer(host.viewer, "fog", "fog", () => {
      shownRef.current = null;
      useKnowledge.getState().setVisible(false);
    });
    layerRef.current = layer;
    return () => {
      layer.destroy();
      layerRef.current = null;
      shownRef.current = null;
    };
  }, [host]);

  // A capture registered as a site (the catalog refetched) may cover this land: a new
  // centimetre record, so the fog is asked again.
  useEffect(() => {
    if (!landId) return;
    let seen = client.getQueryState(queryKeys.sites)?.dataUpdatedAt ?? 0;
    return client.getQueryCache().subscribe((event) => {
      const { queryKey: key, state } = event.query as {
        queryKey: readonly unknown[];
        state: { dataUpdatedAt: number };
      };
      if (event.type !== "updated" || key[0] !== queryKeys.sites[0] || key.length !== 1) return;
      if (state.dataUpdatedAt <= seen) return;
      seen = state.dataUpdatedAt;
      void client.invalidateQueries({ queryKey: ["land-knowledge"] });
    });
  }, [client, landId]);

  // A card belongs to one Land.
  useEffect(() => {
    const { selected: open, pending: asked } = useKnowledge.getState();
    if ((open && open.landId !== landId) || (asked && asked.landId !== landId))
      useKnowledge.getState().clear();
  }, [landId]);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const data = query.data;
    if (!visible || !landId || data?.landId !== landId) {
      layer.hide();
      shownRef.current = null;
      return;
    }
    const grid = decodeGrid(data);
    const previous = shownRef.current?.landId === landId ? shownRef.current.grid : null;
    const image = densityImage(grid);
    if (layer.shown && fogLifted(previous, grid)) {
      layer.lift(image, useSettings.getState().reducedMotion);
    } else {
      layer.show(grid, image, seedOf(landId));
    }
    shownRef.current = { landId, grid };
  }, [query.data, visible, landId, host]);

  // A point asked about from the map menu becomes a cell once the grid is in.
  useEffect(() => {
    const shown = shownRef.current;
    if (!pending || pending.landId !== shown?.landId) return;
    const at = cellAt(shown.grid, pending.longitude, pending.latitude);
    if (!at || densityAt(shown.grid, at.col, at.row) === null) {
      useKnowledge.getState().clear();
      return;
    }
    useKnowledge.getState().select({ ...pending, ...at });
  }, [pending, query.data]);

  useEffect(() => {
    if (!host || !visible || !landId) return;
    const release = host.selection.wantGroundTaps();
    const off = host.events.on("ground-tap", (tap) => {
      const shown = shownRef.current;
      if (shown?.landId !== landId) return;
      const at = cellAt(shown.grid, tap.longitude, tap.latitude);
      if (!at || densityAt(shown.grid, at.col, at.row) === null) {
        useKnowledge.getState().clear();
        return;
      }
      useKnowledge
        .getState()
        .select({ landId, longitude: tap.longitude, latitude: tap.latitude, ...at });
    });
    return () => {
      off();
      release();
    };
  }, [host, visible, landId]);

  useEffect(() => {
    const layer = layerRef.current;
    const shown = shownRef.current;
    if (!layer) return;
    if (!selected || selected.landId !== shown?.landId || !layer.shown) {
      layer.setOutline(null);
      return;
    }
    layer.setOutline(cellRing(shown.grid, selected.col, selected.row));
  }, [selected, query.data, visible]);

  return null;
}
