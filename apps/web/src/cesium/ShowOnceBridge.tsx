import { useEffect, useRef } from "react";

import { footprintContains } from "@twin/geo";

import { cellAt, cellRing } from "@/features/knowledge/grid";
import { NO_SCORE, rampImage, rampThreshold } from "@/features/showOnce/grid";
import { compareAt, dropPendingTaps, protectAt } from "@/features/showOnce/session";
import { useLand } from "@/state/land";
import { useShowOnce } from "@/state/showOnce";

import { GroundGridLayer } from "./GroundGridLayer";
import { useScene } from "./SceneContext";

/**
 * Show once, everywhere on the map (`GroundGridLayer`, material "ramp", owner "signature"),
 * for the selected Land while its card is on screen (`features/showOnce/ShowOnceCard.tsx`).
 *
 * A session takes the ground taps: inside the Land a plain click (or a zone's fill) compares
 * that place, or in Protect mode protects the lit patch there; elsewhere taps go on as before.
 * The answer drapes as lit cells above the slider's threshold, which is the drape's uniform:
 * sliding renders a frame and uploads nothing. Protecting swaps the texture on the same
 * rectangle. The tapped cell is outlined. Escape ends the session, as does the fog taking the
 * ground (one drape at a time, `state/gridDrape.ts`). Hiding the card only lifts the drape and
 * the taps; its answer waits for the card to come back.
 */
/** The map's cursor while it takes taps (the area editor's pick uses the same crosshair). */
function setCursor(canvas: HTMLCanvasElement, cursor: string): void {
  canvas.style.cursor = cursor;
}

export function ShowOnceBridge() {
  const host = useScene();
  const land = useLand((s) => s.active);
  const session = useShowOnce((s) => s.session);
  const shown = useShowOnce((s) => s.shown);
  const result = useShowOnce((s) => s.result);
  const guarded = useShowOnce((s) => s.guarded);
  const threshold = useShowOnce((s) => s.threshold);
  const layerRef = useRef<GroundGridLayer | null>(null);
  const active = Boolean(host && session && shown && land?.id === session.landId);

  useEffect(() => {
    if (!host) return;
    const layer = new GroundGridLayer(host.viewer, "ramp", "signature", () => {
      // The fog took the ground: the session ends rather than wait under it.
      dropPendingTaps();
      useShowOnce.getState().stop();
    });
    layerRef.current = layer;
    return () => {
      layer.destroy();
      layerRef.current = null;
    };
  }, [host]);

  // A session belongs to one Land on one boundary.
  useEffect(() => {
    const current = useShowOnce.getState().session;
    if (current && (current.landId !== land?.id || current.revision !== land.revision)) {
      dropPendingTaps();
      useShowOnce.getState().stop();
    }
  }, [land?.id, land?.revision]);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    if (!active || !result) {
      layer.hide();
      return;
    }
    const { grid } = result;
    layer.show(grid, rampImage(grid, guarded));
    layer.setUniform("threshold", rampThreshold(useShowOnce.getState().threshold));
    const seed = cellAt(grid, result.read.seed.longitude, result.read.seed.latitude);
    layer.setOutline(seed ? cellRing(grid, seed.col, seed.row) : null);
  }, [active, result, guarded]);

  useEffect(() => {
    layerRef.current?.setUniform("threshold", rampThreshold(threshold));
  }, [threshold]);

  useEffect(() => {
    if (!host || !active || !land) return;
    const boundary = land.boundary;
    // On the Land: a cell of the answer when there is one, else the boundary itself.
    const inside = (longitude: number, latitude: number) => {
      const grid = useShowOnce.getState().result?.grid;
      if (grid) {
        const at = cellAt(grid, longitude, latitude);
        if (at && grid.scores[at.row * grid.columns + at.col] !== NO_SCORE) return true;
      }
      return footprintContains(boundary, { longitude, latitude });
    };
    const release = host.selection.wantGroundTaps(inside);
    const off = host.events.on("ground-tap", (tap) => {
      const store = useShowOnce.getState();
      if (!inside(tap.longitude, tap.latitude)) {
        store.setMessage(`That spot is outside ${land.name}. Choose a place inside its boundary.`);
        return;
      }
      if (store.mode === "protect") protectAt(tap.longitude, tap.latitude);
      else void compareAt(tap.longitude, tap.latitude);
    });
    const canvas = host.viewer.canvas;
    setCursor(canvas, "crosshair");
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      // Consumed before the shell takes Escape as closing the panel.
      event.preventDefault();
      event.stopPropagation();
      dropPendingTaps();
      useShowOnce.getState().stop();
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      off();
      release();
      window.removeEventListener("keydown", onKey, true);
      if (canvas.style.cursor === "crosshair") setCursor(canvas, "");
    };
  }, [host, active, land]);

  return null;
}
