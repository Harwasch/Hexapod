import { History, LandPlot, X } from "lucide-react";
import { useEffect } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { useDescent } from "@/state/descent";
import { useLantern, type LanternBbox } from "@/state/lantern";

import { drawOutline, proposeOutline } from "../land/landEntry";
import { mappedFeatureDraft } from "../land/mappedFeature";
import { boxAround, removeMappedPreview, stopDescent } from "./descent";
import "./descent.css";

/** "Look back in time" lists dated imagery this far around the place reached (m, each way). */
const LOOK_BACK_HALF_M = 1_200;

const acres = (value: number) =>
  value.toLocaleString(undefined, { maximumFractionDigits: value < 10 ? 1 : 0 });

/** What was found around the place, in one line. */
function mappedLine(state: ReturnType<typeof useDescent.getState>): string {
  switch (state.mappedStatus) {
    case "found":
      return state.mapped
        ? `Mapped area: ${state.mapped.name} · ${acres(state.mapped.acres)} acres, from OpenStreetMap. Not a legal parcel.`
        : "No mapped boundary here; you can draw one.";
    case "pending":
      return "Looking for a mapped boundary…";
    case "error":
      return "OpenStreetMap could not be reached, so no mapped boundary is shown. You can draw one.";
    case "skipped":
      return "This is a large area. Zoom in on your land to outline it.";
    case "none":
      return "No mapped boundary here; you can draw one.";
  }
}

/**
 * Where the Descent lands (./descent.ts): the place, what is mapped there, and the next steps --
 * outline this land (the mapped area as a draft, or drawing from scratch) and look back in time
 * (the Time Lantern over the place). Shown once the flight has arrived, or was taken over by a
 * hand on the map on the way.
 */
export function DescentArrival() {
  const scene = useScene();
  const state = useDescent();
  const lanternReady = useLantern((s) => s.availability.ok);
  // Leaving the app stops the lookups a descent still has under way.
  useEffect(() => () => stopDescent(), []);
  const { target, phase } = state;
  if (!target || (phase !== "arrived" && phase !== "cancelled")) return null;

  const close = () => {
    stopDescent();
    removeMappedPreview();
    useDescent.getState().clear();
  };
  const outline = () => {
    const mapped = useDescent.getState().mapped;
    if (mapped) proposeOutline(mappedFeatureDraft(mapped), scene);
    else drawOutline(scene);
    removeMappedPreview();
    useDescent.getState().clear();
  };
  const lookBack = () => {
    const box = boxAround(target.longitude, target.latitude, LOOK_BACK_HALF_M * 2);
    const bbox: LanternBbox = [box.west, box.south, box.east, box.north];
    const lantern = useLantern.getState();
    lantern.setBbox(bbox);
    lantern.setMode("sticky");
  };

  return (
    <GlassPanel
      strong
      padding="none"
      className="descent-arrival"
      role="region"
      aria-labelledby="descent-arrival-title"
      data-testid="descent-arrival"
    >
      <div className="descent-arrival__head">
        <h2 id="descent-arrival-title">{target.label}</h2>
        <GlassButton
          iconOnly
          variant="ghost"
          size="sm"
          aria-label="Close"
          onClick={close}
          data-testid="descent-close"
        >
          <X size={16} aria-hidden="true" />
        </GlassButton>
      </div>
      <p className="descent-arrival__mapped" aria-live="polite" data-testid="descent-mapped">
        {mappedLine(state)}
      </p>
      <p className="descent-arrival__credit">{target.attribution}</p>
      <div className="descent-arrival__actions">
        {env.landExplorationEnabled && (
          <GlassButton variant="primary" onClick={outline} data-testid="descent-outline">
            <LandPlot size={16} aria-hidden="true" />
            Outline this land
          </GlassButton>
        )}
        {lanternReady && (
          <GlassButton variant="glass" onClick={lookBack} data-testid="descent-look-back">
            <History size={16} aria-hidden="true" />
            Look back in time
          </GlassButton>
        )}
      </div>
    </GlassPanel>
  );
}
