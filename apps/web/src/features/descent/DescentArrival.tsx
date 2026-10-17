import { History, LandPlot, X } from "lucide-react";
import { useEffect } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { env } from "@/app/env";
import { useScene } from "@/cesium/SceneContext";
import { useDescent } from "@/state/descent";
import { useLantern } from "@/state/lantern";

import { drawOutline, proposeOutline } from "../land/landEntry";
import { mappedFeatureDraft } from "../land/mappedFeature";
import { lookBackArea, removeMappedPreview, stopDescent } from "./descent";
import "./descent.css";

const acres = (value: number) =>
  value.toLocaleString(undefined, { maximumFractionDigits: value < 10 ? 1 : 0 });

/** What was found around the place, in one line. "You can draw one" only where one can. */
function mappedLine(state: ReturnType<typeof useDescent.getState>, canDraw: boolean): string {
  const draw = canDraw ? " You can draw one." : "";
  switch (state.mappedStatus) {
    case "found":
      return state.mapped
        ? `Mapped area: ${state.mapped.name} · ${acres(state.mapped.acres)} acres, from OpenStreetMap. Not a legal parcel.`
        : `No mapped boundary here.${draw}`;
    case "pending":
      return "Looking for a mapped boundary…";
    case "error":
      return `OpenStreetMap could not be reached, so no mapped boundary is shown.${draw}`;
    case "skipped":
      return canDraw
        ? "This is a large area. Zoom in on your land to outline it."
        : "This is a large area. Zoom in to see the ground.";
    case "none":
      return `No mapped boundary here.${draw}`;
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
    const lantern = useLantern.getState();
    lantern.setBbox(lookBackArea(target));
    lantern.setMode("sticky");
    // The lantern's strip takes over; the card would cover the lens.
    removeMappedPreview();
    useDescent.getState().clear();
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
        {mappedLine(state, env.landExplorationEnabled)}
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
