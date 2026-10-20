import { create } from "zustand";

import type { InferredStyle } from "@/lib/inferred";
import { useSettings } from "@/state/settings";
import { useSkinPoke } from "@/state/skinPoke";

/**
 * As captured (ADR 0019 "Living map, scoped"): the scene draws only what a capture measured --
 * no inferred layer, no fill that supersedes measured splats, no modelled motion, no simulated
 * machine. One switch, so a viewer can always see the data alone, and every effect reads it
 * here, in one place each:
 *
 * - inferred layers and split objects' hole fills (`fills/<id>/`, declared as inferred layers):
 *   `effectiveInferredStyle` is Hide, under CesiumJS (`cesium/inferredLayers.ts`) and under
 *   PlayCanvas or Spark (`cesium/scanView/scanLayers.ts`); supersedes swaps end with it
 *   (`state/supersedes.ts`); the per-cell view rules own no inferred cell (`compositorSession.ts`);
 * - motion: the wind calm (`SceneBridge.tsx`), the poke tool off and given back (below),
 *   telemetry and split objects held at their measured rest pose (`cesium/telemetry.ts`,
 *   `cesium/splitObjects.ts`, `cesium/scanView/scanObjects.ts`);
 * - agents: a rehearsal's simulated machines hidden (`cesium/RehearsalManager.ts`); the Land
 *   agent's pins, outlines and research maps (`agentMade`, `cesium/LandContextBridge.tsx`,
 *   `LandOverlays.tsx`, `cesium/LandFocusBridge.tsx`);
 * - Show once's inferred heatmap lifted (`cesium/ShowOnceBridge.tsx`);
 * - procedural fixtures, the synthetic tree, withheld (`SiteManager.hideFixtures`);
 * - the per-cell view rule drawn as Ground truth (`compositorSession.ts`);
 * - the sunlight relight (W6b) is to subscribe here and draw nothing while it is on.
 *
 * None of them writes a viewer's own choice -- the Inferred style, the wind, the view rule stay
 * as set, so turning it off gives back exactly what was drawn before. The Simulated motion
 * badge is not told: it goes because nothing animates, which is the check that this works.
 *
 * Kept only in the page address: `?asCaptured=1` turns it on, and the switch keeps the address
 * in step (replaced, not a new history entry), so a reload or a copied link shows what is on
 * screen. Nothing is stored on the device: a later visit without the address has it off.
 */
interface AsCapturedState {
  on: boolean;
  set: (on: boolean) => void;
}

function fromLocation(): boolean {
  try {
    return new URLSearchParams(window.location.search).get("asCaptured") === "1";
  } catch {
    return false;
  }
}

function toLocation(on: boolean): void {
  try {
    const url = new URL(window.location.href);
    if (on) url.searchParams.set("asCaptured", "1");
    else url.searchParams.delete("asCaptured");
    if (url.href !== window.location.href)
      window.history.replaceState(window.history.state, "", url);
  } catch {
    // No address to keep it in.
  }
}

export const useAsCaptured = create<AsCapturedState>()((set) => ({
  on: fromLocation(),
  set: (on) => {
    set({ on });
    toLocation(on);
  },
}));

/** What a control of something As captured hides says while it is on. */
export const HIDDEN_AS_CAPTURED = "Hidden while As captured is on";

/** The Inferred style drawn now: the viewer's, or Hide while As captured is on. */
export function effectiveInferredStyle(): InferredStyle {
  return useAsCaptured.getState().on ? "hide" : useSettings.getState().inferredStyle;
}

/** `effectiveInferredStyle`, for a component. */
export function useInferredStyle(): InferredStyle {
  const on = useAsCaptured((s) => s.on);
  const style = useSettings((s) => s.inferredStyle);
  return on ? "hide" : style;
}

/**
 * Calls `listener` with the drawn Inferred style each time it changes (the viewer's setting,
 * or As captured); returns the unsubscriber.
 */
export function onInferredStyle(listener: (style: InferredStyle) => void): () => void {
  let last = effectiveInferredStyle();
  const check = (): void => {
    const next = effectiveInferredStyle();
    if (next !== last) listener((last = next));
  };
  const offSettings = useSettings.subscribe(check);
  const offOn = useAsCaptured.subscribe(check);
  return () => {
    offSettings();
    offOn();
  };
}

// The poke tool rings an object under a modelled material: off while As captured is on, and
// turned straight off again if asked meanwhile (`K`, the Settings switch). It is the viewer's
// choice, so it is given back once As captured is off: on if it was on, or was asked for.
let pokeWanted = false;
const pokeOff = (): void => {
  if (useAsCaptured.getState().on && useSkinPoke.getState().active) {
    pokeWanted = true;
    useSkinPoke.getState().setActive(false);
  }
};
useAsCaptured.subscribe(({ on }, prev) => {
  if (on === prev.on) return;
  if (!on && pokeWanted) useSkinPoke.getState().setActive(true);
  pokeWanted = false;
  pokeOff();
});
useSkinPoke.subscribe(pokeOff);
