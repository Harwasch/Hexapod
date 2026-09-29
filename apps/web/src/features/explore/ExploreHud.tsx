import { useEffect, useState } from "react";
import { Footprints, Plane, X } from "lucide-react";

import { formatLength } from "@twin/geo";
import { GlassButton, GlassPanel, Kbd } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/**
 * First-person explore mode: walking or flying, the keys that matter, the switch between
 * them, and the way out.
 *
 * The full key list is one hover away rather than a sentence across the bottom of the screen.
 */
export function ExploreHud() {
  const exploreMode = useUi((s) => s.exploreMode);
  const setExploreMode = useUi((s) => s.setExploreMode);
  const units = useSettings((s) => s.units);
  const scene = useScene();
  // The controller announces every change of mode or mouse capture as an "explore" event.
  const [, refresh] = useState(0);
  useEffect(() => scene?.events.on("explore", () => refresh((n) => n + 1)), [scene]);
  if (!exploreMode) return null;
  const explore = scene?.explore;
  const status = explore?.status;
  const flying = status?.mode === "fly";
  const speed = explore?.currentSpeed ?? 0;
  return (
    <GlassPanel strong pill className="explore-hud" role="status" data-testid="explore-hud">
      {flying ? (
        <Plane size={15} aria-hidden="true" />
      ) : (
        <Footprints size={15} aria-hidden="true" />
      )}
      <strong data-testid="explore-mode">{flying ? "Flying" : "Walking"}</strong>
      <span
        className="explore-hud__keys"
        title={
          "Click the view, then move the mouse to look (Esc gives the mouse back) · " +
          "W A S D to move · Shift to sprint · " +
          (flying ? "Space / E up, C / Q down" : "Space to jump") +
          " · F to " +
          (flying ? "walk" : "fly") +
          " · Esc to leave"
        }
      >
        <Kbd>W</Kbd>
        <Kbd>A</Kbd>
        <Kbd>S</Kbd>
        <Kbd>D</Kbd>
        <Kbd>Shift</Kbd>
        <Kbd>Space</Kbd>
        <span className="explore-hud__hint">
          {status?.looking ? "mouse to look" : "click to look"} ·{" "}
          {flying ? "Space up, C down" : "Space jumps"}
        </span>
      </span>
      <span className="mc-mono">{formatLength(speed, units)}/s</span>
      <GlassButton
        size="sm"
        variant="ghost"
        aria-pressed={flying}
        onClick={() => explore?.toggleMode()}
        leadingIcon={
          flying ? (
            <Footprints size={14} aria-hidden="true" />
          ) : (
            <Plane size={14} aria-hidden="true" />
          )
        }
        data-testid="explore-fly"
      >
        {flying ? "Walk" : "Fly"} <Kbd>F</Kbd>
      </GlassButton>
      <GlassButton
        size="sm"
        variant="ghost"
        onClick={() => setExploreMode(false)}
        leadingIcon={<X size={14} aria-hidden="true" />}
      >
        Exit
      </GlassButton>
    </GlassPanel>
  );
}
