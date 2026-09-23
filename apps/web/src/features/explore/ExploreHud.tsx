import { Footprints, X } from "lucide-react";

import { formatLength } from "@twin/geo";
import { GlassButton, GlassPanel, Kbd } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/**
 * Ground-level explore mode: what it is, the keys that matter, and the way out.
 *
 * The full key list (Q/E, scroll, Shift+scroll) is one hover away rather than a sentence
 * across the bottom of the screen.
 */
export function ExploreHud() {
  const exploreMode = useUi((s) => s.exploreMode);
  const setExploreMode = useUi((s) => s.setExploreMode);
  const units = useSettings((s) => s.units);
  const scene = useScene();
  if (!exploreMode) return null;
  const speed = scene?.explore.currentSpeed ?? 0;
  return (
    <GlassPanel strong pill className="explore-hud" role="status" data-testid="explore-hud">
      <Footprints size={15} aria-hidden="true" />
      <strong>Explore</strong>
      <span
        className="explore-hud__keys"
        title="W A S D to move · Q / E down and up · drag to look · scroll to move forward · Shift + scroll to change speed"
      >
        <Kbd>W</Kbd>
        <Kbd>A</Kbd>
        <Kbd>S</Kbd>
        <Kbd>D</Kbd>
        <span className="explore-hud__hint">move · drag to look</span>
      </span>
      <span className="mc-mono">{formatLength(speed, units)}/s</span>
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
