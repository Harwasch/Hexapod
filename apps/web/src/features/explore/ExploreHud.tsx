import { X } from "lucide-react";

import { formatLength } from "@twin/geo";
import { GlassButton, GlassPanel, Kbd } from "@twin/ui";

import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";
import { useScene } from "@/cesium/SceneContext";

export function ExploreHud() {
  const exploreMode = useUi((s) => s.exploreMode);
  const setExploreMode = useUi((s) => s.setExploreMode);
  const units = useSettings((s) => s.units);
  const scene = useScene();
  if (!exploreMode) return null;
  const speed = scene?.explore.currentSpeed ?? 0;
  return (
    <GlassPanel strong pill className="explore-hud" role="status" data-testid="explore-hud">
      <span>
        <strong>Explore</strong> · drag to look
      </span>
      <span className="explore-hud__keys" aria-label="Movement keys">
        <Kbd>W</Kbd>
        <Kbd>A</Kbd>
        <Kbd>S</Kbd>
        <Kbd>D</Kbd>
        <Kbd>Q</Kbd>
        <Kbd>E</Kbd>
      </span>
      <span>
        scroll · <strong>{formatLength(speed, units)}/s</strong>
      </span>
      <GlassButton
        size="sm"
        variant="ghost"
        onClick={() => setExploreMode(false)}
        leadingIcon={<X size={14} aria-hidden="true" />}
      >
        Exit <Kbd>Esc</Kbd>
      </GlassButton>
    </GlassPanel>
  );
}
