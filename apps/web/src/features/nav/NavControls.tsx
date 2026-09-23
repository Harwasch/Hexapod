import { Compass, Home, Minus, MousePointerSquareDashed, Plus, Square } from "lucide-react";

import { GlassButton, GlassPanel, GlassTooltip } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

/** Bottom-right compact camera controls. */
export function NavControls() {
  const scene = useScene();
  const heading = useViewer((s) => s.camera.heading);
  const nearSiteId = useSites((s) => s.nearSiteId);
  const exploreMode = useUi((s) => s.exploreMode);
  const setExploreMode = useUi((s) => s.setExploreMode);
  const exploreSpeed = useSettings((s) => s.exploreSpeed);

  return (
    <GlassPanel
      strong
      pill
      className="nav-controls"
      role="toolbar"
      aria-label="Camera"
      aria-orientation="vertical"
    >
      <GlassTooltip content="Reset north" shortcut="N" side="left">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Reset north"
          onClick={() => scene?.camera.resetNorth()}
          data-testid="nav-north"
        >
          <Compass
            className="compass"
            size={18}
            aria-hidden="true"
            style={{ transform: `rotate(${-heading}deg)` }}
          />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip content="Top-down view" shortcut="T" side="left">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Top-down view"
          onClick={() => scene?.camera.topDown()}
          data-testid="nav-topdown"
        >
          <Square size={16} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip content="Zoom in" shortcut="+" side="left">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Zoom in"
          onClick={() => scene?.camera.zoomBy(0.5)}
          data-testid="nav-zoom-in"
        >
          <Plus size={18} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip content="Zoom out" shortcut="-" side="left">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Zoom out"
          onClick={() => scene?.camera.zoomBy(-1)}
          data-testid="nav-zoom-out"
        >
          <Minus size={18} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip
        content={exploreMode ? "Exit explore mode" : "Explore mode (ground level)"}
        shortcut="G"
        side="left"
      >
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label={exploreMode ? "Exit explore mode" : "Explore mode"}
          active={exploreMode}
          disabled={!nearSiteId && !exploreMode}
          onClick={() => {
            setExploreMode(!exploreMode);
            if (!exploreMode) scene?.explore.setSpeed(exploreSpeed);
          }}
          data-testid="nav-explore"
        >
          <MousePointerSquareDashed size={18} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip content="Earth" shortcut="H" side="left">
        <GlassButton
          iconOnly
          variant="ghost"
          aria-label="Fly to Earth view"
          onClick={() => scene?.camera.flyHome()}
          data-testid="nav-home"
        >
          <Home size={18} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
    </GlassPanel>
  );
}
