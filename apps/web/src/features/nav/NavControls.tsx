import { Compass, Home } from "lucide-react";

import { GlassButton, GlassTooltip } from "@twin/ui";

import { HOTKEYS, hotkeyKeys } from "@/app/hotkeys";
import { useScene } from "@/cesium/SceneContext";
import { useViewer } from "@/state/viewer";

/**
 * Compass and Earth, in the bottom-right pill beside the data credits (`AppShell`).
 *
 * The rest of the camera is on the keyboard and in the command box: zoom is the wheel, a
 * pinch or `+` / `-`; top-down is `T`; explore mode is `G` — none of them needs a button
 * on the map.
 */
export function NavControls() {
  const scene = useScene();
  const heading = useViewer((s) => s.camera.heading);
  return (
    <div className="nav-controls" role="toolbar" aria-label="Camera">
      <GlassTooltip
        content="Reset north"
        shortcut={hotkeyKeys(HOTKEYS.resetNorth).join(" ")}
        side="top"
      >
        <GlassButton
          iconOnly
          variant="ghost"
          size="sm"
          aria-label="Reset north"
          onClick={() => scene?.camera.resetNorth()}
          data-testid="nav-north"
        >
          <Compass
            className="compass"
            size={17}
            aria-hidden="true"
            style={{ transform: `rotate(${-heading}deg)` }}
          />
        </GlassButton>
      </GlassTooltip>
      <GlassTooltip content="Earth" shortcut={hotkeyKeys(HOTKEYS.home).join(" ")} side="top">
        <GlassButton
          iconOnly
          variant="ghost"
          size="sm"
          aria-label="Fly to Earth view"
          onClick={() => scene?.camera.flyHome()}
          data-testid="nav-home"
        >
          <Home size={17} aria-hidden="true" />
        </GlassButton>
      </GlassTooltip>
    </div>
  );
}
