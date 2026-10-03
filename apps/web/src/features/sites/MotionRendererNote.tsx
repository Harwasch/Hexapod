import { GlassButton } from "@twin/ui";

import { useInstances } from "@/state/instances";
import { useSettings } from "@/state/settings";

/**
 * Says when the splat renderer drawing a scan cannot move its objects -- skins in the wind,
 * telemetry -- and offers the Cesium renderer, as the objects panel does for hide and
 * highlight (`state/instances.ts` `motionGaps`, set by `cesium/scanView/scanMotion.ts`).
 * `assetId` picks one scan's gap; without it, the first scan that has one.
 */
export function MotionRendererNote({ assetId }: { assetId?: string }) {
  const gap = useInstances((s) =>
    assetId === undefined ? Object.values(s.motionGaps)[0] : s.motionGaps[assetId],
  );
  const setSettings = useSettings((s) => s.set);
  if (!gap) return null;
  // One line in the objects panel's note style; the reason is the line's tooltip.
  return (
    <div
      className="objects-panel__gap objects-panel__gap--line"
      role="note"
      title={gap.reason}
      data-testid="motion-renderer-gap"
    >
      <p>Wind and telemetry need the Cesium renderer.</p>
      <GlassButton
        size="sm"
        variant="ghost"
        onClick={() => setSettings({ splatRenderer: "cesium" })}
      >
        Use Cesium
      </GlassButton>
    </div>
  );
}
