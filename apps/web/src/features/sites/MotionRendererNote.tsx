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
  return (
    <div className="instance-panel__gap" role="note" data-testid="motion-renderer-gap">
      <p>Wind sway and live telemetry need the Cesium renderer. {gap.reason}</p>
      <GlassButton
        size="sm"
        variant="ghost"
        onClick={() => setSettings({ splatRenderer: "cesium" })}
      >
        Use the Cesium renderer
      </GlassButton>
    </div>
  );
}
