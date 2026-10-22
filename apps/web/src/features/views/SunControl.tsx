import { GlassSegmentedControl } from "@twin/ui";

import { SUNLIGHT_TEXT, useSeveralDates } from "@/lib/sunlightText";
import { useSites } from "@/state/sites";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting, type SunMode } from "@/state/lighting";

/**
 * Sunlight (ADR 0019): Off, the sun of Now, or the sun At capture -- the shown date's, when its
 * capture time was recorded. CesiumJS draws it; under another renderer the control is offered
 * disabled and the views control's shared notice gives the reason and the switch. Off while As
 * captured is on, and it says so: what it shows is drawn, never the choice it keeps for after.
 * At capture is off while more than one date is drawn (Compare, a per-cell view rule), saying
 * why: one scene clock cannot be each date's capture.
 */
export function SunControl({
  onCesium,
  captureTime,
}: {
  onCesium: boolean;
  captureTime: string | null;
}) {
  const sun = useLighting((s) => s.sun);
  const set = useLighting((s) => s.set);
  const held = useAsCaptured((s) => s.on);
  const several = useSeveralDates(useSites((s) => s.activeSiteId));
  const usable = onCesium && !held;
  return (
    <span
      className="views__rule"
      title={
        captureTime === null
          ? `${SUNLIGHT_TEXT.hint} ${SUNLIGHT_TEXT.noCaptureTime}.`
          : SUNLIGHT_TEXT.hint
      }
      data-testid="views-sun"
    >
      <span className="views__rule-label">{SUNLIGHT_TEXT.label}</span>
      <GlassSegmentedControl<SunMode>
        aria-label={SUNLIGHT_TEXT.label}
        value={usable ? sun : "off"}
        onValueChange={set}
        options={[
          { value: "off", label: SUNLIGHT_TEXT.off, disabled: !usable },
          { value: "now", label: SUNLIGHT_TEXT.now, disabled: !usable },
          {
            value: "capture",
            label: SUNLIGHT_TEXT.capture,
            disabled: !usable || captureTime === null || several,
          },
        ]}
      />
      {held && (
        <span className="views__notice" role="note" data-testid="views-sun-as-captured">
          {SUNLIGHT_TEXT.asCaptured}
        </span>
      )}
      {usable && sun === "capture" && several && (
        <span className="views__notice" role="note" data-testid="views-sun-several">
          {SUNLIGHT_TEXT.several}
        </span>
      )}
      {usable && sun === "capture" && !several && captureTime === null && (
        <span className="views__notice" role="note" data-testid="views-sun-no-time">
          {SUNLIGHT_TEXT.nowInstead}
        </span>
      )}
    </span>
  );
}
