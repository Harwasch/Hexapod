import { GlassButton, GlassSegmentedControl } from "@twin/ui";

import { SUNLIGHT_TEXT, useSeveralDates } from "@/lib/sunlightText";
import { DRAW_WITH_PLAYCANVAS, drawWithPlaycanvas, NEEDS_SUNLIGHT_RENDERER } from "@/lib/viewModes";
import { useSites } from "@/state/sites";
import { useAsCaptured } from "@/state/asCaptured";
import { useLighting, type SunMode } from "@/state/lighting";

/**
 * Sunlight (ADR 0019): Off, the sun of Now, or the sun At capture -- the shown date's, when its
 * capture time was recorded. CesiumJS and PlayCanvas draw it (`drawable`); under Spark or the
 * WebGPU trial the control is offered disabled, saying so, with the switch to PlayCanvas. Off
 * while As captured is on, and it says so: what it shows is drawn, never the choice it keeps for
 * after. At capture is off while more than one date is drawn (Compare, a per-cell view rule),
 * saying why: one scene clock cannot be each date's capture.
 */
export function SunControl({
  drawable,
  captureTime,
}: {
  drawable: boolean;
  captureTime: string | null;
}) {
  const sun = useLighting((s) => s.sun);
  const set = useLighting((s) => s.set);
  const held = useAsCaptured((s) => s.on);
  const several = useSeveralDates(useSites((s) => s.activeSiteId));
  const usable = drawable && !held;
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
      {!drawable && (
        <span className="views__notice" role="note" data-testid="views-sun-needs">
          <span>{NEEDS_SUNLIGHT_RENDERER}</span>
          <GlassButton size="sm" variant="ghost" onClick={drawWithPlaycanvas}>
            {DRAW_WITH_PLAYCANVAS}
          </GlassButton>
        </span>
      )}
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
