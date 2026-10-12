import type { components } from "@twin/contracts";

import { OverviewPeriods } from "./surface/lenses/overview/OverviewPeriods";

type Period = components["schemas"]["VegetationPeriod"];

/**
 * The old research tab's vegetation start. The month picker now lives in the Overview lens
 * (`OverviewPeriods`); this keeps the old entry point working until that tab is retired.
 */
export function VegetationStart({
  disabled,
  busy,
  onStart,
}: {
  disabled: boolean;
  busy: boolean;
  onStart: (periods: Period[]) => void;
}) {
  return (
    <div className="land-terrain-start land-vegetation-start">
      <strong>Vegetation through time</strong>
      <OverviewPeriods
        disabled={disabled}
        busy={busy}
        onStart={onStart}
        startLabel="Compare vegetation"
      />
    </div>
  );
}
