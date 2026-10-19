import type { UnitSystem } from "@twin/geo";

import { bandOf, heightLabel, monthLabel, percentLabel, slopePercent } from "./overviewData";
import type { ShownLayer } from "./useOverview";

/**
 * What the colours on the map mean, under the fact whose layer is shown: a colour ramp with its
 * two ends. Land cover needs none; its fact lists the classes with their colours.
 */
export function OverviewLegend({ shown, units }: { shown: ShownLayer; units: UnitSystem }) {
  const band = bandOf(shown.raster, shown.band);
  if (!band || band.palette === "categorical") return null;
  const low = band.displayMinimum ?? band.minimum;
  const high = band.displayMaximum ?? band.maximum;
  if (low == null || high == null) return null;
  const series = shown.raster.metadata.vegetation;
  const observation = series?.observations.find((item) => item.band === shown.band);
  const first = series?.observations[0];
  let caption: string;
  let ends: [string, string];
  if (shown.view.layer === "elevation") {
    caption = "Elevation";
    ends = [heightLabel(low, units), heightLabel(high, units)];
  } else if (shown.view.layer === "slope") {
    caption = "Slope";
    ends = [percentLabel(slopePercent(low)), percentLabel(slopePercent(high))];
  } else if (observation) {
    // A vegetation month or a signature's season: both are dated NDVI.
    caption = `Greenness, ${monthLabel(observation.period.startDate)}`;
    ends = ["Bare", "Green"];
  } else {
    caption = first ? `Change since ${monthLabel(first.period.startDate)}` : "Greenness change";
    ends = ["Less green", "Greener"];
  }
  return (
    <div
      className="overview-legend"
      role="img"
      aria-label={`Map colours: ${caption}, from ${ends[0]} to ${ends[1]}`}
    >
      {(shown.view.layer === "vegetation" || shown.view.layer === "signature") && (
        <span className="overview-legend__caption">{caption}</span>
      )}
      <span className={`overview-ramp overview-ramp--${band.palette}`} />
      <span className="overview-ends">
        <span>{ends[0]}</span>
        <span>{ends[1]}</span>
      </span>
    </div>
  );
}
