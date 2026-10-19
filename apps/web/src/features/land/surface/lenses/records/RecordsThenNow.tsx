import { GlassSlider } from "@twin/ui";

import { useLandContext } from "@/state/landContext";

import { useRecords } from "./recordsStore";

/** An aligned old map laid over the land, faded between today's map and the old one. */
export function RecordsThenNow() {
  const overlay = useRecords((s) => s.overlay);
  const opacity = useRecords((s) => s.opacity);
  const shown = useLandContext((s) => (overlay ? Boolean(s.rasters[overlay.id]) : false));
  const tileError = useLandContext((s) => (overlay ? s.rasterErrors[overlay.id] : undefined));
  if (!overlay) return null;
  return (
    <div className="records-thennow" role="group" aria-label="Then and now">
      <div className="records-thennow__head">
        <span className="records-thennow__label">Then and now</span>
        <span className="records-thennow__name">{overlay.label}</span>
        <button
          type="button"
          className="land-link"
          onClick={() =>
            shown ? useRecords.getState().hideOverlay() : useRecords.getState().showOverlay(overlay)
          }
        >
          {shown ? "Hide old map" : "Show old map"}
        </button>
      </div>
      {shown && (
        <div className="records-thennow__fade">
          <span aria-hidden="true">Now</span>
          <GlassSlider
            aria-label="Aligned map opacity"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onValueChange={(value) => useRecords.getState().setOpacity(value)}
          />
          <span aria-hidden="true">Then</span>
        </div>
      )}
      {tileError && (
        <p className="land-alert" role="alert">
          The old map could not be drawn. {tileError}
        </p>
      )}
    </div>
  );
}
