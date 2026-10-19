import { Check, Magnet, Undo2, X } from "lucide-react";

import type { Footprint } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import { GlassButton, GlassPanel, GlassSelect } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useLand } from "@/state/land";

import { areaLabel, beginLand, finishSketch, useLandFlow } from "./landFlow";
import { LandSplitBar } from "./LandSplitBar";

function sketchArea(points: [number, number][]): number | null {
  const first = points[0];
  if (points.length < 3 || !first) return null;
  const polygon: Footprint = { type: "Polygon", coordinates: [[...points, first]] };
  return footprintAreaM2(polygon);
}

/**
 * The controls for drawing on the map, kept on the map: where you are clicking, what you have
 * placed so far, and the live area, with Undo and Done beside it.
 */
export function LandDrawBar() {
  const scene = useScene();
  const mode = useLand((s) => s.mode);
  const points = useLand((s) => s.points);
  const operation = useLand((s) => s.boundaryOperation);
  const width = useLand((s) => s.corridorWidth);
  const unit = useLand((s) => s.corridorUnit);
  const snapTarget = useLand((s) => s.snapTarget);
  const snapping = useLand((s) => s.snapEnabled && s.snapMapped);
  const busy = useLandFlow((s) => s.busy);

  if (mode === "split") return <LandSplitBar />;
  if (mode !== "draw" && mode !== "corridor" && mode !== "pick" && mode !== "edit") return null;

  const cancel = () => {
    useLand.getState().setBoundaryOperation(null);
    beginLand(scene, "browse");
  };
  const snapButton = (
    <GlassButton
      iconOnly
      size="sm"
      variant="ghost"
      active={snapping}
      aria-pressed={snapping}
      aria-label={snapping ? "Snapping to edges is on" : "Snapping to edges is off"}
      title="Snap to edges and corners nearby"
      onClick={() => {
        const land = useLand.getState();
        land.setSnapEnabled(!snapping);
        land.setSnapMapped(!snapping);
      }}
    >
      <Magnet size={16} aria-hidden="true" />
    </GlassButton>
  );

  if (mode === "pick")
    return (
      <GlassPanel strong pill className="land-drawbar" data-testid="land-drawbar">
        <span className="land-drawbar__status" aria-live="polite">
          {points.length ? "Looking for the mapped area…" : "Click a field, woodland, park or lake"}
        </span>
        <GlassButton size="sm" variant="ghost" onClick={cancel}>
          Cancel
        </GlassButton>
      </GlassPanel>
    );

  if (mode === "edit")
    return (
      <GlassPanel strong pill className="land-drawbar" data-testid="land-drawbar">
        <span className="land-drawbar__status">Drag the white corners to adjust</span>
        {snapButton}
        <GlassButton
          size="sm"
          variant="primary"
          leadingIcon={<Check size={15} aria-hidden="true" />}
          onClick={() => beginLand(scene, "browse")}
        >
          Done
        </GlassButton>
      </GlassPanel>
    );

  const area = mode === "draw" ? sketchArea(points) : null;
  const noun = mode === "corridor" ? "point" : "corner";
  const prompt =
    operation === "union"
      ? "Draw the area to add"
      : operation === "difference"
        ? "Draw the area to cut out"
        : operation === "intersection"
          ? "Draw the area to keep"
          : mode === "corridor"
            ? "Click along the centre line"
            : "Click the corners of your land";
  const needed = mode === "draw" ? 3 : 2;
  const widthValid = mode !== "corridor" || (Number.isFinite(width) && width > 0);
  return (
    <GlassPanel strong pill className="land-drawbar" data-testid="land-drawbar">
      <span className="land-drawbar__status" aria-live="polite">
        {points.length === 0 ? (
          prompt
        ) : (
          <>
            <b>{points.length}</b> {noun}
            {points.length === 1 ? "" : "s"}
            {area !== null && (
              <>
                {" · "}
                <b>{areaLabel(area)}</b>
              </>
            )}
          </>
        )}
      </span>
      {snapTarget && <span className="land-drawbar__snap">Snapping to {snapTarget}</span>}
      {mode === "corridor" && (
        <span className="land-row">
          <label className="land-visually-hidden" htmlFor="land-corridor-width">
            Total corridor width
          </label>
          <input
            id="land-corridor-width"
            className="glass-input"
            type="number"
            min="0.1"
            max={unit === "ft" ? 328084 : 100000}
            value={width}
            onChange={(event) => useLand.getState().setCorridorWidth(Number(event.target.value))}
          />
          <GlassSelect
            aria-label="Width units"
            value={unit}
            onChange={(event) =>
              useLand.getState().setCorridorUnit(event.target.value === "m" ? "m" : "ft")
            }
          >
            <option value="ft">ft wide</option>
            <option value="m">m wide</option>
          </GlassSelect>
        </span>
      )}
      {snapButton}
      <GlassButton
        size="sm"
        variant="ghost"
        disabled={!points.length || busy}
        leadingIcon={<Undo2 size={15} aria-hidden="true" />}
        onClick={() => useLand.getState().removePoint()}
      >
        Undo
      </GlassButton>
      <GlassButton
        size="sm"
        variant="ghost"
        leadingIcon={<X size={15} aria-hidden="true" />}
        onClick={cancel}
      >
        Cancel
      </GlassButton>
      <GlassButton
        size="sm"
        variant="primary"
        disabled={busy || points.length < needed || !widthValid}
        leadingIcon={<Check size={15} aria-hidden="true" />}
        onClick={() => void finishSketch()}
      >
        {busy ? (mode === "corridor" ? "Building…" : "Combining…") : "Done"}
      </GlassButton>
    </GlassPanel>
  );
}
