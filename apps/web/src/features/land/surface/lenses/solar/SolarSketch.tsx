import { useRef, useState } from "react";
import { Check, Undo2, X } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import { GlassButton, GlassSelect } from "@twin/ui";

import { describeError } from "@/lib/log";

import { importBoundary } from "../../../geometry";
import { useMappedShapes } from "./solarData";
import { zoneArea } from "./solarWords";
import type { SolarDraft } from "./useSolarDraft";

function sketchArea(corners: number[][]): number | null {
  const first = corners[0];
  if (corners.length < 3 || !first) return null;
  return footprintAreaM2({ type: "Polygon", coordinates: [[...corners, first]] });
}

/** Drawing the array on the map: each click is a corner; the live size shows as you go. */
export function SolarSketch({ draft, land }: { draft: SolarDraft; land: LandArea }) {
  const corners = draft.draft?.points ?? [];
  const area = sketchArea(corners);
  const status = !draft.canDraw
    ? "The map is still loading. You can use the whole land or a file instead."
    : corners.length === 0
      ? draft.picking
        ? "Click the corners of the array on the map."
        : "Drawing is paused."
      : `${corners.length} ${corners.length === 1 ? "corner" : "corners"}${area ? ` · ${zoneArea(area)}` : ""}${draft.picking ? "" : " · paused"}`;
  return (
    <div className="land-solar-sketch">
      <p className="land-solar-sketch__status" aria-live="polite">
        {status}
        {draft.picking && corners.length < 3 && <span className="land-muted"> Escape pauses.</span>}
      </p>
      <div className="land-row">
        {!draft.picking && draft.canDraw && (
          <GlassButton size="sm" onClick={draft.keepDrawing}>
            Keep drawing
          </GlassButton>
        )}
        <GlassButton
          size="sm"
          variant="ghost"
          disabled={!corners.length}
          leadingIcon={<Undo2 size={15} aria-hidden="true" />}
          onClick={draft.undoCorner}
        >
          Undo
        </GlassButton>
        <GlassButton
          size="sm"
          variant="ghost"
          leadingIcon={<X size={15} aria-hidden="true" />}
          onClick={draft.cancelSketch}
        >
          Cancel
        </GlassButton>
        <GlassButton
          size="sm"
          variant="primary"
          disabled={corners.length < 3}
          leadingIcon={<Check size={15} aria-hidden="true" />}
          onClick={draft.finishSketch}
        >
          Done
        </GlassButton>
      </div>
      <SolarZoneSources draft={draft} land={land} />
    </div>
  );
}

/** Other ways to set the array: the whole land, a shape already mapped, or a file. */
export function SolarZoneSources({ draft, land }: { draft: SolarDraft; land: LandArea }) {
  const [shapesOpen, setShapesOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const file = useRef<HTMLInputElement>(null);
  const shapes = useMappedShapes(land, shapesOpen);
  return (
    <div className="land-solar-sources">
      <p className="land-muted">
        Or use{" "}
        <button
          type="button"
          className="land-link"
          onClick={() => draft.setZone(land.boundary, "The whole land boundary.")}
        >
          the whole land
        </button>
        ,{" "}
        <button
          type="button"
          className="land-link"
          aria-expanded={shapesOpen}
          onClick={() => setShapesOpen(!shapesOpen)}
        >
          a mapped shape
        </button>{" "}
        or{" "}
        <button type="button" className="land-link" onClick={() => file.current?.click()}>
          a GeoJSON file
        </button>
        .
      </p>
      {shapesOpen && (
        <label className="land-field">
          Mapped shape
          <GlassSelect
            value=""
            disabled={!shapes.data?.length}
            onChange={(event) => {
              const shape = shapes.data?.find((item) => item.id === event.target.value);
              if (
                shape &&
                (shape.geometry.type === "Polygon" || shape.geometry.type === "MultiPolygon")
              )
                draft.setZone(
                  shape.geometry,
                  `Mapped feature: ${shape.name}. Check its outline on site.`,
                );
            }}
          >
            <option value="">
              {shapes.isPending
                ? "Loading shapes…"
                : shapes.isError
                  ? "Shapes could not load"
                  : shapes.data?.length
                    ? "Choose a shape…"
                    : "No mapped shapes on this land"}
            </option>
            {shapes.data?.map((shape) => (
              <option key={shape.id} value={shape.id}>
                {shape.name}
              </option>
            ))}
          </GlassSelect>
        </label>
      )}
      <input
        ref={file}
        className="land-visually-hidden"
        type="file"
        tabIndex={-1}
        aria-label="Import an array outline (GeoJSON)"
        accept=".json,.geojson,application/json,application/geo+json"
        onChange={(event) => {
          const chosen = event.target.files?.[0];
          event.target.value = "";
          if (!chosen) return;
          setError(null);
          void (async () => {
            try {
              if (chosen.size > 5_000_000) throw new Error("Outlines are limited to 5 MB.");
              draft.setZone(
                importBoundary(await chosen.text()),
                `Imported ${chosen.name}; check its accuracy.`,
              );
            } catch (cause) {
              setError(`That file could not be used. ${describeError(cause)}`);
            }
          })();
        }}
      />
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
