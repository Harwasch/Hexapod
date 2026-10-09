import { useEffect, useMemo, useRef, useState } from "react";
import type { components } from "@twin/contracts";
import { boundsOf } from "@twin/geo";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { landScope } from "@/state/landIdentity";

type SplitResult = components["schemas"]["BoundarySplitResult"];
const LAYER = "boundary-split";
export function LandBoundarySplit() {
  const state = useLand();
  const scene = useScene();
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{
    points: typeof state.points;
    boundary: NonNullable<typeof state.draft>["boundary"];
    result: SplitResult;
  } | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      useLandContext.getState().removeLayer(LAYER);
    };
  }, []);
  const current =
    preview?.points === state.points && preview.boundary === state.draft?.boundary
      ? preview.result
      : null;
  const features = useMemo(
    () =>
      current?.parts.map((part, index) => ({
        id: String(index),
        label: `Part ${index + 1}`,
        geometry: part.boundary,
      })),
    [current],
  );
  useEffect(() => {
    if (features)
      useLandContext.getState().setLayer({
        id: LAYER,
        title: "Boundary split preview",
        selectedIds: selected.map(String),
        features,
      });
    else useLandContext.getState().removeLayer(LAYER);
  }, [features, selected]);
  const request = async (apply: boolean) => {
    if (!state.draft || state.points.length < 2) return;
    const { points, draft, session } = state;
    const scope = landScope();
    const valid = () =>
      mounted.current &&
      landScope() === scope &&
      useLand.getState().session === session &&
      useLand.getState().points === points &&
      useLand.getState().draft === draft;
    setBusy(true);
    state.setError(null);
    try {
      const result = await unwrap(
        api.POST("/api/v1/land/split", {
          body: {
            boundary: draft.boundary,
            coordinates: points,
            ...(apply ? { keepParts: selected } : {}),
          },
        }),
      );
      if (!valid()) return;
      if (apply && result.selection) {
        state.updateBoundary(result.selection.boundary);
        state.begin("browse");
      } else {
        setPreview({ points, boundary: draft.boundary, result });
        setSelected([]);
      }
    } catch (error) {
      if (valid()) state.setError(describeError(error));
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  return (
    <section className="land-selection" aria-label="Split boundary">
      <span className="land-eyebrow">Shape your study area</span>
      <h3>Split with a line.</h3>
      <p>
        Draw from outside one boundary edge to outside another. Add bends as needed, then review the
        pieces. The original land stays saved until you save a revision.
      </p>
      <p aria-live="polite">{state.points.length} cut points placed.</p>
      <div className="land-actions">
        <button
          type="button"
          disabled={busy || state.points.length < 2}
          onClick={() => void request(false)}
        >
          {busy ? "Calculating…" : "Preview split"}
        </button>
        <button type="button" disabled={busy || !state.points.length} onClick={state.removePoint}>
          Last cut point
        </button>
        <button type="button" onClick={() => state.begin("browse")}>
          Cancel split
        </button>
      </div>
      {current && (
        <>
          <h4>Choose the pieces to keep</h4>
          <p>
            Selected pieces are highlighted in gold. Unselected pieces will be excluded from the
            draft.
          </p>
          <ul className="land-map-feature-list land-split-pieces" aria-label="Split pieces">
            {current.parts.map((part, index) => (
              <li key={index} data-selected={selected.includes(index)}>
                <label>
                  <input
                    type="checkbox"
                    checked={selected.includes(index)}
                    onChange={(event) =>
                      setSelected(
                        event.target.checked
                          ? [...selected, index]
                          : selected.filter((id) => id !== index),
                      )
                    }
                  />{" "}
                  Part {index + 1} ·{" "}
                  {(part.areaM2 / 4046.8564224).toLocaleString(undefined, {
                    maximumFractionDigits: 2,
                  })}{" "}
                  acres
                </label>
                <button
                  type="button"
                  disabled={!scene}
                  aria-label={`Focus on part ${index + 1}`}
                  onClick={() => {
                    const b = boundsOf(part.boundary);
                    scene?.camera.flyToRectangle(b.west, b.south, b.east, b.north);
                  }}
                >
                  Focus
                </button>
              </li>
            ))}
          </ul>
          <p>
            {selected.length} of {current.parts.length} pieces selected.
          </p>
          <div className="land-actions">
            <button
              type="button"
              className="land-primary"
              disabled={busy || !selected.length}
              onClick={() => void request(true)}
            >
              Keep selected pieces
            </button>
          </div>
        </>
      )}
    </section>
  );
}
