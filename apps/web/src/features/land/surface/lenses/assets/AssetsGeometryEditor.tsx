import { useEffect, useRef, useState } from "react";
import type { components, LandMapGeometry } from "@twin/contracts";
import { api, ApiError, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";

import {
  closeInventoryShape,
  keepGeometryHistory,
  openInventoryShape,
  validVertex,
  vertexCount,
  type InventoryShape,
  type Vertex,
} from "./assetsGeometry";

export type GeometryPreview = components["schemas"]["FeatureGeometryRead"];
type Preview = GeometryPreview;

/**
 * Exact shape editing: every vertex as coordinates, parts and holes, undo and redo, picked on
 * the map or typed. The server checks the shape before it can replace the draft's.
 */
export function AssetsGeometryEditor({
  landId,
  boundaryRevision,
  geometry,
  initialShape,
  active,
  onDraft,
  onApply,
  onCancel,
}: {
  landId: string;
  boundaryRevision: number;
  geometry: LandMapGeometry;
  initialShape?: InventoryShape | null;
  /** False while the lens is hidden: map picks are cancelled and nothing is drawn. */
  active: boolean;
  onDraft?: (shape: InventoryShape) => void;
  onApply: (geometry: LandMapGeometry, preview: Preview) => void;
  onCancel: () => void;
}) {
  const scene = useScene();
  const picker = useLandContext((s) => s.pointPicker);
  const owner = `assets-geometry:${landId}`;
  const [history, setHistory] = useState<{
    past: InventoryShape[];
    present: InventoryShape;
    future: InventoryShape[];
  }>(() => ({ past: [], present: initialShape ?? openInventoryShape(geometry), future: [] }));
  const draft = history.present;
  useEffect(() => {
    onDraft?.(draft);
  }, [draft, onDraft]);
  const [part, setPart] = useState(0),
    [ring, setRing] = useState(0),
    [offset, setOffset] = useState(0);
  const partIndex = Math.min(part, draft.parts.length - 1),
    ringIndex = Math.min(ring, (draft.parts[partIndex]?.length ?? 1) - 1);
  const points = draft.parts[partIndex]?.[ringIndex] ?? [];
  const start = Math.min(offset, Math.max(0, Math.floor((points.length - 1) / 50) * 50));
  const polygon = draft.type === "Polygon" || draft.type === "MultiPolygon";
  const [error, setError] = useState<string | null>(null),
    [busy, setBusy] = useState(false);
  const [review, setReview] = useState<{
    shape: InventoryShape;
    revision: number;
    value: Preview;
  } | null>(null);
  const preview =
    review?.shape === draft && review.revision === boundaryRevision ? review.value : null;
  const ticket = useRef(0),
    pickTicket = useRef(0);
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );
  useEffect(() => {
    if (!active) return;
    return () => {
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === owner) {
        scene?.areas.cancelPick();
        useLandContext.getState().setPointPicker(null);
      }
    };
  }, [active, owner, scene]);
  useEffect(() => {
    if (!active) return;
    // Only draw complete coordinates. Invalid/incomplete rings never become filled polygons.
    const features = draft.parts.flatMap((p, pi) =>
      p.flatMap((r, ri) => {
        const vertices = r.flatMap((coordinates, i) =>
          validVertex(coordinates)
            ? [
                {
                  id: `${pi}:${ri}:${i}`,
                  label: `Part ${pi + 1}, ring ${ri + 1}, vertex ${i + 1}`,
                  geometry: { type: "Point" as const, coordinates },
                },
              ]
            : [],
        );
        const lines =
          r.length >= 2 && r.every(validVertex)
            ? [
                {
                  id: `${pi}:${ri}`,
                  label: `Part ${pi + 1}, ring ${ri + 1}`,
                  geometry: {
                    type: "LineString" as const,
                    coordinates: polygon && r.length >= 3 ? [...r, ...r.slice(0, 1)] : r,
                  },
                },
              ]
            : [];
        // Keep large imports responsive: show all paths, point handles only on the active page.
        return [
          ...lines,
          ...(pi === partIndex && ri === ringIndex ? vertices.slice(start, start + 50) : []),
        ];
      }),
    );
    useLandContext.getState().setLayer({
      id: owner,
      title: "Editing asset shape",
      style: { color: "#f1cb7a", fill: 0.15, width: 3 },
      features,
    });
    return () => useLandContext.getState().removeLayer(owner);
  }, [active, draft, owner, polygon, partIndex, ringIndex, start]);
  function cancelPick() {
    pickTicket.current += 1;
    if (useLandContext.getState().pointPicker === owner) {
      scene?.areas.cancelPick();
      useLandContext.getState().setPointPicker(null);
    }
  }
  function invalidate() {
    ticket.current++;
    cancelPick();
    setBusy(false);
    setReview(null);
    setError(null);
  }
  function update(next: InventoryShape) {
    invalidate();
    setHistory((h) => ({
      past: keepGeometryHistory([...h.past, h.present]),
      present: next,
      future: [],
    }));
  }
  function replacePoints(next: Vertex[]) {
    update({
      ...draft,
      parts: draft.parts.map((p, pi) =>
        pi !== partIndex ? p : p.map((r, ri) => (ri !== ringIndex ? r : next)),
      ),
    });
  }
  async function pick(index?: number) {
    if (!scene || !active) return;
    const current = ++pickTicket.current;
    useLandContext.getState().setPointPicker(owner);
    const point = await scene.areas.pickGround();
    if (current !== pickTicket.current || useLandContext.getState().pointPicker !== owner) return;
    useLandContext.getState().setPointPicker(null);
    if (!point) return;
    const value: Vertex = [point.longitude, point.latitude];
    replacePoints(
      index === undefined ? [...points, value] : points.map((p, i) => (i === index ? value : p)),
    );
    if (index === undefined) setOffset(Math.floor(points.length / 50) * 50);
  }
  async function validate() {
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    setReview(null);
    cancelPick();
    try {
      const value = await unwrap(
        api.POST("/api/v1/land/{land_id}/features/geometry/preview", {
          params: { path: { land_id: landId } },
          body: { geometry: closeInventoryShape(draft) },
        }),
      );
      if (current === ticket.current)
        setReview({ shape: draft, revision: boundaryRevision, value });
    } catch (cause) {
      if (current === ticket.current)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  }
  return (
    <section className="assets-geometry assets-tool" aria-label="Edit asset geometry">
      <p className="land-muted">
        Pick corners on the map or type coordinates, then check the shape before using it.
      </p>
      <div className="land-row">
        <button
          type="button"
          disabled={!history.past.length}
          onClick={() => {
            invalidate();
            setHistory((h) => ({
              past: h.past.slice(0, -1),
              present: h.past.at(-1) ?? h.present,
              future: keepGeometryHistory([h.present, ...h.future].reverse()).reverse(),
            }));
          }}
        >
          Undo geometry
        </button>
        <button
          type="button"
          disabled={!history.future.length}
          onClick={() => {
            invalidate();
            setHistory((h) => ({
              past: keepGeometryHistory([...h.past, h.present]),
              present: h.future[0] ?? h.present,
              future: h.future.slice(1),
            }));
          }}
        >
          Redo geometry
        </button>
      </div>
      <label className="land-field">
        Geometry type
        <select
          aria-label="Geometry type"
          value={draft.type}
          onChange={(e) => {
            const type = e.target.value as InventoryShape["type"];
            const first = draft.parts[0]?.[0] ?? [];
            const parts: Vertex[][][] =
              type === "Point"
                ? [[first.length ? first.slice(0, 1) : [[NaN, NaN]]]]
                : type === "LineString"
                  ? [[first]]
                  : polygon
                    ? type === "Polygon"
                      ? draft.parts.slice(0, 1)
                      : draft.parts
                    : [[first]];
            update({ type, parts });
            setPart(0);
            setRing(0);
            setOffset(0);
          }}
        >
          <option value="Point">Point</option>
          <option value="LineString">Line</option>
          <option value="Polygon">Area</option>
          <option value="MultiPolygon">Multiple areas</option>
        </select>
      </label>
      <p className="land-muted">
        Changing the type can drop rings or parts. Undo brings them back.
      </p>
      {polygon && (
        <>
          {draft.type === "MultiPolygon" && (
            <label className="land-field">
              Area part
              <select
                aria-label="Area part"
                value={partIndex}
                onChange={(e) => {
                  cancelPick();
                  setPart(Number(e.target.value));
                  setRing(0);
                  setOffset(0);
                }}
              >
                {draft.parts.map((_, i) => (
                  <option key={i} value={i}>
                    Part {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="land-field">
            Boundary ring
            <select
              aria-label="Boundary ring"
              value={ringIndex}
              onChange={(e) => {
                cancelPick();
                setRing(Number(e.target.value));
                setOffset(0);
              }}
            >
              {draft.parts[partIndex]?.map((_, i) => (
                <option key={i} value={i}>
                  {i === 0 ? "Outer boundary" : `Exclusion ${i}`}
                </option>
              ))}
            </select>
          </label>
          <div className="land-row">
            <button
              type="button"
              onClick={() => {
                update({
                  ...draft,
                  parts: draft.parts.map((p, i) => (i === partIndex ? [...p, []] : p)),
                });
                setRing(draft.parts[partIndex]?.length ?? 0);
                setOffset(0);
              }}
            >
              Add exclusion
            </button>
            {ringIndex > 0 && (
              <button
                type="button"
                onClick={() => {
                  update({
                    ...draft,
                    parts: draft.parts.map((p, i) =>
                      i === partIndex ? p.filter((_, ri) => ri !== ringIndex) : p,
                    ),
                  });
                  setRing(0);
                  setOffset(0);
                }}
              >
                Remove exclusion
              </button>
            )}
            {draft.type === "MultiPolygon" && (
              <button
                type="button"
                onClick={() => {
                  update({ ...draft, parts: [...draft.parts, [[]]] });
                  setPart(draft.parts.length);
                  setRing(0);
                  setOffset(0);
                }}
              >
                Add area part
              </button>
            )}
            {draft.parts.length > 1 && (
              <button
                type="button"
                onClick={() => {
                  update({ ...draft, parts: draft.parts.filter((_, i) => i !== partIndex) });
                  setPart(0);
                  setRing(0);
                  setOffset(0);
                }}
              >
                Remove area part
              </button>
            )}
          </div>
        </>
      )}
      <ol className="assets-vertices" start={start + 1}>
        {points.slice(start, start + 50).map((point, local) => {
          const index = start + local;
          return (
            <li key={index}>
              <div className="assets-coordinates">
                {(["Longitude", "Latitude"] as const).map((label, axis) => (
                  <label key={label}>
                    {label}
                    <input
                      aria-label={`Vertex ${index + 1} ${label.toLowerCase()}`}
                      type="number"
                      step="any"
                      min={axis ? -90 : -180}
                      max={axis ? 90 : 180}
                      value={Number.isFinite(point[axis]) ? point[axis] : ""}
                      onChange={(e) =>
                        replacePoints(
                          points.map((p, i) =>
                            i === index
                              ? axis === 0
                                ? [e.target.valueAsNumber, p[1]]
                                : [p[0], e.target.valueAsNumber]
                              : p,
                          ),
                        )
                      }
                    />
                  </label>
                ))}
              </div>
              <div className="land-row">
                <button type="button" disabled={!scene || !active} onClick={() => void pick(index)}>
                  Pick vertex {index + 1} on map
                </button>
                {draft.type !== "Point" && (
                  <>
                    <button
                      type="button"
                      onClick={() =>
                        replacePoints([
                          ...points.slice(0, index + 1),
                          [NaN, NaN],
                          ...points.slice(index + 1),
                        ])
                      }
                    >
                      Insert after vertex {index + 1}
                    </button>
                    <button
                      type="button"
                      onClick={() => replacePoints(points.filter((_, i) => i !== index))}
                    >
                      Remove vertex {index + 1}
                    </button>
                  </>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      {points.length > 50 && (
        <div className="land-row">
          <button type="button" disabled={start === 0} onClick={() => setOffset(start - 50)}>
            Previous vertices
          </button>
          <span>
            {start + 1}–{Math.min(start + 50, points.length)} of {points.length}
          </span>
          <button
            type="button"
            disabled={start + 50 >= points.length}
            onClick={() => setOffset(start + 50)}
          >
            More vertices
          </button>
        </div>
      )}
      {draft.type !== "Point" && (
        <div className="land-row">
          <button
            type="button"
            disabled={!scene || !active || vertexCount(draft) >= 20_000}
            onClick={() => void pick()}
          >
            Pick next vertex on map
          </button>
          <button
            type="button"
            disabled={vertexCount(draft) >= 20_000}
            onClick={() => {
              replacePoints([...points, [NaN, NaN]]);
              setOffset(Math.floor(points.length / 50) * 50);
            }}
          >
            Add coordinate
          </button>
        </div>
      )}
      {picker === owner && (
        <p role="status" className="land-muted">
          Click the map to place this vertex.{" "}
          <button type="button" className="land-link" onClick={cancelPick}>
            Cancel map pick
          </button>
        </p>
      )}
      {error && (
        <p role="alert" className="land-alert">
          {error}
        </p>
      )}
      {preview && (
        <div role="status" className="assets-geometry__ok">
          <strong>Geometry is valid.</strong>
          <p>
            {preview.intersectsLand
              ? "Intersects the current land boundary."
              : `${Math.round(preview.distanceM).toLocaleString()} m outside the current land boundary.`}
          </p>
          <dl>
            {(
              [
                ["Area", preview.areaM2, "m²"],
                ["Length", preview.lengthM, "m"],
                ["Perimeter", preview.perimeterM, "m"],
              ] as const
            )
              .filter(([, value]) => value !== null)
              .map(([label, value, unit]) => (
                <div key={label}>
                  <dt>{label}</dt>
                  <dd>
                    {value?.toLocaleString(undefined, { maximumFractionDigits: 1 })} {unit}
                  </dd>
                </div>
              ))}
          </dl>
          <p className="land-muted">Map coordinates are not a survey.</p>
        </div>
      )}
      <div className="land-row">
        <button type="button" disabled={busy} onClick={() => void validate()}>
          {busy ? "Checking geometry…" : "Preview geometry"}
        </button>
        <button
          type="button"
          disabled={!preview || busy}
          onClick={() => {
            if (preview) {
              cancelPick();
              onApply(preview.geometry, preview);
            }
          }}
        >
          Apply geometry to draft
        </button>
        <button
          type="button"
          onClick={() => {
            invalidate();
            onCancel();
          }}
        >
          Discard geometry edits
        </button>
      </div>
    </section>
  );
}
