import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components } from "@twin/contracts";
import { api, ApiError, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useUi } from "@/state/ui";
import { AlignmentFit, SavedAlignment } from "./SavedAlignment";

type Point = components["schemas"]["ImageControlPoint"];
type Fit = components["schemas"]["ImageRegistrationResult"];
type Request = components["schemas"]["ImageRegistrationCreate"];
type Snapshot = components["schemas"]["ArchiveImageRead"];
interface Props {
  id: string;
  image: Snapshot;
  url: string;
  title: string;
  attribution: string;
}
interface Draft {
  name: string;
  notes: string;
  points: Point[];
  requestKey: string;
}

function restore(key: string, title: string): Draft {
  const empty = {
    name: title.slice(0, 200),
    notes: "",
    points: [],
    requestKey: crypto.randomUUID(),
  };
  try {
    const raw = localStorage.getItem(key);
    if (!raw || raw.length > 32000) return empty;
    const value = JSON.parse(raw) as Partial<Draft>;
    if (
      typeof value.name !== "string" ||
      value.name.length > 200 ||
      typeof value.notes !== "string" ||
      value.notes.length > 2000 ||
      !Array.isArray(value.points) ||
      value.points.length > 30 ||
      typeof value.requestKey !== "string" ||
      !/^[a-f0-9-]{36}$/.test(value.requestKey)
    )
      return empty;
    if (
      !value.points.every(
        (p) =>
          p &&
          typeof p.label === "string" &&
          p.label.length <= 100 &&
          [p.imageX, p.imageY, p.longitude, p.latitude].every(Number.isFinite) &&
          p.imageX >= 0 &&
          p.imageX <= 8192 &&
          p.imageY >= 0 &&
          p.imageY <= 8192 &&
          Math.abs(p.longitude) <= 180 &&
          Math.abs(p.latitude) <= 90,
      )
    )
      return empty;
    return value as Draft;
  } catch {
    return empty;
  }
}

/** Remount on identity/image change: private draft state and in-flight work never cross scopes. */
export function ArchiveMapAlignment(props: Props) {
  const scope = useLandScope();
  return (
    <AlignmentEditor key={`${scope}:${props.id}:${props.image.sha256}`} {...props} scope={scope} />
  );
}

function AlignmentEditor({ id, image, url, title, attribution, scope }: Props & { scope: string }) {
  const scene = useScene();
  const ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const key = `living-world-land-draft:${encodeURIComponent(scope)}:alignment:${id}:${image.sha256}`;
  const [draft, setDraft] = useState(() => restore(key, title));
  const [open, setOpen] = useState(false);
  const [pixel, setPixel] = useState({ x: "", y: "" });
  const [location, setLocation] = useState({ longitude: "", latitude: "" });
  const [editing, setEditing] = useState<string | null>(null);
  const [result, setResult] = useState<Fit | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [storageError, setStorageError] = useState(false);
  const [saved, setSaved] = useState(false);
  const [offset, setOffset] = useState(0);
  const section = useLandContext((s) => s.section);
  const picker = useLandContext((s) => s.pointPicker);
  const panel = useUi((s) => s.activePanel);
  const visible = open && section === "discover" && panel === "land" && ready;
  const picking = picker === id;
  const mounted = useRef(true);
  const work = useRef<AbortController | null>(null);
  const pickTicket = useRef(0);
  const pickButton = useRef<HTMLButtonElement>(null);
  const layerId = `alignment-controls:${id}`;
  const queryKey = ["land-image-alignments", scope, id];
  const query = useQuery({
    queryKey: [...queryKey, offset],
    enabled: open && ready,
    queryFn: ({ signal }) =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}/image/registrations", {
          params: { path: { evidence_id: id }, query: { limit: 20, offset } },
          signal,
        }),
      ),
    retry: false,
  });
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      work.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (!visible) return;
    return () => {
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === id) {
        useLandContext.getState().setPointPicker(null);
        scene?.areas.cancelPick();
      }
    };
  }, [visible, id, scene]);
  useEffect(() => {
    if (!visible || !draft.points.length) return;
    useLandContext.getState().setLayer({
      id: layerId,
      title: "Map alignment control points",
      features: draft.points.map((point) => ({
        id: point.label,
        label: point.label,
        geometry: { type: "Point", coordinates: [point.longitude, point.latitude] },
      })),
    });
    return () => useLandContext.getState().removeLayer(layerId);
  }, [visible, draft.points, layerId]);

  const update = (change: Partial<Draft>) => {
    work.current?.abort();
    setBusy(false);
    setResult(null);
    setSaved(false);
    setError(null);
    const next = { ...draft, ...change, requestKey: crypto.randomUUID() };
    setDraft(next);
    try {
      if (next.points.length || next.notes || next.name !== title.slice(0, 200))
        localStorage.setItem(key, JSON.stringify(next));
      else localStorage.removeItem(key);
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  };
  const validPixel =
    pixel.x.trim() !== "" &&
    pixel.y.trim() !== "" &&
    Number.isFinite(+pixel.x) &&
    Number.isFinite(+pixel.y) &&
    +pixel.x >= 0 &&
    +pixel.x <= image.width &&
    +pixel.y >= 0 &&
    +pixel.y <= image.height;
  const addPoint = (longitude: number, latitude: number) => {
    if (
      !validPixel ||
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      Math.abs(longitude) > 180 ||
      Math.abs(latitude) > 90
    )
      return;
    const retained = draft.points.filter((point) => point.label !== editing);
    if (retained.length >= 30) {
      setError("Use at most 30 point matches.");
      return;
    }
    if (
      retained.some(
        (p) =>
          (p.imageX === +pixel.x && p.imageY === +pixel.y) ||
          (p.longitude === longitude && p.latitude === latitude),
      )
    ) {
      setError("Choose a distinct image point and map location for each match.");
      return;
    }
    let n = 1;
    while (retained.some((p) => p.label === `Point ${n}`)) n++;
    const point = {
      label: editing ?? `Point ${n}`,
      imageX: +pixel.x,
      imageY: +pixel.y,
      longitude,
      latitude,
    };
    update({
      points: editing
        ? draft.points.map((p) => (p.label === editing ? point : p))
        : [...draft.points, point],
    });
    setPixel({ x: "", y: "" });
    setLocation({ longitude: "", latitude: "" });
    setEditing(null);
  };
  const pick = async () => {
    if (!scene || !validPixel) return;
    const ticket = ++pickTicket.current;
    useUi.getState().setMeasureMode(null);
    useUi.getState().setExploreMode(false);
    // AreaEditor cancels the previous one-shot picker before taking ownership.
    const promise = scene.areas.pickGround();
    useLandContext.getState().setPointPicker(id);
    try {
      const point = await promise;
      if (!mounted.current || ticket !== pickTicket.current) return;
      if (point) addPoint(point.longitude, point.latitude);
    } finally {
      if (useLandContext.getState().pointPicker === id && ticket === pickTicket.current) {
        useLandContext.getState().setPointPicker(null);
        pickButton.current?.focus();
      }
    }
  };
  const execute = async (save: boolean) => {
    const controller = new AbortController();
    work.current?.abort();
    work.current = controller;
    setBusy(true);
    setError(null);
    setSaved(false);
    const body: Request = { ...draft, imageSha256: image.sha256 };
    try {
      if (save) {
        const response = await unwrap(
          api.POST("/api/v1/research/evidence/{evidence_id}/image/registrations", {
            params: { path: { evidence_id: id } },
            body,
            signal: controller.signal,
          }),
        );
        if (controller.signal.aborted) return;
        setResult(response.result);
        setSaved(true);
        setOffset(0);
        useLandContext.getState().setRaster({
          id: response.id,
          kind: "archive-alignment",
          band: 1,
          bounds: response.result.bounds,
          attribution,
          opacity: 0.6,
        });
        void cache.invalidateQueries({ queryKey });
      } else {
        const response = await unwrap(
          api.POST("/api/v1/research/evidence/{evidence_id}/image/registrations/preview", {
            params: { path: { evidence_id: id } },
            body,
            signal: controller.signal,
          }),
        );
        if (!controller.signal.aborted) setResult(response);
      }
    } catch (cause) {
      if (!controller.signal.aborted)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      if (mounted.current && !controller.signal.aborted) setBusy(false);
    }
  };
  const valid = draft.points.length >= 4 && draft.name.trim().length > 0 && !editing;
  return (
    <section className="land-alignment" aria-label="Historical map alignment">
      <div className="land-actions">
        <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "Close alignment editor" : "Align this map"}
          {!open && draft.points.length > 0 ? ` · ${draft.points.length} draft matches` : ""}
        </button>
      </div>
      {open && (
        <>
          <p>
            Match at least four recognizable places on this image and the globe. Spread them across
            the map; add more points to check the fit.
          </p>
          <p className="land-footnote">
            Coordinates use the saved {image.width} × {image.height} pixel image, measured from its
            top-left edge. The catalog footprint does not locate features within the image.
          </p>
          {storageError && (
            <p className="land-notice">
              This browser could not preserve your unfinished matches. Save an alignment before
              leaving.
            </p>
          )}
          <label>
            Alignment name
            <input
              disabled={picking}
              value={draft.name}
              maxLength={200}
              onChange={(e) => update({ name: e.target.value })}
            />
          </label>
          <div className="land-alignment-image-wrap">
            <button
              type="button"
              className="land-alignment-image"
              aria-label="Choose a point on the historical map"
              disabled={picking}
              onClick={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                setPixel(
                  event.detail === 0
                    ? { x: String(image.width / 2), y: String(image.height / 2) }
                    : {
                        x: (((event.clientX - box.left) / box.width) * image.width).toFixed(2),
                        y: (((event.clientY - box.top) / box.height) * image.height).toFixed(2),
                      },
                );
              }}
            >
              <img src={url} alt={title} draggable={false} />
              {draft.points.map((p) => (
                <span
                  key={p.label}
                  className="land-alignment-marker"
                  style={{
                    left: `${(p.imageX / image.width) * 100}%`,
                    top: `${(p.imageY / image.height) * 100}%`,
                  }}
                  aria-hidden="true"
                >
                  {p.label.replace("Point ", "")}
                </span>
              ))}
              {validPixel && (
                <span
                  className="land-alignment-marker is-pending"
                  style={{
                    left: `${(+pixel.x / image.width) * 100}%`,
                    top: `${(+pixel.y / image.height) * 100}%`,
                  }}
                  aria-hidden="true"
                >
                  +
                </span>
              )}
            </button>
          </div>
          <fieldset disabled={picking} className="land-alignment-fields">
            <legend>
              {editing ? `Edit ${editing}` : `New point match (${draft.points.length}/30)`}
            </legend>
            <label>
              Image X
              <input
                type="number"
                min={0}
                max={image.width}
                step="any"
                value={pixel.x}
                onChange={(e) => setPixel({ ...pixel, x: e.target.value })}
              />
            </label>
            <label>
              Image Y
              <input
                type="number"
                min={0}
                max={image.height}
                step="any"
                value={pixel.y}
                onChange={(e) => setPixel({ ...pixel, y: e.target.value })}
              />
            </label>
          </fieldset>
          <div className="land-actions">
            <button
              type="button"
              ref={pickButton}
              disabled={!validPixel || picking || !scene || !visible}
              onClick={() => void pick()}
            >
              Pick matching map point
            </button>
            {picking && (
              <button type="button" onClick={() => scene?.areas.cancelPick()}>
                Cancel map pick
              </button>
            )}
            {editing && (
              <button
                type="button"
                disabled={picking}
                onClick={() => {
                  setEditing(null);
                  setPixel({ x: "", y: "" });
                  setLocation({ longitude: "", latitude: "" });
                }}
              >
                Cancel point edit
              </button>
            )}
          </div>
          {picking && (
            <p role="status">Click the matching place on the globe. Press Escape to cancel.</p>
          )}
          <details>
            <summary>Enter map coordinates instead</summary>
            <fieldset className="land-alignment-fields" disabled={picking}>
              <legend>WGS84 coordinates</legend>
              <label>
                Longitude
                <input
                  type="number"
                  min={-180}
                  max={180}
                  step="any"
                  value={location.longitude}
                  onChange={(e) => setLocation({ ...location, longitude: e.target.value })}
                />
              </label>
              <label>
                Latitude
                <input
                  type="number"
                  min={-90}
                  max={90}
                  step="any"
                  value={location.latitude}
                  onChange={(e) => setLocation({ ...location, latitude: e.target.value })}
                />
              </label>
            </fieldset>
            <div className="land-actions">
              <button
                type="button"
                disabled={
                  !validPixel ||
                  picking ||
                  !location.longitude.trim() ||
                  !location.latitude.trim() ||
                  !Number.isFinite(+location.longitude) ||
                  !Number.isFinite(+location.latitude) ||
                  Math.abs(+location.longitude) > 180 ||
                  Math.abs(+location.latitude) > 90
                }
                onClick={() => addPoint(+location.longitude, +location.latitude)}
              >
                {editing ? "Update point match" : "Add point match"}
              </button>
            </div>
          </details>
          {draft.points.length > 0 && (
            <ol className="land-alignment-points" aria-label="Point matches">
              {draft.points.map((point) => (
                <li key={point.label}>
                  <strong>{point.label}</strong>
                  <span>
                    Image {point.imageX.toFixed(1)}, {point.imageY.toFixed(1)} ·{" "}
                    {point.latitude.toFixed(6)}°, {point.longitude.toFixed(6)}°
                  </span>
                  <div className="land-actions">
                    <button
                      type="button"
                      disabled={picking}
                      onClick={() => {
                        setEditing(point.label);
                        setPixel({ x: String(point.imageX), y: String(point.imageY) });
                        setLocation({
                          longitude: String(point.longitude),
                          latitude: String(point.latitude),
                        });
                      }}
                    >
                      Edit {point.label}
                    </button>
                    <button
                      type="button"
                      disabled={picking}
                      onClick={() => {
                        update({ points: draft.points.filter((p) => p.label !== point.label) });
                        if (editing === point.label) setEditing(null);
                      }}
                    >
                      Remove {point.label}
                    </button>
                  </div>
                </li>
              ))}
            </ol>
          )}
          <label>
            Alignment notes
            <textarea
              disabled={picking}
              value={draft.notes}
              maxLength={2000}
              placeholder="Places matched, uncertainties, or distortions…"
              onChange={(e) => update({ notes: e.target.value })}
            />
          </label>
          <div className="land-actions">
            <button
              type="button"
              disabled={!valid || busy || picking || !ready}
              onClick={() => void execute(false)}
            >
              {busy ? "Calculating alignment…" : "Check alignment"}
            </button>
            <button
              type="button"
              disabled={!valid || !result || busy || picking || !ready || !canEdit || saved}
              onClick={() => void execute(true)}
            >
              Save alignment and show overlay
            </button>
          </div>
          {!canEdit && <p className="land-footnote">A workspace editor can save this alignment.</p>}
          {error && (
            <p role="alert" className="land-error">
              {error}
            </p>
          )}
          {saved && (
            <p role="status">
              Alignment saved. The overlay is on the map; the source image and earlier alignments
              are preserved.
            </p>
          )}
          {result && <AlignmentFit result={result} />}
          <h5>Saved alignments</h5>
          {query.isPending && <p role="status">Loading saved alignments…</p>}
          {query.isError && (
            <p role="alert">
              Saved alignments could not load.{" "}
              <button type="button" onClick={() => void query.refetch()}>
                Retry alignments
              </button>
            </p>
          )}
          {query.data?.length === 0 && (
            <p className="land-footnote">No saved alignments on this page.</p>
          )}
          {query.data?.map((row) => (
            <SavedAlignment
              key={row.id}
              row={row}
              attribution={attribution}
              refiningDisabled={picking}
              onRefine={() => {
                update({
                  name: row.request.name,
                  notes: row.request.notes ?? "",
                  points: row.request.points,
                });
                setEditing(null);
                setPixel({ x: "", y: "" });
              }}
            />
          ))}
          <div className="land-actions">
            <button
              type="button"
              disabled={!offset || query.isFetching}
              onClick={() => setOffset(Math.max(0, offset - 20))}
            >
              Newer alignments
            </button>
            <button
              type="button"
              disabled={query.data?.length !== 20 || query.isFetching}
              onClick={() => setOffset(offset + 20)}
            >
              Older alignments
            </button>
          </div>
        </>
      )}
    </section>
  );
}
