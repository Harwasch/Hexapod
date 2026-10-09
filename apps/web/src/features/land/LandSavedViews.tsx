import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea } from "@twin/contracts";
import { ApiError, api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";
import { landScope, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useLand } from "@/state/land";
import { useViewer } from "@/state/viewer";
import {
  downloadViewCapture,
  parseViewCapture,
  serializeViewCapture,
  viewDraftKey,
} from "./landViewDraft";

type View = components["schemas"]["LandViewRead"];
type Create = components["schemas"]["LandViewCreate"];

export function LandSavedViews({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <SavedViews key={`${scope}/${land.id}`} land={land} scope={scope} />;
}
function SavedViews({ land, scope }: { land: LandArea; scope: string }) {
  const scene = useScene(),
    cache = useQueryClient(),
    canEdit = useLandCanEdit();
  const context = useLandContext();
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const [expanded, setExpanded] = useState(false),
    [name, setName] = useState("");
  const [offset, setOffset] = useState(0),
    [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Create | null>(null);
  const storageKey = viewDraftKey(scope, land.id);
  const [initialStorage] = useState(() => {
    try {
      return { raw: localStorage.getItem(storageKey), failed: false };
    } catch {
      return { raw: null, failed: true };
    }
  });
  const [recovery, setRecovery] = useState(initialStorage.raw);
  const [storageError, setStorageError] = useState(initialStorage.failed);
  const clearStored = (raw: string) => {
    try {
      if (localStorage.getItem(storageKey) === raw) localStorage.removeItem(storageKey);
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  };
  const [error, setError] = useState<string | null>(null),
    [notice, setNotice] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [renaming, setRenaming] = useState<View | null>(null),
    [rename, setRename] = useState("");
  const [removing, setRemoving] = useState<View | null>(null);
  const query = useQuery({
    queryKey: ["land-views", scope, land.id, offset],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/views", {
          params: { path: { land_id: land.id }, query: { offset, limit: 25 } },
        }),
      ),
    enabled: expanded,
    retry: false,
  });
  const run = async (operation: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await operation();
    } catch (cause) {
      if (active.current) setError(describeError(cause));
    } finally {
      if (active.current) setBusy(false);
    }
  };
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-views", scope, land.id] });
  const recover = async () => {
    if (!recovery) return;
    const capture = parseViewCapture(recovery, land.id);
    try {
      const saved = await unwrap(
        api.POST("/api/v1/land/{land_id}/views/recover", {
          params: { path: { land_id: land.id } },
          body: capture,
        }),
      );
      if (!active.current || landScope() !== scope) return;
      clearStored(recovery);
      setRecovery(null);
      setPending(null);
      setName("");
      setOffset(0);
      setNotice(`This capture was already saved as “${saved.name}”. No new view was created.`);
      await refresh();
    } catch (cause) {
      if (!active.current || landScope() !== scope) return;
      if (!(cause instanceof ApiError) || cause.status !== 404) throw cause;
      if (!canEdit)
        throw new Error(
          "This capture has not been saved. An editor can retry it; you can download it here.",
          { cause },
        );
      setPending(capture);
      setName(capture.name);
      setRecovery(null);
      setNotice("Recovered your captured view. Review it before retrying the save.");
    }
  };
  const skipped = Object.values(context.layers).filter(
    (layer) => !layer.researchArtifactId && layer.id !== "inventory" && layer.features.length > 0,
  );
  const picking = Boolean(context.pointPicker);
  const open = async (view: View) => {
    const result = await unwrap(
      api.GET("/api/v1/land/{land_id}/views/{view_id}", {
        params: { path: { land_id: land.id, view_id: view.id } },
      }),
    );
    if (!active.current || landScope() !== scope || useLand.getState().active?.id !== land.id)
      return;
    const current = useLandContext.getState();
    if (current.pointPicker) {
      setError("Finish the current map pick before opening a view.");
      return;
    }
    useLandContext.setState({
      selectedMapFeature: null,
      section: result.state.section,
      selectedInvestigationId: result.state.investigationId ?? null,
      selectedInventoryId: result.state.inventoryId ?? null,
      selectedSurveyId: result.state.surveyId ?? null,
      selectedSolarId: result.state.solarId ?? null,
      inventoryVisible: result.state.inventoryVisible ?? true,
      layers: {
        ...Object.fromEntries(
          Object.entries(current.layers).filter(([, layer]) => !layer.researchArtifactId),
        ),
        ...Object.fromEntries(
          result.maps.map((layer) => [
            layer.id,
            {
              id: layer.id,
              researchArtifactId: layer.id,
              unit: layer.unit,
              legend: layer.legend,
              title: layer.title,
              features: layer.features.map((feature, index) => ({ id: String(index), ...feature })),
            },
          ]),
        ),
      },
      rasters: Object.fromEntries(
        result.rasters.map((raster) => [
          raster.id,
          {
            id: raster.id,
            ...(raster.kind === "archive-alignment" ? { kind: "archive-alignment" as const } : {}),
            band: raster.band ?? 1,
            opacity: raster.opacity ?? 0.8,
            categorical: raster.categorical ?? false,
            bounds: raster.bounds,
            attribution: raster.attribution,
          },
        ]),
      ),
      rasterErrors: {},
    });
    const camera = result.state.camera;
    scene?.camera.flyTo(camera.longitude, camera.latitude, camera.height, {
      heading: camera.heading,
      pitch: camera.pitch,
      roll: camera.roll,
    });
    setWarnings(result.warnings);
    setNotice(`Opened “${result.view.name}”. Inventory details show their current saved records.`);
  };
  const save = async () => {
    const camera = useViewer.getState().camera;
    const state = useLandContext.getState();
    const payload: Create = pending ?? {
      name: name.trim(),
      requestKey: crypto.randomUUID(),
      state: {
        camera: {
          longitude: camera.longitude,
          latitude: camera.latitude,
          height: camera.height,
          heading: camera.heading,
          pitch: camera.pitch,
          roll: camera.roll,
        },
        boundaryRevision: land.revision,
        section: state.section,
        investigationId: state.selectedInvestigationId,
        inventoryId: state.selectedInventoryId,
        surveyId: state.selectedSurveyId,
        solarId: state.selectedSolarId,
        inventoryVisible: state.inventoryVisible,
        artifactIds: Object.values(state.layers).flatMap((layer) =>
          layer.researchArtifactId ? [layer.researchArtifactId] : [],
        ),
        rasters: Object.values(state.rasters).map((raster) => ({
          id: raster.id,
          kind: raster.kind ?? "raster",
          band: raster.band,
          opacity: raster.opacity,
        })),
      },
    };
    const raw = serializeViewCapture(land.id, payload);
    parseViewCapture(raw, land.id);
    setPending(payload);
    try {
      const previous = localStorage.getItem(storageKey);
      if (previous && previous !== raw) setStorageError(true);
      else {
        localStorage.setItem(storageKey, raw);
        setStorageError(false);
      }
    } catch {
      setStorageError(true);
    }
    const result = await unwrap(
      api.POST("/api/v1/land/{land_id}/views", {
        params: { path: { land_id: land.id } },
        body: payload,
      }),
    );
    if (!active.current || landScope() !== scope) return;
    clearStored(raw);
    setPending(null);
    setName("");
    setOffset(0);
    setNotice(`Saved “${result.name}” to this workspace.`);
    await refresh();
  };
  return (
    <details
      className="land-saved-views"
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>Saved exploration views{recovery ? " · unfinished save" : ""}</summary>
      {recovery && (
        <section aria-label="Recover captured view">
          <p>
            An unfinished view save is stored in this browser. Check whether it already reached the
            workspace before saving another view.
          </p>
          <div className="land-actions">
            <button type="button" disabled={busy} onClick={() => void run(recover)}>
              Recover view save
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => downloadViewCapture(recovery, land.id)}
            >
              Download captured view
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                clearStored(recovery);
                setRecovery(null);
              }}
            >
              Discard local view capture
            </button>
          </div>
        </section>
      )}
      {storageError && (
        <p role="alert">
          This browser could not preserve this capture, or another tab has an unfinished save.
          Download your capture before closing if the save fails.
        </p>
      )}
      <p>
        Return to this camera, workspace tab, research maps, imagery bands and selected records.
        Views are shared with this land’s workspace.
      </p>
      {canEdit && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void run(save);
          }}
        >
          <label className="land-name">
            View name
            <input
              value={name}
              maxLength={160}
              required
              disabled={busy || Boolean(pending) || Boolean(recovery)}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <p>
            Includes{" "}
            {pending
              ? (pending.state.artifactIds?.length ?? 0)
              : Object.values(context.layers).filter((layer) => layer.researchArtifactId)
                  .length}{" "}
            research maps and{" "}
            {pending ? (pending.state.rasters?.length ?? 0) : Object.keys(context.rasters).length}{" "}
            imagery layers. Inventory remains live.
          </p>
          {!pending && skipped.length > 0 && (
            <p>
              Temporary overlays are not saved: {skipped.map((layer) => layer.title).join(", ")}.
              Their underlying records and unfinished forms remain separate.
            </p>
          )}
          <div className="land-actions">
            <button
              type="submit"
              className="land-primary"
              disabled={
                busy || Boolean(recovery) || (!pending && (picking || !scene || !name.trim()))
              }
            >
              {pending ? "Retry captured view save" : "Save this view"}
            </button>
            {pending && (
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  clearStored(serializeViewCapture(land.id, pending));
                  setPending(null);
                }}
              >
                Capture a different view
              </button>
            )}
          </div>
          {pending && (
            <div className="land-actions">
              <button
                type="button"
                onClick={() => downloadViewCapture(serializeViewCapture(land.id, pending), land.id)}
              >
                Download captured view
              </button>
            </div>
          )}
          {pending && (
            <p>
              Captured boundary revision {pending.state.boundaryRevision} ·{" "}
              {pending.state.section ?? "discover"} · camera{" "}
              {pending.state.camera.latitude.toFixed(5)},{" "}
              {pending.state.camera.longitude.toFixed(5)}
            </p>
          )}
          {pending && (
            <p>
              A retry uses the exact captured view and cannot create a duplicate. Refresh the list
              to check an uncertain result.
            </p>
          )}
        </form>
      )}
      {picking && <p>Finish the current map pick before saving or opening a view.</p>}
      {query.isError && <p role="alert">Saved views could not be loaded.</p>}
      <div className="land-actions">
        <button type="button" disabled={busy} onClick={() => void query.refetch()}>
          Refresh saved views
        </button>
      </div>
      {query.isPending ? (
        <p>Loading views…</p>
      ) : !query.data?.length ? (
        <p>No saved views on this page.</p>
      ) : (
        <ul className="land-view-list">
          {query.data.map((view) => (
            <li key={view.id}>
              <strong>{view.name}</strong>
              <small>
                {new Date(view.createdAt).toLocaleString()} · boundary revision{" "}
                {view.state.boundaryRevision}
              </small>
              <div className="land-actions">
                <button
                  type="button"
                  disabled={busy || picking || !scene}
                  onClick={() => void run(() => open(view))}
                >
                  Open {view.name}
                </button>
                {canEdit && (
                  <>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => {
                        setRenaming(view);
                        setRename(view.name);
                      }}
                    >
                      Rename {view.name}
                    </button>
                    <button type="button" disabled={busy} onClick={() => setRemoving(view)}>
                      Delete {view.name}
                    </button>
                  </>
                )}
              </div>
              {renaming?.id === view.id && (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void run(async () => {
                      await unwrap(
                        api.PATCH("/api/v1/land/{land_id}/views/{view_id}", {
                          params: { path: { land_id: land.id, view_id: view.id } },
                          body: { name: rename, expectedRevision: renaming.revision },
                        }),
                      );
                      if (!active.current) return;
                      setRenaming(null);
                      await refresh();
                    });
                  }}
                >
                  <label className="land-name">
                    New view name
                    <input
                      value={rename}
                      required
                      maxLength={160}
                      onChange={(event) => setRename(event.target.value)}
                    />
                  </label>
                  <div className="land-actions">
                    <button disabled={busy || !rename.trim()}>Save name</button>
                    <button type="button" disabled={busy} onClick={() => setRenaming(null)}>
                      Cancel rename
                    </button>
                  </div>
                </form>
              )}
              {removing?.id === view.id && (
                <div>
                  <p>Delete this saved view? Its research, imagery and land records will remain.</p>
                  <div className="land-actions">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        void run(async () => {
                          await unwrap(
                            api.DELETE("/api/v1/land/{land_id}/views/{view_id}", {
                              params: {
                                path: { land_id: land.id, view_id: view.id },
                                query: { expected_revision: removing.revision },
                              },
                            }),
                          );
                          if (!active.current) return;
                          setRemoving(null);
                          await refresh();
                        })
                      }
                    >
                      Confirm view deletion
                    </button>
                    <button type="button" disabled={busy} onClick={() => setRemoving(null)}>
                      Keep view
                    </button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="land-actions">
        <button
          type="button"
          disabled={busy || offset === 0}
          onClick={() => setOffset(Math.max(0, offset - 25))}
        >
          Newer views
        </button>
        <span>Page {offset / 25 + 1}</span>
        <button
          type="button"
          disabled={busy || query.isPending || (query.data?.length ?? 0) < 25}
          onClick={() => setOffset(offset + 25)}
        >
          Older views
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {notice && <p role="status">{notice}</p>}
      {warnings.length > 0 && (
        <ul aria-label="Saved view notices">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </details>
  );
}
