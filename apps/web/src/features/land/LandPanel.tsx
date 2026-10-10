import {
  Check,
  ChevronRight,
  Download,
  Focus,
  History,
  LandPlot,
  MapPin,
  Pencil,
  Plus,
  Route,
  Undo2,
  Upload,
} from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { useQueryClient } from "@tanstack/react-query";

import type { Footprint, LandArea } from "@twin/contracts";
import { boundsOf, footprintAreaM2 } from "@twin/geo";

import { useScene } from "@/cesium/SceneContext";
import { ApiError, api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { fetchOsmContaining, OSM_ATTRIBUTION } from "@/missions/osm";
import { useLand, type LandMode } from "@/state/land";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";
import { WriteTokenField } from "../captures/WriteTokenField";
import { createCorridor, createLand, reviseLand, useBoundaryHistory, useLandAreas } from "./api";
import { LandDraftRecovery } from "./LandDraftRecovery";
import { LandBoundaryImport } from "./LandBoundaryImport";
import { LandBoundarySplit } from "./LandBoundarySplit";
import { BoundaryReferences } from "./BoundaryReferences";
import { drawingReferences } from "./boundarySources";
import "./land.css";
import { LandWorkspaceResizer } from "./LandWorkspaceResizer";
import { LandCandidatePicker } from "./LandCandidatePicker";
import { LandWorkspace } from "./LandWorkspace";
import { WorkspaceIdentity } from "./WorkspaceIdentity";
import { landUsesOidc, useLandScope } from "@/state/landIdentity";

function areaLabel(squareMetres: number): string {
  if (squareMetres < 4046.8564224) return `${Math.round(squareMetres).toLocaleString()} m²`;
  return `${(squareMetres / 4046.8564224).toLocaleString(undefined, { maximumFractionDigits: 1 })} acres`;
}

function landError(error: unknown): string {
  return error instanceof ApiError && error.fieldErrors.length
    ? error.fieldErrors.join(". ")
    : describeError(error);
}

export function LandPanel() {
  const scope = useLandScope();
  const open = useUi((s) => s.activePanel === "land");
  const tokenPrompt = useUi((s) => s.writeTokenPrompt);
  const state = useLand();
  const drawingGuides = useLandContext((s) => s.layers["drawing-guides"]);
  const scene = useScene();
  const queryClient = useQueryClient();
  const catalog = useLandAreas(open);
  const [panelWidth, setPanelWidth] = useState(440);
  const [sheetSize, setSheetSize] = useState<"compact" | "expanded">("compact");
  const pointPicker = useLandContext((s) => s.pointPicker);
  const [historyOpen, setHistoryOpen] = useState(false);
  const history = useBoundaryHistory(open && historyOpen ? (state.active?.id ?? null) : null);
  const [busy, setBusy] = useState(false);
  const combination = state.boundaryOperation;
  const setCombination = state.setBoundaryOperation;
  const { corridorWidth: width, corridorUnit: unit } = state;
  const fileInput = useRef<HTMLInputElement>(null);
  const operation = useRef(0);
  const { active, draft, points, mode } = state;
  const visible = draft ?? active;
  const area = draft ? footprintAreaM2(draft.boundary) : (active?.areaM2 ?? 0);

  const frame = (boundary: Footprint) => {
    const b = boundsOf(boundary);
    const dx = Math.max(0.0001, b.east - b.west);
    const dy = Math.max(0.0001, b.north - b.south);
    // Leave room for the land panel and for the outline around the subject.
    scene?.camera.flyToRectangle(
      Math.max(-180, b.west - dx * 0.8),
      Math.max(-90, b.south - dy * 0.35),
      Math.min(180, b.east + dx * 0.35),
      Math.min(90, b.north + dy * 0.35),
    );
  };
  const begin = (next: LandMode) => {
    operation.current += 1;
    setBusy(false);
    useUi.getState().setMeasureMode(null);
    useUi.getState().setExploreMode(false);
    scene?.areas.cancelPick();
    scene?.areas.edit(null);
    state.begin(next);
    if (next === "draw" || next === "corridor")
      state.setTraceSources(drawingReferences(drawingGuides?.drawingGuideSources ?? []));
  };

  useEffect(() => {
    if (!open || mode !== "pick" || points.length !== 1) return;
    const point = points[0];
    if (!point) return;
    let cancelled = false;
    void fetchOsmContaining({ longitude: point[0], latitude: point[1] })
      .then((feature) => {
        if (cancelled) return;
        if (!feature) {
          useLand.getState().begin("browse");
          useLand
            .getState()
            .setError("No mapped boundary found here. Draw the area or import a boundary.");
          return;
        }
        useLand.getState().propose({
          name: feature.name,
          description: "",
          boundary: feature.footprint,
          source: {
            method: "mapped-feature",
            meaning: "physical-feature",
            label: "OpenStreetMap feature",
            attribution: OSM_ATTRIBUTION,
            url: "https://www.openstreetmap.org/copyright",
          },
        });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        useLand.getState().begin("browse");
        useLand.getState().setError(`Map data could not be reached. ${describeError(error)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [open, mode, points]);

  const finish = async () => {
    const ticket = ++operation.current;
    state.setError(null);
    if (mode === "draw") {
      if (points.length < 3 || !points[0]) return;
      const boundary: Footprint = { type: "Polygon", coordinates: [[...points, points[0]]] };
      if (combination && draft) {
        setBusy(true);
        try {
          const result = await unwrap(
            api.POST("/api/v1/land/operations", {
              body: { operation: combination, left: draft.boundary, right: boundary },
            }),
          );
          if (operation.current !== ticket) return;
          const references = drawingReferences([
            ...(draft.source.references ?? []),
            ...state.traceSources,
          ]);
          state.updateBoundary(result.boundary);
          const updated = useLand.getState().draft;
          if (updated) state.updateDraft({ source: { ...updated.source, references } });
          state.begin("browse");
          setCombination(null);
        } catch (error) {
          if (operation.current === ticket) state.setError(landError(error));
        } finally {
          if (operation.current === ticket) setBusy(false);
        }
      } else {
        state.propose({
          name: active?.name ?? "Untitled land",
          description: active?.description ?? "",
          boundary,
          source: {
            method: "drawn",
            label: "Drawn on the map",
            meaning: "study-area",
            references: state.traceSources,
          },
        });
      }
      return;
    }
    setBusy(true);
    try {
      const result = await createCorridor(points, unit === "ft" ? width * 0.3048 : width);
      if (operation.current !== ticket) return;
      state.propose({
        name: active?.name ?? "Land corridor",
        description: active?.description ?? "",
        boundary: result.boundary,
        source: {
          method: "corridor",
          label: `${width} ${unit} total width, centered on the drawn line`,
          meaning: "study-area",
          references: state.traceSources,
        },
      });
    } catch (error) {
      if (operation.current === ticket) state.setError(landError(error));
    } finally {
      if (operation.current === ticket) setBusy(false);
    }
  };

  const save = async () => {
    if (!draft) return;
    state.begin("browse");
    const ticket = ++operation.current;
    setBusy(true);
    state.setError(null);
    try {
      const saved = active
        ? await reviseLand(active.id, {
            ...draft,
            expectedRevision: active.revision,
            note: "Boundary updated in the land workspace",
          })
        : await createLand(draft);
      if (operation.current === ticket) state.select(saved);
      await queryClient.invalidateQueries({ queryKey: ["land-areas"] });
      await queryClient.invalidateQueries({ queryKey: ["land-history", saved.id] });
    } catch (error) {
      if (operation.current === ticket) state.setError(landError(error));
    } finally {
      if (operation.current === ticket) setBusy(false);
    }
  };

  const choose = (land: LandArea) => {
    state.select(land);
    setHistoryOpen(false);
    frame(land.boundary);
  };
  const exportBoundary = () => {
    if (!visible) return;
    const feature = {
      type: "Feature",
      geometry: visible.boundary,
      properties: { name: visible.name, source: visible.source, revision: active?.revision },
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(feature, null, 2)], { type: "application/geo+json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "land-boundary.geojson";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <FloatingPanel
      open={open}
      title="Your land"
      wide
      className={`land-panel land-panel--${pointPicker ? "compact" : sheetSize}`}
      style={{ "--land-workspace-width": `${panelWidth}px` } as CSSProperties}
      actions={
        <button
          className="land-sheet-toggle"
          type="button"
          onClick={() => setSheetSize(sheetSize === "compact" ? "expanded" : "compact")}
        >
          {sheetSize === "compact" ? "Expand" : "Collapse"}
        </button>
      }
      testId="land-panel"
      onClose={() => {
        operation.current += 1;
        setBusy(false);
        state.begin("browse");
        useUi.getState().setPanel(null);
      }}
    >
      <LandWorkspaceResizer panelWidth={panelWidth} setPanelWidth={setPanelWidth} />
      <div className={`land-workspace ${active && !draft ? "land-workspace--saved" : ""}`}>
        <WorkspaceIdentity />
        <LandDraftRecovery key={`recovery:${scope}`} scope={scope} />
        {tokenPrompt && !landUsesOidc && (
          <WriteTokenField
            hint="This server requires its access token to load and save land areas."
            onSaved={() => {
              state.setError(null);
              void catalog.refetch();
              if (draft) void save();
            }}
          />
        )}
        {state.error && (
          <div className="land-error" role="alert">
            {state.error}
          </div>
        )}
        {catalog.isError && (
          <div className="land-notice">
            Saved land is unavailable. Your current boundary stays on the map.{" "}
            <button type="button" onClick={() => void catalog.refetch()}>
              Retry
            </button>
          </div>
        )}
        {!visible && mode === "browse" && (
          <>
            <div className="land-intro">
              <span className="land-eyebrow">A place of your own</span>
              <h3>Start with the land.</h3>
              <p>
                Outline the place you want to understand. A home, a field, an entire ranch, or a
                corridor connecting them.
              </p>
            </div>
            <div className="land-methods">
              <button type="button" onClick={() => begin("candidates")} disabled={!scene}>
                <LandPlot size={20} />
                <span>
                  <strong>Find parcels, lines or buildings</strong>
                  <small>Select mapped records and describe your boundary</small>
                </span>
                <ChevronRight size={16} />
              </button>
              <button type="button" onClick={() => begin("draw")} disabled={!scene}>
                <Pencil size={20} />
                <span>
                  <strong>Draw a boundary</strong>
                  <small>Click around your land on the map</small>
                </span>
                <ChevronRight size={16} />
              </button>
              <button type="button" onClick={() => begin("pick")} disabled={!scene}>
                <MapPin size={20} />
                <span>
                  <strong>Pick a mapped feature</strong>
                  <small>Fields, woodland, parks, and water</small>
                </span>
                <ChevronRight size={16} />
              </button>
              <button type="button" onClick={() => begin("corridor")} disabled={!scene}>
                <Route size={20} />
                <span>
                  <strong>Trace a corridor</strong>
                  <small>Follow a line and choose its width</small>
                </span>
                <ChevronRight size={16} />
              </button>
              <button type="button" onClick={() => fileInput.current?.click()}>
                <Upload size={20} />
                <span>
                  <strong>Import a boundary</strong>
                  <small>GeoJSON, KML, shapefile, or GeoPackage</small>
                </span>
                <ChevronRight size={16} />
              </button>
            </div>
            <p className="land-footnote">
              Mapped features describe what is on the ground. They are not property records.
            </p>
          </>
        )}

        <LandCandidatePicker key={state.session} />
        {(mode === "draw" || mode === "corridor" || mode === "split" || mode === "edit") && (
          <div>
            {visible && (
              <label className="land-snap-toggle">
                <input
                  type="checkbox"
                  checked={state.snapEnabled}
                  onChange={(event) => state.setSnapEnabled(event.target.checked)}
                />
                Snap to this boundary
                <small>
                  A gold ring marks an exact edge or corner within reach of the pointer.
                </small>
              </label>
            )}
            <label className="land-snap-toggle">
              <input
                type="checkbox"
                checked={state.snapMapped}
                onChange={(event) => state.setSnapMapped(event.target.checked)}
              />
              Snap to visible mapped features
              <small>
                Uses drawing guides, saved assets and research maps. Precision follows the source
                data.
              </small>
            </label>
            {state.snapTarget && (
              <p role="status" className="land-footnote">
                Snapping to {state.snapTarget}
              </p>
            )}
          </div>
        )}
        {drawingGuides && (
          <div className="land-notice">
            {drawingGuides.features.length} drawing guide
            {drawingGuides.features.length === 1 ? "" : "s"} on the map.
            <button
              type="button"
              onClick={() => useLandContext.getState().removeLayer("drawing-guides")}
            >
              Clear drawing guides
            </button>
            <BoundaryReferences sources={drawingGuides.drawingGuideSources ?? []} />
          </div>
        )}
        {mode === "split" && draft && <LandBoundarySplit key={`${scope}:${state.session}`} />}

        {(mode === "draw" || mode === "corridor" || mode === "pick") && (
          <div className="land-selection">
            <span className="land-eyebrow">Select your land</span>
            <h3>
              {mode === "pick"
                ? "Point to the place."
                : mode === "draw"
                  ? combination === "difference"
                    ? "Draw the area to exclude."
                    : combination === "intersection"
                      ? "Draw the area to retain."
                      : combination === "union"
                        ? "Draw another piece of land."
                        : "Trace the boundary."
                  : "Follow the centerline."}
            </h3>
            <p aria-live="polite">
              {mode === "pick"
                ? points.length
                  ? "Looking for a mapped boundary…"
                  : "Click a field, woodland, park, or body of water."
                : `${points.length} points placed. Click the map to add ${points.length ? "another" : "the first"} point.`}
            </p>
            {mode === "corridor" && (
              <div className="land-width">
                <label htmlFor="land-width">Total corridor width</label>
                <div>
                  <input
                    id="land-width"
                    type="number"
                    min="0.1"
                    max={unit === "ft" ? 328084 : 100000}
                    value={width}
                    onChange={(e) => state.setCorridorWidth(Number(e.target.value))}
                  />
                  <select
                    aria-label="Width units"
                    value={unit}
                    onChange={(e) => state.setCorridorUnit(e.target.value === "m" ? "m" : "ft")}
                  >
                    <option value="ft">feet</option>
                    <option value="m">meters</option>
                  </select>
                </div>
                <small>Half the width on each side of the line.</small>
              </div>
            )}
            <div className="land-actions">
              {mode !== "pick" && (
                <button
                  type="button"
                  className="land-primary"
                  disabled={
                    busy ||
                    points.length < (mode === "draw" ? 3 : 2) ||
                    (mode === "corridor" && (!Number.isFinite(width) || width <= 0))
                  }
                  onClick={() => void finish()}
                >
                  <Check size={16} />
                  {busy ? "Building corridor…" : "Review boundary"}
                </button>
              )}
              {mode !== "pick" && (
                <button type="button" disabled={!points.length || busy} onClick={state.removePoint}>
                  <Undo2 size={16} />
                  Last point
                </button>
              )}
              <button
                type="button"
                onClick={() => {
                  setCombination(null);
                  begin("browse");
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {visible && (mode === "browse" || mode === "edit") && (
          <>
            <div className="land-place-heading">
              <span className="land-eyebrow">
                {draft ? "Review your boundary" : "Land overview"}
              </span>
              <button
                type="button"
                aria-label="Show boundary on map"
                onClick={() => frame(visible.boundary)}
              >
                <Focus size={18} />
              </button>
            </div>
            {draft ? (
              <label className="land-name">
                Name this land
                <input
                  autoComplete="off"
                  maxLength={200}
                  value={draft.name}
                  onChange={(e) => state.updateDraft({ name: e.target.value })}
                />
              </label>
            ) : (
              <h3 className="land-title">{visible.name}</h3>
            )}
            <div className="land-metrics">
              <div>
                <strong>
                  {draft ? "≈ " : ""}
                  {areaLabel(area)}
                </strong>
                <span>selected area</span>
              </div>
              {active && !draft && (
                <div>
                  <strong>
                    {(active.perimeterM / 1000).toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}{" "}
                    km
                  </strong>
                  <span>boundary length</span>
                </div>
              )}
            </div>
            <details className="land-provenance" open={Boolean(draft)}>
              <summary>Boundary source and meaning</summary>
              <div className="land-source">
                <LandPlot size={16} />
                <div>
                  <strong>{visible.source.label}</strong>
                  <span>
                    {visible.source.meaning === "recorded-parcel"
                      ? "Recorded parcel boundary"
                      : visible.source.meaning === "physical-feature"
                        ? "Physical feature · not a property boundary"
                        : "Study area · not a verified property boundary"}
                  </span>
                  {visible.source.attribution && <small>{visible.source.attribution}</small>}
                </div>
              </div>
              <BoundaryReferences sources={visible.source.references ?? []} />
            </details>
            {draft ? (
              <>
                <label className="land-name">
                  Notes
                  <textarea
                    rows={2}
                    maxLength={5000}
                    value={draft.description ?? ""}
                    onChange={(e) => state.updateDraft({ description: e.target.value })}
                    placeholder="What would you like to understand about this place?"
                  />
                </label>
                {mode === "edit" && (
                  <p className="land-notice">
                    Drag the white points to adjust the boundary. Exclusions move independently.
                    Press Escape when finished.
                  </p>
                )}
                <div className="land-actions" aria-label="Combine boundary shapes">
                  <button type="button" disabled={!scene || busy} onClick={() => begin("split")}>
                    Split with a line
                  </button>
                  {(
                    [
                      ["union", "Add an area"],
                      ["difference", "Draw an exclusion"],
                      ["intersection", "Keep an intersection"],
                    ] as const
                  ).map(([operation, label]) => (
                    <button
                      key={operation}
                      type="button"
                      disabled={!scene || busy}
                      onClick={() => {
                        setCombination(operation);
                        begin("draw");
                      }}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="land-actions">
                  <button
                    className="land-primary"
                    type="button"
                    disabled={busy || !draft.name.trim()}
                    onClick={() => void save()}
                  >
                    <Check size={16} />
                    {busy ? "Saving…" : active ? "Save revision" : "Save this land"}
                  </button>
                  <button
                    type="button"
                    disabled={!scene || busy}
                    onClick={() => begin(mode === "edit" ? "browse" : "edit")}
                  >
                    <Pencil size={16} />
                    {mode === "edit" ? "Finish editing" : "Adjust"}
                  </button>
                  <button type="button" disabled={busy} onClick={state.cancel}>
                    Discard
                  </button>
                </div>
              </>
            ) : (
              <>
                {active?.description && <p className="land-description">{active.description}</p>}
                <div className="land-actions">
                  <button
                    type="button"
                    onClick={() => {
                      if (!active) return;
                      state.propose({
                        name: active.name,
                        description: active.description,
                        boundary: active.boundary,
                        source: active.source,
                      });
                      begin("edit");
                    }}
                  >
                    <Pencil size={16} />
                    Edit boundary
                  </button>
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(!historyOpen)}
                    aria-expanded={historyOpen}
                  >
                    <History size={16} />
                    History
                  </button>
                  <button type="button" onClick={exportBoundary}>
                    <Download size={16} />
                    Export
                  </button>
                </div>
                {historyOpen && (
                  <div className="land-history">
                    <h4>Boundary history</h4>
                    {history.isPending && <p>Loading revisions…</p>}
                    {history.isError && <p role="alert">History could not be loaded.</p>}
                    {history.data?.map((revision) => (
                      <div key={revision.revision}>
                        <strong>Revision {revision.revision}</strong>
                        <span>{revision.note}</span>
                        <small>{new Date(revision.createdAt).toLocaleString()}</small>
                        <button
                          type="button"
                          onClick={() => {
                            if (!active) return;
                            state.propose({
                              name: active.name,
                              description: active.description,
                              boundary: revision.boundary,
                              source: revision.source,
                            });
                          }}
                        >
                          Use as new draft
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <button
                  type="button"
                  className="land-back"
                  onClick={() => {
                    state.clear();
                    setHistoryOpen(false);
                  }}
                >
                  <Plus size={16} />
                  Select another area
                </button>
              </>
            )}
          </>
        )}

        {active && !draft && mode === "browse" && (
          <>
            <LandWorkspace key={`workspace:${scope}:${active.id}`} land={active} />
          </>
        )}

        {!draft && mode === "browse" && (
          <div className="land-library">
            <h4>
              Saved land <span>{catalog.data?.length ?? ""}</span>
            </h4>
            {catalog.isPending && <p className="land-footnote">Loading saved areas…</p>}
            {catalog.data?.map((land) => (
              <button
                key={land.id}
                type="button"
                className={active?.id === land.id ? "is-selected" : ""}
                onClick={() => choose(land)}
              >
                <LandPlot size={17} />
                <span>
                  <strong>{land.name}</strong>
                  <small>
                    {areaLabel(land.areaM2)} · revision {land.revision}
                  </small>
                </span>
                <ChevronRight size={15} />
              </button>
            ))}
            {catalog.data?.length === 0 && (
              <p className="land-footnote">
                Your saved areas will stay here, independently of plans and captures.
              </p>
            )}
          </div>
        )}
        {open && <LandBoundaryImport key={`import:${scope}`} inputRef={fileInput} frame={frame} />}
      </div>
    </FloatingPanel>
  );
}
