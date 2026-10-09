import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea, LandCandidate, LandMapGeometry } from "@twin/contracts";
import { boundsOf } from "@twin/geo";
import { api, unwrap, ApiError } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandScope, useLandCanEdit, useLandAccessReady } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { useSelection } from "@/state/selection";
import { useUi } from "@/state/ui";
import { InventoryGeometryEditor } from "./InventoryGeometryEditor";

import { InventoryConflictReview } from "./InventoryConflictReview";
import {
  downloadAssetDraft,
  parseInventoryDraft,
  sameAssetContent,
  type InventoryDraftSnapshot,
} from "./inventoryDraft";
import type { InventoryShape } from "./inventoryGeometry";
import { InventoryImport } from "./InventoryImport";
import { LandInspectionForm } from "./LandInspectionForm";

type Feature = components["schemas"]["LandFeatureRead"];
type FeatureDraft = components["schemas"]["LandFeatureCreate"];

export function LandInventory({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <Inventory key={`${scope}:${land.id}`} land={land} scope={scope} />;
}
function Inventory({ land, scope }: { land: LandArea; scope: string }) {
  const canEdit = useLandCanEdit(),
    ready = useLandAccessReady();
  const cache = useQueryClient(),
    scene = useScene();
  const inspected = useSelection((state) => state.selection);
  const selectedId = useLandContext((state) => state.selectedInventoryId);
  const section = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(true);
  const [offset, setOffset] = useState(0);
  const [draft, setDraft] = useState<FeatureDraft | null>(null);
  const [editing, setEditing] = useState<Feature | null>(null);
  const [geometryEditing, setGeometryEditing] = useState(false);
  const [revisionNote, setRevisionNote] = useState("");
  const [working, setWorking] = useState<InventoryShape | null>(null);
  const [concurrent, setConcurrent] = useState<Feature | null>(null);
  const storage = `living-world-land-draft:${encodeURIComponent(scope)}:inventory:${land.id}`;
  const [recovery, setRecovery] = useState<string | null>(() => {
    try {
      return localStorage.getItem(storage);
    } catch {
      return null;
    }
  });
  const [storageError, setStorageError] = useState(false);
  useEffect(() => {
    if (!draft) return;
    const value: InventoryDraftSnapshot = {
      version: 1,
      landId: land.id,
      boundaryRevision: land.revision,
      draft,
      editing,
      revisionNote,
      working: geometryEditing ? working : null,
    };
    let failed = false,
      cancelled = false;
    try {
      localStorage.setItem(storage, JSON.stringify(value));
    } catch {
      failed = true;
    }
    queueMicrotask(() => {
      if (!cancelled) setStorageError(failed);
    });
    return () => {
      cancelled = true;
    };
  }, [draft, editing, revisionNote, working, geometryEditing, storage, land.id, land.revision]);
  function clearRecovery() {
    setRecovery(null);
    setNotice(null);
    try {
      localStorage.removeItem(storage);
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  }

  const activeSection = useLandContext((state) => state.section);
  const panel = useUi((state) => state.activePanel);
  const picker = useLandContext((state) => state.pointPicker);
  const pickOwner = `inventory-new:${land.id}`;
  const pickTicket = useRef(0);
  const active = ready && canEdit && activeSection === "inventory" && panel === "land";
  useEffect(() => {
    if (!active) return;
    return () => {
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === pickOwner) {
        scene?.areas.cancelPick();
        useLandContext.getState().setPointPicker(null);
      }
    };
  }, [active, scene, pickOwner]);
  const [candidates, setCandidates] = useState<LandCandidate[]>([]);
  const [lookupMessage, setLookupMessage] = useState("");
  const [lookupKind, setLookupKind] = useState<"line" | "building">("building");
  const [useInspected, setUseInspected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ticket = useRef(0);
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );
  const key = ["land-inventory", scope, land.id, land.revision, offset];
  const catalog = useQuery({
    queryKey: key,
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features", {
          params: { path: { land_id: land.id }, query: { limit: 100, offset } },
        }),
      ),
    retry: false,
  });
  const selectedRecord = useQuery({
    queryKey: ["land-inventory-record", scope, land.id, land.revision, selectedId],
    enabled: ready && Boolean(selectedId),
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}", {
          params: { path: { land_id: land.id, feature_id: selectedId ?? "" } },
        }),
      ),
  });
  const selected =
    selectedRecord.data ?? catalog.data?.find((feature) => feature.id === selectedId) ?? null;
  const inspections = useQuery({
    queryKey: ["land-inspections", scope, land.id, selectedId],
    enabled: ready && Boolean(selectedId),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}/inspections", {
          params: { path: { land_id: land.id, feature_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  const history = useQuery({
    queryKey: ["land-feature-history", scope, land.id, selectedId, selected?.revision],
    enabled: ready && Boolean(selectedId),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}/revisions", {
          params: { path: { land_id: land.id, feature_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  useEffect(() => {
    if (!catalog.data) return;
    if (!visible) {
      useLandContext.getState().removeLayer("inventory");
      return;
    }
    useLandContext.getState().setLayer({
      id: "inventory",
      title: "Land inventory",
      selectedIds: selectedId ? [selectedId] : [],
      features: catalog.data
        .filter((feature) => feature.status !== "retired")
        .map((feature) => ({ id: feature.id, label: feature.name, geometry: feature.geometry })),
    });
  }, [catalog.data, visible, selectedId]);
  useEffect(() => {
    if (selectedId) section.current?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);
  useEffect(() => {
    if (draft)
      useLandContext.getState().setLayer({
        id: "inventory-draft",
        title: "Feature under review",
        selectedIds: ["draft"],
        features: [{ id: "draft", label: draft.name, geometry: draft.geometry }],
      });
    else useLandContext.getState().removeLayer("inventory-draft");
    return () => useLandContext.getState().removeLayer("inventory-draft");
  }, [draft]);
  const refresh = () => cache.invalidateQueries({ queryKey: ["land-inventory", scope, land.id] });
  const frame = (geometry: LandMapGeometry) => {
    const coordinates =
      geometry.type === "Point"
        ? [geometry.coordinates]
        : geometry.type === "LineString"
          ? geometry.coordinates
          : null;
    const bounds = coordinates
      ? {
          west: Math.min(...coordinates.map((point) => point[0] ?? 0)),
          east: Math.max(...coordinates.map((point) => point[0] ?? 0)),
          south: Math.min(...coordinates.map((point) => point[1] ?? 0)),
          north: Math.max(...coordinates.map((point) => point[1] ?? 0)),
        }
      : boundsOf(geometry as LandArea["boundary"]);
    scene?.camera.flyToRectangle(
      bounds.west - 0.001,
      bounds.south - 0.001,
      bounds.east + 0.001,
      bounds.north + 0.001,
    );
  };
  const fromCandidate = (candidate: LandCandidate) => {
    setEditing(null);
    setGeometryEditing(false);
    setRevisionNote("");
    setDraft({
      requestKey: crypto.randomUUID(),
      name: candidate.label,
      category: lookupKind === "building" ? "building" : "other",
      geometry: candidate.geometry,
      source: candidate.source,
      status: "candidate",
      attributes: Object.fromEntries(
        Object.entries(candidate.properties ?? {}).map(([key, value]) => [key, String(value)]),
      ),
    });
    frame(candidate.geometry);
  };
  const lookup = async () => {
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    const b = boundsOf(land.boundary);
    const point =
      useInspected && inspected
        ? [inspected.longitude, inspected.latitude]
        : [(b.west + b.east) / 2, (b.south + b.north) / 2];
    try {
      const response = await unwrap(
        api.POST("/api/v1/land/selection/candidates", {
          body: { kind: lookupKind, point: { type: "Point", coordinates: point }, radiusM: 2000 },
        }),
      );
      if (ticket.current === current) {
        setCandidates(response.candidates);
        setLookupMessage(
          `${response.message}${response.truncated ? " More records may exist; inspect a closer point to narrow the search." : ""}`,
        );
      }
    } catch (cause) {
      if (ticket.current === current) setError(describeError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };
  const placeAsset = async () => {
    if (!scene || !active || draft) return;
    const current = ++pickTicket.current;
    useLandContext.getState().setPointPicker(pickOwner);
    const point = await scene.areas.pickGround();
    if (current !== pickTicket.current || useLandContext.getState().pointPicker !== pickOwner)
      return;
    useLandContext.getState().setPointPicker(null);
    if (!point) return;
    setEditing(null);
    setGeometryEditing(false);
    setRevisionNote("");
    setDraft({
      requestKey: crypto.randomUUID(),
      name: "",
      category: "other",
      status: "candidate",
      geometry: { type: "Point", coordinates: [point.longitude, point.latitude] },
      source: { method: "drawn", meaning: "physical-feature", label: "User-picked map location" },
    });
  };
  const recover = async () => {
    if (!recovery || !canEdit) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const value = parseInventoryDraft(recovery, land.id);
      let baseline = value.editing;
      let latest: Feature | null = null;
      let creationRecorded = false;
      if (baseline)
        latest = await unwrap(
          api.GET("/api/v1/land/{land_id}/features/{feature_id}", {
            params: { path: { land_id: land.id, feature_id: baseline.id } },
          }),
        );
      else {
        try {
          const saved = await unwrap(
            api.GET("/api/v1/land/{land_id}/features/requests/{request_key}", {
              params: { path: { land_id: land.id, request_key: value.draft.requestKey ?? "" } },
            }),
          );
          latest = saved.current;
          creationRecorded = sameAssetContent(saved.original, value.draft);
          baseline = { ...saved.current, ...saved.original, revision: 1 };
        } catch (cause) {
          if (!(cause instanceof ApiError && cause.status === 404)) throw cause;
        }
      }
      if (current !== ticket.current) return;
      if (latest)
        cache.setQueryData(
          ["land-inventory-record", scope, land.id, land.revision, latest.id],
          latest,
        );
      if (
        latest &&
        !value.working &&
        (creationRecorded ||
          (latest.requestKey === value.draft.requestKey && sameAssetContent(latest, value.draft)))
      ) {
        clearRecovery();
        useLandContext.getState().selectInventory(latest.id);
        setNotice(
          "This draft was already saved. Opened the saved asset without creating another revision.",
        );
        return;
      }
      setDraft(value.draft);
      setEditing(baseline);
      setRevisionNote(value.revisionNote);
      setWorking(value.working);
      setGeometryEditing(Boolean(value.working));
      setConcurrent(latest && latest.revision !== baseline?.revision ? latest : null);
      setRecovery(null);
      if (latest) useLandContext.getState().selectInventory(latest.id);
      if (value.boundaryRevision !== land.revision)
        setError(
          "Recovered against a newer land boundary. Review the location and any geometry changes before saving.",
        );
      frame(value.draft.geometry);
    } catch (cause) {
      if (current === ticket.current) setError(describeError(cause));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  };
  const checkLatest = async () => {
    if (!editing) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const latest = await unwrap(
        api.GET("/api/v1/land/{land_id}/features/{feature_id}", {
          params: { path: { land_id: land.id, feature_id: editing.id } },
        }),
      );
      if (current === ticket.current) {
        cache.setQueryData(
          ["land-inventory-record", scope, land.id, land.revision, latest.id],
          latest,
        );
        void refresh();
        setConcurrent(latest.revision !== editing.revision ? latest : null);
        if (latest.revision === editing.revision)
          setError("This draft already uses the latest saved revision.");
      }
    } catch (cause) {
      if (current === ticket.current) setError(describeError(cause));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  };
  const save = async () => {
    if (!draft || geometryEditing || concurrent || !canEdit || busy) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const feature = editing
        ? await unwrap(
            api.PUT("/api/v1/land/{land_id}/features/{feature_id}", {
              params: { path: { land_id: land.id, feature_id: editing.id } },
              body: {
                ...draft,
                expectedRevision: editing.revision,
                note: revisionNote.trim(),
              },
            }),
          )
        : await unwrap(
            api.POST("/api/v1/land/{land_id}/features", {
              params: { path: { land_id: land.id } },
              body: draft,
            }),
          );
      if (ticket.current === current) {
        clearRecovery();
        setWorking(null);
        setConcurrent(null);
        setDraft(null);
        setEditing(null);
        useLandContext.getState().selectInventory(feature.id);
      }
      await refresh();
      await cache.invalidateQueries({ queryKey: ["land-inventory-record", scope, land.id] });
    } catch (cause) {
      if (ticket.current === current)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };
  return (
    <section ref={section} className="land-inventory" aria-label="Land inventory">
      <span className="land-eyebrow">Know what is here</span>
      <h3>Assets and features.</h3>
      <p>
        Record buildings, infrastructure and field features. Confirm their identities and keep dated
        inspections alongside the map.
      </p>
      {error && (
        <p className="land-error" role="alert">
          {error}
        </p>
      )}
      {!draft && notice && <p role="status">{notice}</p>}
      {storageError && (
        <p role="alert">
          This browser could not update the asset recovery copy. Keep this page open or download the
          draft.
        </p>
      )}
      {recovery && canEdit && (
        <div className="land-actions">
          <p>An unfinished asset draft is available in this browser.</p>
          <button type="button" disabled={busy} onClick={() => void recover()}>
            Recover asset draft
          </button>
          <button type="button" onClick={() => downloadAssetDraft(recovery)}>
            Download saved asset draft
          </button>
          <button type="button" disabled={busy} onClick={clearRecovery}>
            Discard saved asset draft
          </button>
        </div>
      )}
      {catalog.isError && (
        <p role="alert">
          Inventory could not be loaded.{" "}
          <button type="button" onClick={() => void catalog.refetch()}>
            Retry
          </button>
        </p>
      )}
      <label className="land-dismissed">
        <input
          type="checkbox"
          checked={visible}
          onChange={(event) => setVisible(event.target.checked)}
        />{" "}
        Show inventory on map
      </label>
      {selectedRecord.isError && (
        <p role="alert">
          The selected feature could not be loaded.{" "}
          <button type="button" onClick={() => void selectedRecord.refetch()}>
            Retry selected feature
          </button>
        </p>
      )}
      {canEdit && !draft && !recovery && (
        <div className="land-actions">
          <button
            type="button"
            disabled={!scene || !active || busy}
            onClick={() => void placeAsset()}
          >
            Place new asset on map
          </button>
        </div>
      )}
      {picker === pickOwner && (
        <p role="status">
          Click the map to place the asset.{" "}
          <button type="button" onClick={() => scene?.areas.cancelPick()}>
            Cancel asset placement
          </button>
        </p>
      )}
      {!draft && !recovery && <InventoryImport land={land} />}
      {canEdit && !draft && !recovery && (
        <details>
          <summary>Add a mapped or inspected feature</summary>
          <label className="land-name">
            Mapped feature type
            <select
              value={lookupKind}
              onChange={(event) => setLookupKind(event.target.value as typeof lookupKind)}
            >
              <option value="building">Buildings</option>
              <option value="line">Lines, roads and waterways</option>
            </select>
          </label>
          <label className="land-dismissed">
            <input
              type="checkbox"
              checked={useInspected}
              disabled={!inspected}
              onChange={(event) => setUseInspected(event.target.checked)}
            />{" "}
            Search near the inspected map location
          </label>
          <div className="land-actions">
            <button type="button" disabled={busy} onClick={() => void lookup()}>
              Find mapped features within 2 km
            </button>
            <button
              type="button"
              disabled={!inspected || busy}
              onClick={() => {
                if (!inspected) return;
                setEditing(null);
                setGeometryEditing(false);
                setRevisionNote("");
                setDraft({
                  requestKey: crypto.randomUUID(),
                  name: inspected.title,
                  category: "other",
                  status: "candidate",
                  geometry: {
                    type: "Point",
                    coordinates: [inspected.longitude, inspected.latitude],
                  },
                  source: {
                    method: "drawn",
                    meaning: "physical-feature",
                    label: "Inspected map location",
                  },
                });
              }}
            >
              Record inspected point
            </button>
          </div>
          <p className="land-footnote">
            By default, lookup searches near the land's center. Inspect another location on the map
            to search there.
          </p>
          {lookupMessage && <p role="status">{lookupMessage}</p>}
          <div className="land-candidates">
            {candidates.map((candidate) => (
              <button key={candidate.id} type="button" onClick={() => fromCandidate(candidate)}>
                <strong>{candidate.label}</strong>
                <span>
                  {candidate.source.label} · {Math.round(candidate.distanceM)} m from search point
                </span>
              </button>
            ))}
          </div>
        </details>
      )}
      {draft && canEdit && (
        <form
          className="land-inventory-form"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <fieldset disabled={busy} className="land-inventory-inputs">
            <label className="land-name">
              Feature name
              <input
                required
                maxLength={200}
                value={draft.name}
                onChange={(event) => setDraft({ ...draft, name: event.target.value })}
              />
            </label>
            <label className="land-name">
              Category
              <select
                value={draft.category}
                onChange={(event) =>
                  setDraft({ ...draft, category: event.target.value as FeatureDraft["category"] })
                }
              >
                {[
                  "building",
                  "power",
                  "water",
                  "transport",
                  "equipment",
                  "vegetation",
                  "other",
                ].map((kind) => (
                  <option key={kind}>{kind}</option>
                ))}
              </select>
            </label>
            <label className="land-name">
              Identity status
              <select
                value={draft.status ?? "candidate"}
                onChange={(event) =>
                  setDraft({ ...draft, status: event.target.value as FeatureDraft["status"] })
                }
              >
                <option value="candidate">Candidate · needs verification</option>
                <option value="confirmed">Confirmed identity</option>
                <option value="retired">Retired</option>
              </select>
            </label>
            <label className="land-name">
              Feature notes
              <textarea
                rows={3}
                value={draft.description ?? ""}
                onChange={(event) => setDraft({ ...draft, description: event.target.value })}
              />
            </label>
            <p className="land-footnote">
              Source: {draft.source.label}. Identity confirmation does not establish ownership.
            </p>
            {geometryEditing ? (
              <InventoryGeometryEditor
                key={draft.requestKey}
                landId={land.id}
                boundaryRevision={land.revision}
                geometry={draft.geometry}
                initialShape={working}
                onDraft={setWorking}
                onApply={(geometry) => {
                  setDraft({ ...draft, geometry });
                  setGeometryEditing(false);
                  setWorking(null);
                }}
                onCancel={() => {
                  setGeometryEditing(false);
                  setWorking(null);
                }}
              />
            ) : (
              <button
                type="button"
                onClick={() => {
                  setWorking(null);
                  setGeometryEditing(true);
                }}
              >
                Edit shape and location
              </button>
            )}
            {concurrent && editing && (
              <InventoryConflictReview
                key={concurrent.revision}
                baseline={editing}
                draft={draft}
                current={concurrent}
                working={Boolean(working)}
                onResolve={(merged, keepWorking) => {
                  setDraft(merged);
                  setEditing(concurrent);
                  setConcurrent(null);
                  setError(null);
                  if (!keepWorking) {
                    setWorking(null);
                    setGeometryEditing(false);
                  }
                }}
              />
            )}
            {editing && (
              <button type="button" onClick={() => void checkLatest()}>
                Check for newer asset revision
              </button>
            )}
            {editing && (
              <label className="land-name">
                Revision note
                <textarea
                  required
                  maxLength={1000}
                  value={revisionNote}
                  onChange={(event) => setRevisionNote(event.target.value)}
                  placeholder="Describe what changed and how you checked it"
                />
              </label>
            )}
            {geometryEditing && (
              <p className="land-footnote">
                Apply or discard the geometry edits before saving this feature.
              </p>
            )}
            <div className="land-actions">
              <button
                disabled={
                  busy ||
                  geometryEditing ||
                  Boolean(concurrent) ||
                  (Boolean(editing) && !revisionNote.trim())
                }
                type="submit"
              >
                {editing ? "Save feature revision" : "Add to inventory"}
              </button>
              <button
                type="button"
                onClick={() => {
                  ticket.current++;
                  setBusy(false);
                  clearRecovery();
                  setWorking(null);
                  setConcurrent(null);
                  setDraft(null);
                  setEditing(null);
                  setGeometryEditing(false);
                }}
              >
                Discard asset draft
              </button>
              <button
                type="button"
                onClick={() =>
                  downloadAssetDraft({
                    version: 1,
                    landId: land.id,
                    boundaryRevision: land.revision,
                    draft,
                    editing,
                    revisionNote,
                    working,
                  })
                }
              >
                Download asset draft
              </button>
            </div>
          </fieldset>
        </form>
      )}
      <div className="land-candidates">
        {catalog.data?.map((feature) => (
          <button
            key={feature.id}
            type="button"
            aria-pressed={selectedId === feature.id}
            onClick={() => {
              useLandContext.getState().selectInventory(feature.id);
              frame(feature.geometry);
            }}
          >
            <strong>{feature.name}</strong>
            <span>
              {feature.category} · {feature.status}
              {!feature.intersectsLand
                ? ` · ${Math.round(feature.distanceM)} m outside the current boundary`
                : ""}
            </span>
          </button>
        ))}
      </div>
      {(offset > 0 || catalog.data?.length === 100) && (
        <div className="land-actions">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => {
              useLandContext.getState().selectInventory(null);
              setOffset(Math.max(0, offset - 100));
            }}
          >
            Previous features
          </button>
          <button
            type="button"
            disabled={catalog.data?.length !== 100}
            onClick={() => {
              useLandContext.getState().selectInventory(null);
              setOffset(offset + 100);
            }}
          >
            More features
          </button>
        </div>
      )}
      {selected && (
        <article className="land-inventory-record">
          <h4>{selected.name}</h4>
          <p>{selected.description}</p>
          <p className="land-footnote">
            Revision {selected.revision} · source: {selected.source.label}
          </p>
          {selected.externalRef && (
            <p className="land-footnote">
              Dataset: {selected.externalRef.namespace} · record: {selected.externalRef.recordId}
            </p>
          )}
          {canEdit && !draft && !recovery && (
            <button
              type="button"
              onClick={() => {
                setEditing(selected);
                setRevisionNote("");
                setGeometryEditing(false);
                setDraft({
                  requestKey: crypto.randomUUID(),
                  name: selected.name,
                  category: selected.category,
                  geometry: selected.geometry,
                  source: selected.source,
                  status: selected.status,
                  description: selected.description,
                  attributes: selected.attributes,
                  evidenceIds: selected.evidenceIds,
                  externalRef: selected.externalRef,
                });
              }}
            >
              Edit feature details and geometry
            </button>
          )}
          {Object.keys(selected.attributes ?? {}).length > 0 && (
            <details>
              <summary>Recorded attributes</summary>
              <dl>
                {Object.entries(selected.attributes ?? {}).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{String(value ?? "Unknown")}</dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
          <details>
            <summary>Feature history</summary>
            {history.isError && <p role="alert">Feature history could not be loaded.</p>}
            {history.data?.map((revision) => (
              <p key={revision.revision}>
                Revision {revision.revision} · {revision.note} ·{" "}
                {new Date(revision.createdAt).toLocaleDateString()}
              </p>
            ))}
          </details>
          <h4>Inspection record</h4>
          {inspections.isError && (
            <p role="alert">
              Inspections could not be loaded.{" "}
              <button type="button" onClick={() => void inspections.refetch()}>
                Retry inspections
              </button>
            </p>
          )}
          {inspections.data?.map((inspection) => (
            <article key={inspection.id}>
              <strong>
                {inspection.condition} · {new Date(inspection.observedAt).toLocaleString()}
              </strong>
              <p>{inspection.notes}</p>
              {Object.entries(inspection.measurements ?? {}).length > 0 && (
                <dl>
                  {Object.entries(inspection.measurements ?? {}).map(([name, value]) => (
                    <div key={name}>
                      <dt>{name}</dt>
                      <dd>
                        {value} {inspection.measurementUnits?.[name]}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
              <small>Observed feature revision {inspection.featureRevision}</small>
            </article>
          ))}
          {canEdit && (
            <LandInspectionForm
              key={selected.id}
              landId={land.id}
              featureId={selected.id}
              onSaved={() => inspections.refetch()}
            />
          )}
        </article>
      )}
    </section>
  );
}
