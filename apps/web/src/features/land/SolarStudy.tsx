import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useUi } from "@/state/ui";
import { describeError } from "@/lib/log";
import { beginInvestigation, cancelResearch, useInvestigation } from "./researchApi";
import { importBoundary } from "./geometry";
import { parseSurveyCorners } from "./surveyDraft";
import { SolarAssessmentView } from "./SolarAssessmentView";
import {
  parseSolarDraft,
  solarRequest,
  type SolarAssessment,
  type SolarRequest,
} from "./solarStudy";
import "./solar.css";
const fields: {
  key: keyof SolarRequest;
  label: string;
  min: number;
  max: number;
  percent?: boolean;
}[] = [
  {
    key: "year",
    label: "Historical weather year",
    min: 2001,
    max: new Date().getUTCFullYear() - 1,
  },
  { key: "moduleAreaM2", label: "Total module face area (m²)", min: 0.01, max: 1e7 },
  { key: "moduleEfficiency", label: "Module efficiency (%)", min: 0.01, max: 50, percent: true },
  { key: "tiltDegrees", label: "Tilt above horizontal (degrees)", min: 0, max: 85 },
  { key: "azimuthDegrees", label: "Facing clockwise from north (degrees)", min: 0, max: 359.999 },
  { key: "dcAcRatio", label: "DC capacity / AC inverter ratio", min: 0.5, max: 2 },
  {
    key: "inverterEfficiency",
    label: "Nominal inverter efficiency (%)",
    min: 80,
    max: 100,
    percent: true,
  },
  {
    key: "temperatureCoefficient",
    label: "Power temperature coefficient (% / °C)",
    min: -1,
    max: 0,
    percent: true,
  },
  { key: "systemLoss", label: "Other DC system losses (%)", min: 0, max: 100, percent: true },
  {
    key: "additionalShadeLoss",
    label: "Additional uniform shading (%)",
    min: 0,
    max: 100,
    percent: true,
  },
  { key: "albedo", label: "Ground reflectance (%)", min: 0, max: 100, percent: true },
  { key: "iamB", label: "ASHRAE incidence coefficient", min: 0, max: 0.2 },
];
export function SolarStudy({
  land,
  onUse,
}: {
  land: LandArea;
  onUse: (a: SolarAssessment) => void;
}) {
  const scope = useLandScope();
  return <Study key={`${scope}:${land.id}`} land={land} scope={scope} onUse={onUse} />;
}
function Study({
  land,
  scope,
  onUse,
}: {
  land: LandArea;
  scope: string;
  onUse: (a: SolarAssessment) => void;
}) {
  const ready = useLandAccessReady(),
    canEdit = useLandCanEdit(),
    scene = useScene(),
    cache = useQueryClient();
  const storage = `living-world-land-draft:${encodeURIComponent(scope)}:solar:${land.id}`;
  const [draft, setDraft] = useState<SolarRequest | null>(null),
    [points, setPoints] = useState<number[][]>([]);
  const [recovery, setRecovery] = useState<string | null>(() => {
    try {
      return localStorage.getItem(storage);
    } catch {
      return null;
    }
  });
  const [error, setError] = useState<string | null>(null),
    [busy, setBusy] = useState(false);
  const [review, setReview] = useState<{
    signature: string;
    value: components["schemas"]["SolarPreview"];
  } | null>(null);
  const [offset, setOffset] = useState(0);
  const [investigationId, setInvestigationId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(storage + ":run");
    } catch {
      return null;
    }
  });
  const selectedId = useLandContext((s) => s.selectedSolarId);
  const picker = useLandContext((s) => s.pointPicker),
    section = useLandContext((s) => s.section),
    panel = useUi((s) => s.activePanel);
  const visible = ready && section === "scenarios" && panel === "land";
  const owner = `solar-draft:${land.id}`,
    ticket = useRef(0),
    pickTicket = useRef(0);
  const operation = useRef<{ signature: string; id: string; key: string } | null>(null);
  const signature = JSON.stringify({ draft, revision: land.revision });
  const preview = review?.signature === signature ? review.value : null;
  const investigation = useInvestigation(investigationId);
  const run = investigation.data?.runs[0];
  const latestOutput = investigation.data?.artifacts.find((a) => a.output.kind === "solar")?.output;
  const assessmentId =
    selectedId ?? (latestOutput?.kind === "solar" ? latestOutput.assessmentId : null);
  const running = run?.status === "queued" || run?.status === "running";
  const catalog = useQuery({
    queryKey: ["land-solar", scope, land.id, land.revision, offset, run?.status],
    enabled: ready,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/solar-assessments", {
          params: { path: { land_id: land.id }, query: { limit: 10, offset } },
        }),
      ),
  });
  const features = useQuery({
    queryKey: ["land-solar-features", scope, land.id],
    enabled: ready && !!draft,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features", {
          params: { path: { land_id: land.id }, query: { limit: 100 } },
        }),
      ),
  });
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );
  useEffect(() => {
    if (!visible) return;
    return () => {
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === owner) {
        scene?.areas.cancelPick();
        useLandContext.getState().setPointPicker(null);
      }
    };
  }, [visible, owner, scene]);
  useEffect(() => {
    if (!visible || !draft) return;
    useLandContext.getState().setLayer({
      id: owner,
      title: "Proposed solar array zone",
      features: [
        { id: "zone", label: "Proposed array zone", geometry: draft.arrayZone },
        ...points.map((coordinates, i) => ({
          id: `corner-${i}`,
          label: `Corner ${i + 1}`,
          geometry: { type: "Point" as const, coordinates },
        })),
      ],
    });
    return () => useLandContext.getState().removeLayer(owner);
  }, [visible, draft, points, owner]);
  function update(next: SolarRequest | null, corners = points) {
    ticket.current++;
    pickTicket.current += 1;
    if (useLandContext.getState().pointPicker === owner) {
      scene?.areas.cancelPick();
      useLandContext.getState().setPointPicker(null);
    }
    setBusy(false);
    setReview(null);
    setDraft(next);
    setPoints(corners);
    setError(null);
    try {
      if (next)
        localStorage.setItem(
          storage,
          JSON.stringify({ revision: land.revision, request: next, points: corners }),
        );
      else localStorage.removeItem(storage);
    } catch {
      setError("This browser could not save the draft for recovery. Keep this page open.");
    }
  }
  async function pick() {
    if (!scene || !draft) return;
    const current = ++pickTicket.current;
    useLandContext.getState().setPointPicker(owner);
    const point = await scene.areas.pickGround();
    if (current !== pickTicket.current) return;
    useLandContext.getState().setPointPicker(null);
    if (point) update(draft, [...points, [point.longitude, point.latitude]]);
  }
  async function calculate(start: boolean) {
    if (!draft || !canEdit) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      if (!start) {
        const value = await unwrap(
          api.POST("/api/v1/land/{land_id}/solar-assessments/preview", {
            params: { path: { land_id: land.id } },
            body: draft,
          }),
        );
        if (current === ticket.current) setReview({ signature, value });
      } else {
        let pending = operation.current;
        if (!pending) {
          try {
            const saved: unknown = JSON.parse(
              localStorage.getItem(storage + ":operation") ?? "null",
            );
            if (
              saved &&
              typeof saved === "object" &&
              "signature" in saved &&
              "id" in saved &&
              "key" in saved &&
              typeof saved.signature === "string" &&
              typeof saved.id === "string" &&
              typeof saved.key === "string"
            )
              pending = { signature: saved.signature, id: saved.id, key: saved.key };
          } catch {
            /* Malformed browser state cannot authorize a different server request. */
          }
        }
        if (pending?.signature !== signature) {
          const inv = await beginInvestigation(land, `Hourly solar assessment · ${draft.year}`);
          pending = { signature, id: inv.id, key: crypto.randomUUID() };
          operation.current = pending;
        }
        // Remember the investigation before queueing: accepted work remains discoverable after a lost response.
        try {
          localStorage.setItem(storage + ":run", pending.id);
          localStorage.setItem(storage + ":operation", JSON.stringify(pending));
        } catch {
          /* The queued investigation is still available in Discover. */
        }
        useLandContext.getState().selectSolar(null);
        setInvestigationId(pending.id);
        await unwrap(
          api.POST("/api/v1/research/investigations/{investigation_id}/runs", {
            params: { path: { investigation_id: pending.id } },
            body: {
              requestKey: pending.key,
              kind: "solar",
              question: `Calculate hourly generation for the mapped array in ${draft.year}.`,
              analysis: draft,
            },
          }),
        );
        if (current === ticket.current) {
          update(null, []);
          setRecovery(null);
        }
        await cache.invalidateQueries({ queryKey: ["land-research", scope] });
      }
    } catch (cause) {
      if (current === ticket.current) setError(describeError(cause));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  }
  return (
    <details className="land-solar-study" open>
      <summary>Model solar generation from hourly weather</summary>
      <p>
        Map an array zone, set its orientation and equipment, then compare monthly generation and
        economics. NASA provides regional weather; your roof and shading assumptions remain
        explicit.
      </p>
      {!draft && !recovery && canEdit && (
        <button
          type="button"
          disabled={busy}
          onClick={() => {
            operation.current = null;
            try {
              localStorage.removeItem(storage + ":operation");
            } catch {
              /* Recovery is optional. */
            }
            update(solarRequest(land.boundary), []);
          }}
        >
          New hourly solar assessment
        </button>
      )}
      {!draft && recovery && canEdit && (
        <div className="land-actions">
          <button
            type="button"
            onClick={() => {
              try {
                const parsed = JSON.parse(recovery) as {
                  revision: number;
                  request: unknown;
                  points: unknown;
                };
                const value = parseSolarDraft(JSON.stringify(parsed.request));
                update(value, parseSurveyCorners(JSON.stringify(parsed.points)));
                if (parsed.revision !== land.revision)
                  setError(
                    "The boundary changed. Review the recovered array against the current land.",
                  );
                setRecovery(null);
              } catch (cause) {
                setError(describeError(cause));
              }
            }}
          >
            Recover solar draft
          </button>
          <button
            type="button"
            onClick={() => {
              update(null, []);
              setRecovery(null);
            }}
          >
            Discard solar draft
          </button>
        </div>
      )}
      {draft && (
        <form
          className="land-scenario-form"
          onSubmit={(e) => {
            e.preventDefault();
            void calculate(false);
          }}
        >
          <fieldset disabled={busy || !canEdit}>
            <legend>1 · Locate the array</legend>
            <p className="land-footnote">
              Starts with the land boundary. Draw a smaller zone or use a saved polygon for a roof.
              The outline is a plan area; tilt and module area do not create a fitted panel layout.
            </p>
            <div className="land-actions">
              <button
                type="button"
                disabled={!scene || !!picker || points.length >= 1000}
                onClick={() => void pick()}
              >
                Pick next array corner
              </button>
              <button
                type="button"
                disabled={!points.length}
                onClick={() => update(draft, points.slice(0, -1))}
              >
                Undo corner
              </button>
              <button
                type="button"
                disabled={points.length < 3}
                onClick={() => {
                  try {
                    update(
                      {
                        ...draft,
                        arrayZone: importBoundary(
                          JSON.stringify({
                            type: "Polygon",
                            coordinates: [[...points, points[0]]],
                          }),
                        ),
                        zoneBasis: "Manually mapped array zone; verify on site.",
                      },
                      [],
                    );
                  } catch (cause) {
                    setError(describeError(cause));
                  }
                }}
              >
                Finish array ({points.length} corners)
              </button>
              <button
                type="button"
                onClick={() =>
                  update(
                    {
                      ...draft,
                      arrayZone: land.boundary,
                      zoneBasis: "Selected land boundary used as planning array zone.",
                    },
                    [],
                  )
                }
              >
                Use land boundary
              </button>
            </div>
            {picker === owner && (
              <p role="status">Choose a corner on the map. Press Escape to cancel.</p>
            )}
            <label className="land-name">
              Use a saved mapped polygon
              <select
                aria-label="Use a saved mapped polygon"
                value=""
                onChange={(e) => {
                  const f = features.data?.find((v) => v.id === e.target.value);
                  if (f && (f.geometry.type === "Polygon" || f.geometry.type === "MultiPolygon"))
                    update(
                      {
                        ...draft,
                        arrayZone: f.geometry,
                        zoneBasis: `Mapped feature: ${f.name}. Verify footprint and roof geometry.`,
                      },
                      [],
                    );
                }}
              >
                <option value="">Choose a feature…</option>
                {features.data
                  ?.filter(
                    (f) => f.geometry.type === "Polygon" || f.geometry.type === "MultiPolygon",
                  )
                  .map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.name}
                    </option>
                  ))}
              </select>
            </label>
            <label className="land-name">
              Import array outline (GeoJSON)
              <input
                type="file"
                accept=".json,.geojson,application/json"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  e.target.value = "";
                  if (!file) return;
                  const current = ticket.current;
                  void (async () => {
                    try {
                      if (file.size > 5_000_000) throw new Error("Outlines are limited to 5 MB.");
                      const boundary = importBoundary(await file.text());
                      if (current !== ticket.current) return;
                      update(
                        {
                          ...draft,
                          arrayZone: boundary,
                          zoneBasis: `Imported ${file.name}; verify its coordinate system and accuracy.`,
                        },
                        [],
                      );
                    } catch (cause) {
                      if (current === ticket.current) setError(describeError(cause));
                    }
                  })();
                }}
              />
            </label>
            <label className="land-name">
              Outline source and accuracy
              <textarea
                aria-label="Outline source and accuracy"
                required
                maxLength={2000}
                value={draft.zoneBasis}
                onChange={(e) => update({ ...draft, zoneBasis: e.target.value })}
              />
            </label>
          </fieldset>
          <fieldset disabled={busy || !canEdit}>
            <legend>2 · Equipment and orientation</legend>
            <p className="land-notice">
              Starting values are planning assumptions. North is 0°, east 90°, south 180°, west
              270°.
            </p>
            <div className="land-scenario-grid">
              {fields.map((f) => (
                <label className="land-name" key={f.key}>
                  {f.label}
                  <input
                    type="number"
                    step={f.key === "year" ? 1 : "any"}
                    required
                    min={f.min}
                    max={f.max}
                    value={
                      Number.isFinite(draft[f.key])
                        ? Number((Number(draft[f.key]) * (f.percent ? 100 : 1)).toPrecision(10))
                        : ""
                    }
                    onChange={(e) =>
                      update({
                        ...draft,
                        [f.key]:
                          e.target.value === ""
                            ? Number.NaN
                            : Number(e.target.value) / (f.percent ? 100 : 1),
                      })
                    }
                  />
                </label>
              ))}
            </div>
            <label className="land-name">
              Mounting thermal model
              <select
                aria-label="Mounting thermal model"
                value={draft.mounting}
                onChange={(e) =>
                  update({ ...draft, mounting: e.target.value as SolarRequest["mounting"] })
                }
              >
                <option value="open_rack_glass_glass">Open rack · glass/glass</option>
                <option value="close_mount_glass_glass">Close mount · glass/glass</option>
                <option value="insulated_back_glass_polymer">Insulated back · glass/polymer</option>
              </select>
            </label>
          </fieldset>
          <fieldset disabled={busy || !canEdit}>
            <legend>3 · Horizon and assumptions</legend>
            <p className="land-footnote">
              An empty profile assumes an open horizon. Enter at least four compass points with gaps
              ≤90°. Elevation is the obstruction angle above horizontal, not its height in meters.
            </p>
            {(draft.horizon ?? []).map((point, i) => (
              <div className="land-scenario-grid" key={i}>
                <label className="land-name">
                  Horizon azimuth {i + 1}
                  <input
                    type="number"
                    required
                    min={0}
                    max={359.999}
                    step="any"
                    value={point.azimuthDegrees}
                    onChange={(e) =>
                      update({
                        ...draft,
                        horizon: draft.horizon?.map((p, j) =>
                          j === i ? { ...p, azimuthDegrees: Number(e.target.value) } : p,
                        ),
                      })
                    }
                  />
                </label>
                <label className="land-name">
                  Horizon elevation {i + 1}
                  <input
                    type="number"
                    required
                    min={0}
                    max={90}
                    step="any"
                    value={point.elevationDegrees}
                    onChange={(e) =>
                      update({
                        ...draft,
                        horizon: draft.horizon?.map((p, j) =>
                          j === i ? { ...p, elevationDegrees: Number(e.target.value) } : p,
                        ),
                      })
                    }
                  />
                </label>
              </div>
            ))}
            <div className="land-actions">
              <button
                type="button"
                onClick={() =>
                  update({
                    ...draft,
                    horizon: [0, 90, 180, 270].map((azimuthDegrees) => ({
                      azimuthDegrees,
                      elevationDegrees: 0,
                    })),
                  })
                }
              >
                Enter four-point horizon
              </button>
              <button type="button" onClick={() => update({ ...draft, horizon: [] })}>
                Assume open horizon
              </button>
            </div>
            <label className="land-name">
              Horizon source and uncertainty
              <textarea
                aria-label="Horizon source and uncertainty"
                required
                maxLength={2000}
                value={draft.horizonBasis}
                onChange={(e) => update({ ...draft, horizonBasis: e.target.value })}
              />
            </label>
            <label className="land-name">
              Equipment sources and other assumptions
              <textarea
                aria-label="Equipment sources and other assumptions"
                required
                maxLength={5000}
                value={draft.assumptions}
                onChange={(e) => update({ ...draft, assumptions: e.target.value })}
              />
            </label>
          </fieldset>
          <div className="land-actions">
            <button type="submit" disabled={busy || !canEdit || points.length > 0}>
              {busy ? "Checking…" : "Review array"}
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                update(null, []);
                setRecovery(null);
              }}
            >
              Discard draft
            </button>
          </div>
          {preview && (
            <div className="land-notice">
              <p>
                {preview.mappedZoneAreaM2.toLocaleString(undefined, { maximumFractionDigits: 1 })}{" "}
                m² mapped zone · {preview.capacityKwDc.toLocaleString()} kW DC ·{" "}
                {preview.inverterKwAc.toLocaleString(undefined, { maximumFractionDigits: 1 })} kW
                AC. Uses boundary revision {land.revision}.
              </p>
              <p>
                Retrieve {draft.year} hourly regional weather and preserve the assumptions with the
                calculation.
              </p>
              <button
                type="button"
                disabled={busy || !canEdit}
                onClick={() => void calculate(true)}
              >
                Run hourly analysis
              </button>
            </div>
          )}
        </form>
      )}
      {run && (
        <div className="land-notice" role="status">
          Hourly analysis: {run.status}. {run.error}
          {running && canEdit && (
            <button
              type="button"
              onClick={() =>
                void cancelResearch(run.id)
                  .then(() => investigation.refetch())
                  .catch((cause: unknown) => setError(describeError(cause)))
              }
            >
              Cancel analysis
            </button>
          )}
        </div>
      )}
      {investigation.isError && (
        <p role="alert">
          Job status could not be loaded.{" "}
          <button type="button" onClick={() => void investigation.refetch()}>
            Retry status
          </button>
        </p>
      )}
      {catalog.isError && (
        <p role="alert">
          Saved assessments could not be loaded.{" "}
          <button type="button" onClick={() => void catalog.refetch()}>
            Retry assessments
          </button>
        </p>
      )}
      {!!catalog.data?.length && (
        <label className="land-name">
          Saved solar assessments
          <select
            aria-label="Saved solar assessments"
            value={selectedId ?? ""}
            onChange={(e) => useLandContext.getState().selectSolar(e.target.value || null)}
          >
            <option value="">Choose an assessment…</option>
            {catalog.data.map((a) => (
              <option key={a.id} value={a.id}>
                {a.request.year} · {a.metadata.capacityKwDc.toLocaleString()} kW ·{" "}
                {new Date(a.createdAt).toLocaleString()}
                {a.stale ? " · earlier boundary" : ""}
              </option>
            ))}
          </select>
        </label>
      )}
      {(offset > 0 || catalog.data?.length === 10) && (
        <div className="land-actions">
          <button
            type="button"
            disabled={!offset}
            onClick={() => setOffset(Math.max(0, offset - 10))}
          >
            Newer assessments
          </button>
          <button
            type="button"
            disabled={catalog.data?.length !== 10}
            onClick={() => setOffset(offset + 10)}
          >
            Older assessments
          </button>
        </div>
      )}
      {assessmentId && (
        <SolarAssessmentView id={assessmentId} onUse={canEdit ? onUse : undefined} />
      )}
      {error && (
        <p role="alert" className="land-error">
          {error}
        </p>
      )}
    </details>
  );
}
