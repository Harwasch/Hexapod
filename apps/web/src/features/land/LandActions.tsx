import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea, LandEvidence } from "@twin/contracts";
import { boundsOf } from "@twin/geo";
import { api, unwrap, ApiError } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useMission } from "@/state/mission";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useInvestigation, useInvestigations } from "./researchApi";
import { ActionEvidence } from "./ActionEvidence";
import { EvidenceView } from "./LandResearch";
import { ActionPlanSummary } from "./ActionPlanSummary";
import {
  actionDraftKey,
  actionEdit,
  actionPayload,
  downloadActionDraft,
  parseActionDraft,
  sameAction,
  serializeActionDraft,
  type ActionDraftSnapshot,
  type ActionEdit,
} from "./actionDraft";

type Action = components["schemas"]["LandActionRead"];
type Draft = components["schemas"]["LandActionCreate"];
type Step = components["schemas"]["LandActionStep"];
type Scenario = components["schemas"]["ScenarioRead"];
const newStep = (index: number): Step => ({
  id: crypto.randomUUID().slice(0, 8),
  title: "",
  detail: "",
  startDay: index,
  days: 1,
  successMeasure: "",
});
function editable(action: Action): Draft {
  return {
    requestKey: crypto.randomUUID(),
    title: action.title,
    objective: action.objective,
    boundaryRevision: action.boundaryRevision,
    scenario: action.scenario,
    features: action.features,
    evidenceIds: action.evidenceIds,
    startDate: action.startDate,
    currency: action.currency,
    steps: action.steps,
    constraints: action.constraints,
    exclusions: action.exclusions,
    assumptions: action.assumptions,
  };
}
function fresh(land: LandArea): Draft {
  return {
    requestKey: crypto.randomUUID(),
    title: "",
    objective: "",
    boundaryRevision: land.revision,
    startDate: new Date().toISOString().slice(0, 10),
    currency: "USD",
    steps: [newStep(0)],
    constraints: [],
    evidenceIds: [],
    features: [],
    exclusions: [],
    assumptions: [],
  };
}
function fromScenario(land: LandArea, scenario: Scenario): Draft {
  const draft = fresh(land);
  draft.title = scenario.name;
  draft.scenario = { id: scenario.id, revision: scenario.revision };
  draft.boundaryRevision = scenario.boundaryRevision;
  draft.evidenceIds = scenario.evidenceIds;
  draft.currency = scenario.inputs.currency;
  draft.assumptions = [scenario.inputs.assumptions];
  if (scenario.inputs.kind === "restoration") {
    draft.objective = `Restore toward ${scenario.inputs.referenceEcosystem}`;
    draft.steps = scenario.inputs.treatments.map((treatment, index) => ({
      ...newStep(index),
      title: treatment.name,
      detail: treatment.objective,
      startDay: treatment.year * 365,
      days: 1,
      estimatedCost: treatment.areaHa * treatment.costPerHa,
      costBasis:
        "Area × cost per hectare from the selected scenario; review duration and scheduling.",
      successMeasure: treatment.objective,
    }));
    draft.steps.push(
      ...scenario.inputs.monitoringYears.map((year, index) => ({
        ...newStep(index),
        title: `Monitor restoration in year ${year}`,
        startDay: year * 365,
        days: 1,
        estimatedCost:
          scenario.inputs.kind === "restoration" ? scenario.inputs.monitoringCostPerVisit : 0,
        costBasis: "Monitoring cost per visit from the selected restoration scenario.",
        successMeasure:
          "Repeat the recorded survey method and compare measured cover with scenario targets.",
      })),
    );
    draft.assumptions?.push(
      "Scenario years are converted to 365-day intervals. One-day step durations are placeholders for review before approval.",
    );
    if (!draft.steps.length) draft.steps = [newStep(0)];
    draft.constraints = [
      {
        text: "Confirm treatment footprints, timing, permissions and monitoring with site evidence.",
        resolved: false,
      },
    ];
  } else {
    draft.objective = "Evaluate and deliver the selected solar option";
    draft.steps = [
      {
        ...newStep(0),
        title: "Site and roof assessment",
        successMeasure: "Verify roof capacity, usable area, shading and electrical connection.",
      },
      {
        ...newStep(1),
        title: "Installation proposal",
        startDay: 1,
        successMeasure: "Review a site-specific design and contractor quote.",
      },
    ];
    draft.constraints = [
      {
        text: "Verify roof structure, design, permits and connection requirements before installation.",
        resolved: false,
      },
    ];
  }
  return draft;
}

export function LandActions({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <Actions key={`${scope}:${land.id}`} land={land} scope={scope} />;
}
function Actions({ land, scope }: { land: LandArea; scope: string }) {
  const ready = useLandAccessReady(),
    canEdit = useLandCanEdit();
  const cache = useQueryClient(),
    scene = useScene();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [editing, setEditing] = useState<ActionEdit | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [reviewNote, setReviewNote] = useState("");
  const project = useMission((state) => state.project);
  const [projectId, setProjectId] = useState(`land:${land.id}`);
  const [offset, setOffset] = useState(0);
  const [investigationId, setInvestigationId] = useState<string | null>(null);
  const [evidenceOffset, setEvidenceOffset] = useState(0);
  const [sourceEvidence, setSourceEvidence] = useState<LandEvidence | null>(null);
  const ticket = useRef(0);
  const review = useRef<HTMLElement>(null);
  const draftKey = actionDraftKey(scope, land.id);
  const [recovery, setRecovery] = useState<string | null>(() => {
    try {
      return localStorage.getItem(draftKey);
    } catch {
      return null;
    }
  });
  const [storageError, setStorageError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [conflict, setConflict] = useState<Action | null>(null);
  const snapshot: ActionDraftSnapshot | null = draft
    ? { version: 1, landId: land.id, draft, editing }
    : null;
  const serialized = snapshot ? serializeActionDraft(snapshot) : null;
  useEffect(() => {
    if (!serialized || recovery || !canEdit) return;
    let message: string | null = null;
    try {
      if (serialized.length > 4_000_000) throw new Error("Draft exceeds the recovery limit");
      localStorage.setItem(draftKey, serialized);
    } catch {
      message = "This browser could not preserve the action draft. Download a copy before leaving.";
    }
    let active = true;
    queueMicrotask(() => {
      if (active) setStorageError(message);
    });
    return () => {
      active = false;
    };
  }, [serialized, recovery, canEdit, draftKey]);
  const clearStored = () => {
    try {
      localStorage.removeItem(draftKey);
      setStorageError(null);
    } catch {
      setStorageError(
        "The browser could not remove the saved draft. It may reappear after reload.",
      );
    }
    setRecovery(null);
  };

  useEffect(
    () => () => {
      ticket.current++;
      useLandContext.getState().removeLayer("action-preview");
    },
    [],
  );
  const catalog = useQuery({
    queryKey: ["land-actions", scope, land.id, land.revision, offset],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions", {
          params: { path: { land_id: land.id }, query: { limit: 50, offset } },
        }),
      ),
    retry: false,
  });
  const scenarios = useQuery({
    queryKey: ["land-scenarios", scope, land.id],
    enabled: ready,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/scenarios", { params: { path: { land_id: land.id } } }),
      ),
    retry: false,
  });
  const features = useQuery({
    queryKey: ["land-action-features", scope, land.id],
    enabled: ready && Boolean(draft),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features", {
          params: { path: { land_id: land.id }, query: { limit: 500 } },
        }),
      ),
    retry: false,
  });
  const investigations = useInvestigations(land.id);
  const evidence = useInvestigation(investigationId, evidenceOffset);
  const selectedRecord = useQuery({
    queryKey: ["land-action-record", scope, land.id, selectedId],
    enabled: ready && Boolean(selectedId),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions/{action_id}", {
          params: { path: { land_id: land.id, action_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  const selected =
    selectedRecord.data ?? catalog.data?.find((item) => item.id === selectedId) ?? null;
  useEffect(() => {
    if (!draft && selected?.id) review.current?.scrollIntoView({ block: "nearest" });
  }, [draft, selected?.id]);
  const history = useQuery({
    queryKey: ["land-action-history", scope, land.id, selectedId, selected?.revision],
    enabled: ready && Boolean(selectedId),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions/{action_id}/revisions", {
          params: { path: { land_id: land.id, action_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  const mission = useQuery({
    queryKey: ["land-action-mission", scope, selectedId, selected?.missionId],
    enabled: ready && Boolean(selected?.missionId),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/actions/{action_id}/mission", {
          params: { path: { land_id: land.id, action_id: selectedId ?? "" } },
        }),
      ),
    retry: false,
  });
  const refresh = () =>
    Promise.all([
      cache.invalidateQueries({ queryKey: ["land-actions", scope, land.id] }),
      cache.invalidateQueries({ queryKey: ["land-action-record", scope, land.id] }),
    ]);
  const perform = async (operation: () => Promise<unknown>, after?: () => void) => {
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      await operation();
      await refresh();
      if (ticket.current === current) after?.();
    } catch (cause) {
      if (ticket.current === current) {
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
        if (draft && editing && cause instanceof ApiError && cause.status === 409) {
          try {
            const latest = await unwrap(
              api.GET("/api/v1/land/{land_id}/actions/{action_id}", {
                params: { path: { land_id: land.id, action_id: editing.id } },
              }),
            );
            if (ticket.current === current) {
              setConflict(latest);
              setError(null);
            }
          } catch {
            /* Keep the original error and local draft available. */
          }
        }
      }
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };
  const save = () => {
    if (!draft || conflict) return;
    return perform(
      async () => {
        const current = ticket.current;
        const payload = actionPayload(draft);
        const result = editing
          ? await unwrap(
              api.PUT("/api/v1/land/{land_id}/actions/{action_id}", {
                params: { path: { land_id: land.id, action_id: editing.id } },
                body: {
                  ...payload,
                  expectedRevision: editing.revision,
                  note: "Revised in the land action workspace",
                },
              }),
            )
          : await unwrap(
              api.POST("/api/v1/land/{land_id}/actions", {
                params: { path: { land_id: land.id } },
                body: payload,
              }),
            );
        if (ticket.current === current) setSelectedId(result.id);
      },
      () => {
        clearStored();
        setConflict(null);
        setDraft(null);
        setEditing(null);
        setReviewNote("");
      },
    );
  };
  const beginEdit = (action: Action) => {
    clearStored();
    setConflict(null);
    setNotice(null);
    setError(null);
    setReviewNote("");
    setEditing(actionEdit(action));
    setDraft(editable(action));
  };
  const recover = async (asNew = false) => {
    if (!recovery) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    setNotice(null);
    setReviewNote("");
    try {
      const saved = parseActionDraft(recovery, land.id);
      let baseline = asNew ? null : saved.editing;
      let latest: Action | null = null;
      let value = asNew ? { ...saved.draft, requestKey: crypto.randomUUID() } : saved.draft;
      if (!asNew) {
        const response = await api.GET("/api/v1/land/{land_id}/actions/requests/{request_key}", {
          params: { path: { land_id: land.id, request_key: saved.draft.requestKey ?? "" } },
        });
        if (response.response.status !== 404) {
          const found = await unwrap(Promise.resolve(response));
          if (ticket.current !== current) return;
          if (sameAction(saved.draft, found.saved)) {
            cache.setQueryData(
              ["land-action-record", scope, land.id, found.current.id],
              found.current,
            );
            clearStored();
            setSelectedId(found.current.id);
            setNotice(
              "This draft was already saved. Opened the current action without approving or scheduling it.",
            );
            await refresh();
            return;
          }
          baseline = actionEdit(found.saved);
          latest = found.current;
          value = { ...saved.draft, requestKey: crypto.randomUUID() };
        }
        if (baseline && !latest)
          latest = await unwrap(
            api.GET("/api/v1/land/{land_id}/actions/{action_id}", {
              params: { path: { land_id: land.id, action_id: baseline.id } },
            }),
          );
      }
      if (ticket.current !== current) return;
      setEditing(baseline);
      setDraft(value);
      setRecovery(null);
      setConflict(
        latest &&
          baseline &&
          (latest.revision !== baseline.revision ||
            latest.status !== baseline.status ||
            latest.missionId !== baseline.missionId)
          ? latest
          : null,
      );
      setNotice("Recovered your draft. Review the work sequence and references before saving.");
    } catch (cause) {
      if (ticket.current === current) setError(describeError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };
  const showMap = (action: Action) => {
    useLandContext.getState().setLayer({
      id: "action-preview",
      title: action.title,
      features: [
        {
          id: "work-area",
          label: "Work area after exclusions",
          geometry: action.effectiveBoundary,
        },
        ...action.steps.flatMap((step) =>
          step.footprint ? [{ id: step.id, label: step.title, geometry: step.footprint }] : [],
        ),
      ],
    });
    const b = boundsOf(action.effectiveBoundary);
    scene?.camera.flyToRectangle(b.west - 0.001, b.south - 0.001, b.east + 0.001, b.north + 0.001);
  };
  const updateStep = (index: number, change: Partial<Step>) => {
    if (draft)
      setDraft({
        ...draft,
        steps: draft.steps.map((step, i) => (i === index ? { ...step, ...change } : step)),
      });
  };
  const exportAction = (action: Action) => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(action, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "land-action.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <section className="land-action-workspace" aria-label="Land actions">
      <span className="land-eyebrow">From understanding to action</span>
      <h3>Plan what happens next.</h3>
      <p>
        Build a work sequence from this land, its evidence and a saved scenario. Review a specific
        version before scheduling.
      </p>
      {recovery && canEdit && (
        <div className="land-notice" role="region" aria-label="Recover action draft">
          <p>
            An unfinished action is saved in this browser for this land. Recovery does not approve
            or schedule work.
          </p>
          <div className="land-actions">
            <button type="button" disabled={busy} onClick={() => void recover()}>
              Recover action draft
            </button>
            <button type="button" disabled={busy} onClick={() => void recover(true)}>
              Recover as a new action
            </button>
            <button type="button" onClick={() => downloadActionDraft(recovery)}>
              Download action draft
            </button>
            <button type="button" disabled={busy} onClick={clearStored}>
              Discard action draft
            </button>
          </div>
        </div>
      )}
      {notice && <p role="status">{notice}</p>}
      {storageError && <p role="alert">{storageError}</p>}
      {draft && (
        <div className="land-actions">
          <button type="button" onClick={() => downloadActionDraft(snapshot)}>
            Download action draft
          </button>
        </div>
      )}
      {conflict && (
        <div className="land-notice" role="region" aria-label="Review changed action">
          <p>
            The saved action is now revision {conflict.revision}, {conflict.status}. Your draft
            started from revision {editing?.revision}, {editing?.status}.
          </p>
          <details>
            <summary>Current saved work sequence</summary>
            <ActionPlanSummary action={conflict} />
            <div className="land-actions">
              <button type="button" onClick={() => showMap(conflict)}>
                Show saved work areas on map
              </button>
            </div>
          </details>
          <div className="land-actions">
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                if (draft) setDraft({ ...draft, requestKey: crypto.randomUUID() });
                setEditing(null);
                setConflict(null);
                setError(null);
              }}
            >
              Keep my work as a new action
            </button>
            <button type="button" disabled={busy} onClick={() => beginEdit(conflict)}>
              Load latest plan into the draft
            </button>
          </div>
          <p>Approval and scheduling require a separate review of the saved revision.</p>
        </div>
      )}
      {selectedRecord.isError && (
        <p role="alert">
          The selected action could not be loaded.{" "}
          <button type="button" onClick={() => void selectedRecord.refetch()}>
            Retry selected action
          </button>
        </p>
      )}
      {error && (
        <p className="land-error" role="alert">
          {error}
        </p>
      )}
      {catalog.isError && (
        <p role="alert">
          Actions could not be loaded.{" "}
          <button type="button" onClick={() => void catalog.refetch()}>
            Retry
          </button>
        </p>
      )}
      {canEdit && !draft && !recovery && (
        <div className="land-actions">
          <button
            type="button"
            onClick={() => {
              setEditing(null);
              setNotice(null);
              setError(null);
              setDraft(fresh(land));
            }}
          >
            Plan an action
          </button>
        </div>
      )}
      {canEdit && !draft && !recovery && Boolean(scenarios.data?.length) && (
        <details>
          <summary>Start from a saved scenario</summary>
          <div className="land-candidates">
            {scenarios.data?.map((scenario) => (
              <button
                key={scenario.id}
                type="button"
                onClick={() => {
                  setEditing(null);
                  setNotice(null);
                  setError(null);
                  setDraft(fromScenario(land, scenario));
                }}
              >
                <strong>{scenario.name}</strong>
                <span>
                  Revision {scenario.revision} · {scenario.inputs.kind}
                  {scenario.stale ? " · earlier boundary" : ""}
                </span>
              </button>
            ))}
          </div>
        </details>
      )}
      {draft && (
        <form
          className="land-action-form"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <fieldset disabled={busy || !canEdit}>
            <legend>Objective and timing</legend>
            <label className="land-name">
              Action title
              <input
                required
                maxLength={200}
                value={draft.title}
                onChange={(event) => setDraft({ ...draft, title: event.target.value })}
              />
            </label>
            <label className="land-name">
              Outcome to achieve
              <textarea
                required
                maxLength={2000}
                value={draft.objective}
                onChange={(event) => setDraft({ ...draft, objective: event.target.value })}
              />
            </label>
            <div className="land-action-grid">
              <label className="land-name">
                Start date
                <input
                  required
                  type="date"
                  value={draft.startDate}
                  onChange={(event) => setDraft({ ...draft, startDate: event.target.value })}
                />
              </label>
              <label className="land-name">
                Cost currency
                <input
                  required
                  pattern="[A-Z]{3}"
                  maxLength={3}
                  value={draft.currency}
                  onChange={(event) =>
                    setDraft({ ...draft, currency: event.target.value.toUpperCase() })
                  }
                />
              </label>
            </div>
            <label className="land-name">
              Supporting scenario
              <select
                aria-label="Supporting scenario"
                value={draft.scenario ? `${draft.scenario.id}:${draft.scenario.revision}` : ""}
                onChange={(event) => {
                  const selectedScenario = scenarios.data?.find(
                    (item) => `${item.id}:${item.revision}` === event.target.value,
                  );
                  setDraft({
                    ...draft,
                    scenario: selectedScenario
                      ? { id: selectedScenario.id, revision: selectedScenario.revision }
                      : null,
                    boundaryRevision: selectedScenario?.boundaryRevision ?? draft.boundaryRevision,
                  });
                }}
              >
                <option value="">No scenario attached</option>
                {draft.scenario &&
                  !scenarios.data?.some(
                    (item) =>
                      item.id === draft.scenario?.id && item.revision === draft.scenario?.revision,
                  ) && (
                    <option value={`${draft.scenario.id}:${draft.scenario.revision}`}>
                      Preserved scenario revision {draft.scenario.revision}
                    </option>
                  )}
                {scenarios.data?.map((item) => (
                  <option key={item.id} value={`${item.id}:${item.revision}`}>
                    {item.name} · revision {item.revision} · boundary {item.boundaryRevision}
                  </option>
                ))}
              </select>
            </label>
            <p className="land-footnote">
              Selecting a scenario uses its boundary revision. Review the work areas and exclusions
              after changing it.
            </p>
            <p className="land-footnote">
              Based on boundary revision {draft.boundaryRevision}.{" "}
              {draft.scenario
                ? `Scenario revision ${draft.scenario.revision} is preserved with this draft.`
                : ""}
            </p>
            {draft.boundaryRevision !== land.revision && (
              <p className="land-notice">
                The land boundary changed.{" "}
                <button
                  type="button"
                  onClick={() =>
                    setDraft({ ...draft, boundaryRevision: land.revision, scenario: null })
                  }
                >
                  Use current boundary and detach earlier scenario
                </button>
              </p>
            )}
          </fieldset>
          <fieldset disabled={busy || !canEdit}>
            <legend>Work sequence</legend>
            {draft.steps.map((step, index) => (
              <article key={step.id} className="land-action-step">
                <span className="land-eyebrow">Step {index + 1}</span>
                <label className="land-name">
                  Step name
                  <input
                    aria-label={`Step ${index + 1} name`}
                    required
                    maxLength={200}
                    value={step.title}
                    onChange={(event) => updateStep(index, { title: event.target.value })}
                  />
                </label>
                <label className="land-name">
                  Work description
                  <textarea
                    maxLength={1000}
                    value={step.detail ?? ""}
                    onChange={(event) => updateStep(index, { detail: event.target.value })}
                  />
                </label>
                <div className="land-action-grid">
                  <label className="land-name">
                    Start day
                    <input
                      aria-label={`Step ${index + 1} start day`}
                      type="number"
                      min={1}
                      max={36501}
                      required
                      value={Number.isFinite(step.startDay) ? step.startDay + 1 : ""}
                      onChange={(event) =>
                        updateStep(index, { startDay: event.target.valueAsNumber - 1 })
                      }
                    />
                  </label>
                  <label className="land-name">
                    Duration (days)
                    <input
                      type="number"
                      min={1}
                      max={3650}
                      required
                      value={Number.isFinite(step.days) ? step.days : ""}
                      onChange={(event) => updateStep(index, { days: event.target.valueAsNumber })}
                    />
                  </label>
                </div>
                <label className="land-name">
                  Success measure
                  <textarea
                    aria-label={`Step ${index + 1} success measure`}
                    required
                    maxLength={2000}
                    value={step.successMeasure}
                    onChange={(event) => updateStep(index, { successMeasure: event.target.value })}
                  />
                </label>
                <label className="land-name">
                  Step work area
                  <select
                    aria-label={`Step ${index + 1} work area`}
                    value={step.footprint ? "custom" : "whole"}
                    onChange={(event) => {
                      const feature = features.data?.find((item) => item.id === event.target.value);
                      const geometry = feature?.geometry;
                      if (
                        feature &&
                        geometry &&
                        (geometry.type === "Polygon" || geometry.type === "MultiPolygon")
                      ) {
                        setDraft({
                          ...draft,
                          steps: draft.steps.map((item, i) =>
                            i === index ? { ...item, footprint: geometry } : item,
                          ),
                          features: [
                            ...(draft.features ?? []).filter((item) => item.id !== feature.id),
                            { id: feature.id, revision: feature.revision },
                          ],
                        });
                      } else updateStep(index, { footprint: null });
                    }}
                  >
                    <option value="whole">Whole work area, excluding protected areas</option>
                    {step.footprint && (
                      <option value="custom">Specific footprint · review on map</option>
                    )}
                    {features.data
                      ?.filter(
                        (item) =>
                          item.geometry.type === "Polygon" || item.geometry.type === "MultiPolygon",
                      )
                      .map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.name}
                        </option>
                      ))}
                  </select>
                </label>
                <details>
                  <summary>Resources, costs and prerequisites</summary>
                  <label className="land-name">
                    Resources (one per line)
                    <textarea
                      value={(step.resources ?? []).join("\n")}
                      onChange={(event) =>
                        updateStep(index, { resources: event.target.value.split("\n") })
                      }
                    />
                  </label>
                  <label className="land-name">
                    Estimated cost ({draft.currency})
                    <input
                      type="number"
                      min={0}
                      step="any"
                      value={step.estimatedCost ?? ""}
                      placeholder="Not estimated"
                      onChange={(event) =>
                        updateStep(index, {
                          estimatedCost:
                            event.target.value === "" ? null : Number(event.target.value),
                        })
                      }
                    />
                  </label>
                  {step.estimatedCost !== null && step.estimatedCost !== undefined && (
                    <label className="land-name">
                      Cost basis
                      <textarea
                        required
                        value={step.costBasis ?? ""}
                        onChange={(event) => updateStep(index, { costBasis: event.target.value })}
                      />
                    </label>
                  )}
                  {draft.steps
                    .filter((other) => other.id !== step.id)
                    .map((other) => (
                      <label className="land-dismissed" key={other.id}>
                        <input
                          type="checkbox"
                          checked={step.dependsOn?.includes(other.id) ?? false}
                          onChange={(event) =>
                            updateStep(index, {
                              dependsOn: event.target.checked
                                ? [...(step.dependsOn ?? []), other.id]
                                : step.dependsOn?.filter((id) => id !== other.id),
                            })
                          }
                        />{" "}
                        After {other.title || "Untitled step"}
                      </label>
                    ))}
                </details>
                <div className="land-actions">
                  <button
                    type="button"
                    disabled={draft.steps.length === 1}
                    onClick={() =>
                      setDraft({
                        ...draft,
                        steps: draft.steps
                          .filter((_, i) => i !== index)
                          .map((other) => ({
                            ...other,
                            dependsOn: other.dependsOn?.filter((id) => id !== step.id),
                          })),
                      })
                    }
                  >
                    Remove step {index + 1}
                  </button>
                </div>
              </article>
            ))}
            <div className="land-actions">
              <button
                type="button"
                disabled={draft.steps.length >= 250}
                onClick={() =>
                  setDraft({
                    ...draft,
                    steps: [
                      ...draft.steps,
                      newStep(Math.max(...draft.steps.map((step) => step.startDay + step.days))),
                    ],
                  })
                }
              >
                Add step
              </button>
            </div>
          </fieldset>
          <fieldset disabled={busy || !canEdit}>
            <legend>Evidence, features and exclusions</legend>
            <label className="land-name">
              Research investigation
              <select
                aria-label="Research investigation"
                value={investigationId ?? ""}
                onChange={(event) => {
                  setInvestigationId(event.target.value || null);
                  setEvidenceOffset(0);
                }}
              >
                <option value="">Choose evidence to support this action</option>
                {investigations.data?.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title}
                  </option>
                ))}
              </select>
            </label>
            {evidence.isError && <p role="alert">Evidence could not be loaded.</p>}
            {evidence.data?.evidence.map((item) => (
              <label className="land-dismissed" key={item.id}>
                <input
                  type="checkbox"
                  checked={draft.evidenceIds?.includes(item.id) ?? false}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      evidenceIds: event.target.checked
                        ? [...(draft.evidenceIds ?? []), item.id]
                        : draft.evidenceIds?.filter((id) => id !== item.id),
                    })
                  }
                />
                <span>
                  {item.title} · {item.spatialRelevance}{" "}
                  <button type="button" onClick={() => setSourceEvidence(item)}>
                    Inspect source{item.document ? ` · page ${item.document.page}` : ""}
                  </button>
                </span>
              </label>
            ))}
            {sourceEvidence && (
              <EvidenceView evidence={sourceEvidence} onClose={() => setSourceEvidence(null)} />
            )}
            {(evidenceOffset > 0 ||
              (evidence.data?.page.totals.evidence ?? 0) > evidenceOffset + 100) && (
              <div className="land-actions">
                <button
                  type="button"
                  disabled={!evidenceOffset}
                  onClick={() => setEvidenceOffset(Math.max(0, evidenceOffset - 100))}
                >
                  Previous evidence
                </button>
                <button
                  type="button"
                  disabled={(evidence.data?.page.totals.evidence ?? 0) <= evidenceOffset + 100}
                  onClick={() => setEvidenceOffset(evidenceOffset + 100)}
                >
                  More evidence
                </button>
              </div>
            )}
            <p className="land-footnote">
              {draft.evidenceIds?.length ?? 0} source records attached. Evidence remains traceable
              to its research record.
            </p>
            <details>
              <summary>Inventory features and areas to avoid</summary>
              {features.isError && <p role="alert">Inventory could not be loaded.</p>}
              {features.data?.map((feature) => (
                <div key={feature.id} className="land-action-feature">
                  <label className="land-dismissed">
                    <input
                      type="checkbox"
                      checked={draft.features?.some((item) => item.id === feature.id) ?? false}
                      onChange={(event) =>
                        setDraft({
                          ...draft,
                          features: event.target.checked
                            ? [
                                ...(draft.features ?? []).filter((item) => item.id !== feature.id),
                                { id: feature.id, revision: feature.revision },
                              ]
                            : draft.features?.filter((item) => item.id !== feature.id),
                        })
                      }
                    />
                    {feature.name} · revision {feature.revision}
                  </label>
                  {draft.features?.some(
                    (item) => item.id === feature.id && item.revision !== feature.revision,
                  ) && (
                    <p className="land-notice">
                      This action references an earlier feature version.{" "}
                      <button
                        type="button"
                        onClick={() =>
                          setDraft({
                            ...draft,
                            features: draft.features?.map((item) =>
                              item.id === feature.id
                                ? { id: feature.id, revision: feature.revision }
                                : item,
                            ),
                          })
                        }
                      >
                        Use current feature revision
                      </button>
                    </p>
                  )}
                  {(feature.geometry.type === "Polygon" ||
                    feature.geometry.type === "MultiPolygon") && (
                    <div className="land-actions">
                      <button
                        type="button"
                        onClick={() => {
                          const geometry = feature.geometry;
                          if (geometry.type !== "Polygon" && geometry.type !== "MultiPolygon")
                            return;
                          setDraft({
                            ...draft,
                            exclusions: [...(draft.exclusions ?? []), geometry],
                          });
                        }}
                      >
                        Exclude this footprint
                      </button>
                    </div>
                  )}
                </div>
              ))}
              {features.data?.length === 500 && (
                <p className="land-notice">Showing the first 500 inventory features.</p>
              )}
              <p>{draft.exclusions?.length ?? 0} excluded footprints</p>
              {Boolean(draft.exclusions?.length) && (
                <div className="land-actions">
                  <button type="button" onClick={() => setDraft({ ...draft, exclusions: [] })}>
                    Clear exclusions
                  </button>
                </div>
              )}
            </details>
          </fieldset>
          <fieldset disabled={busy || !canEdit}>
            <legend>Constraints and assumptions</legend>
            {(draft.constraints ?? []).map((constraint, index) => (
              <div className="land-action-step" key={index}>
                <label className="land-name">
                  Constraint
                  <textarea
                    required
                    value={constraint.text}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        constraints: draft.constraints?.map((item, i) =>
                          i === index ? { ...item, text: event.target.value } : item,
                        ),
                      })
                    }
                  />
                </label>
                <label className="land-dismissed">
                  <input
                    type="checkbox"
                    checked={constraint.resolved ?? false}
                    onChange={(event) =>
                      setDraft({
                        ...draft,
                        constraints: draft.constraints?.map((item, i) =>
                          i === index ? { ...item, resolved: event.target.checked } : item,
                        ),
                      })
                    }
                  />
                  Resolved
                </label>
                {constraint.resolved && (
                  <label className="land-name">
                    How was it resolved?
                    <textarea
                      required
                      value={constraint.resolution ?? ""}
                      onChange={(event) =>
                        setDraft({
                          ...draft,
                          constraints: draft.constraints?.map((item, i) =>
                            i === index ? { ...item, resolution: event.target.value } : item,
                          ),
                        })
                      }
                    />
                  </label>
                )}
                <div className="land-actions">
                  <button
                    type="button"
                    onClick={() =>
                      setDraft({
                        ...draft,
                        constraints: draft.constraints?.filter((_, i) => i !== index),
                      })
                    }
                  >
                    Remove constraint
                  </button>
                </div>
              </div>
            ))}
            <div className="land-actions">
              <button
                type="button"
                onClick={() =>
                  setDraft({
                    ...draft,
                    constraints: [...(draft.constraints ?? []), { text: "", resolved: false }],
                  })
                }
              >
                Add constraint
              </button>
            </div>
            <label className="land-name">
              Assumptions (one per line)
              <textarea
                value={(draft.assumptions ?? []).join("\n")}
                onChange={(event) =>
                  setDraft({ ...draft, assumptions: event.target.value.split("\n") })
                }
              />
            </label>
          </fieldset>
          <div className="land-actions">
            <button type="submit" disabled={busy || !canEdit || !!conflict}>
              {editing ? "Save action revision" : "Save action draft"}
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => {
                clearStored();
                setConflict(null);
                setNotice(null);
                setError(null);
                setDraft(null);
                setEditing(null);
              }}
            >
              Cancel edit
            </button>
          </div>
        </form>
      )}
      {!draft && (
        <div className="land-candidates">
          {catalog.data?.map((action) => (
            <button
              key={action.id}
              type="button"
              aria-pressed={selectedId === action.id}
              onClick={() => {
                setSelectedId(action.id);
                setReviewNote("");
              }}
            >
              <strong>{action.title}</strong>
              <span>
                {action.status} · revision {action.revision}
                {action.staleReasons.length ? " · supporting information changed" : ""}
              </span>
            </button>
          ))}
        </div>
      )}
      {!draft && (offset > 0 || catalog.data?.length === 50) && (
        <div className="land-actions">
          <button
            type="button"
            disabled={!offset}
            onClick={() => {
              setOffset(Math.max(0, offset - 50));
              setSelectedId(null);
            }}
          >
            Previous actions
          </button>
          <button
            type="button"
            disabled={catalog.data?.length !== 50}
            onClick={() => {
              setOffset(offset + 50);
              setSelectedId(null);
            }}
          >
            More actions
          </button>
        </div>
      )}
      {!draft && selected && (
        <article ref={review} className="land-action-review" aria-label="Action review">
          <h4>{selected.title}</h4>
          <p>{selected.objective}</p>
          {selected.staleReasons.map((reason) => (
            <p key={reason} className="land-notice">
              {reason}
            </p>
          ))}
          <p>
            {selected.startDate} · {selected.totalKnownCost.toLocaleString()} {selected.currency}{" "}
            known cost
            {selected.uncostedSteps > 0
              ? ` · ${selected.uncostedSteps} steps still need estimates`
              : ""}
          </p>
          <ol className="land-action-sequence">
            {selected.steps.map((step) => (
              <li key={step.id}>
                <strong>{step.title}</strong>
                <span>
                  Day {step.startDay + 1} · {step.days} days
                </span>
                <p>{step.detail}</p>
                <p>Success: {step.successMeasure}</p>
                {Boolean(step.resources?.length) && <p>Resources: {step.resources?.join(", ")}</p>}
                {step.estimatedCost !== null && step.estimatedCost !== undefined && (
                  <p>
                    {step.estimatedCost.toLocaleString()} {selected.currency} · {step.costBasis}
                  </p>
                )}
              </li>
            ))}
          </ol>
          {selected.constraints?.map((constraint, index) => (
            <p key={index} className={constraint.resolved ? "land-footnote" : "land-notice"}>
              {constraint.resolved ? "Resolved" : "Unresolved"}: {constraint.text}
              {constraint.resolution ? ` — ${constraint.resolution}` : ""}
            </p>
          ))}
          {selected.assumptions?.map((assumption, index) => (
            <p key={index} className="land-footnote">
              Assumption: {assumption}
            </p>
          ))}
          <p className="land-footnote">
            Boundary revision {selected.boundaryRevision} · {selected.evidenceIds?.length ?? 0}{" "}
            evidence records · {selected.features?.length ?? 0} inventory references
            {selected.scenario ? ` · scenario revision ${selected.scenario.revision}` : ""}
          </p>
          <ActionEvidence key={`${selected.id}:${selected.revision}`} action={selected} />
          <div className="land-actions">
            <button type="button" onClick={() => showMap(selected)}>
              Show work area on map
            </button>
            <button type="button" onClick={() => exportAction(selected)}>
              Export action
            </button>
            {canEdit && (
              <button
                type="button"
                disabled={busy || !!draft || !!recovery}
                onClick={() => beginEdit(selected)}
              >
                Revise action
              </button>
            )}
          </div>
          {selected.approvedAt && (
            <p>
              Approved revision {selected.revision} ·{" "}
              {new Date(selected.approvedAt).toLocaleString()}
              <br />
              {selected.approvalNote}
            </p>
          )}
          {canEdit && selected.status !== "scheduled" && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void perform(
                  () =>
                    selected.status === "draft"
                      ? unwrap(
                          api.POST("/api/v1/land/{land_id}/actions/{action_id}/approve", {
                            params: { path: { land_id: land.id, action_id: selected.id } },
                            body: { expectedRevision: selected.revision, note: reviewNote },
                          }),
                        )
                      : unwrap(
                          api.POST("/api/v1/land/{land_id}/actions/{action_id}/mission", {
                            params: { path: { land_id: land.id, action_id: selected.id } },
                            body: {
                              expectedRevision: selected.revision,
                              note: reviewNote,
                              projectId,
                            },
                          }),
                        ),
                  () => setReviewNote(""),
                );
              }}
            >
              <label className="land-name">
                {selected.status === "draft" ? "Review note" : "Scheduling note"}
                <textarea
                  required
                  maxLength={2000}
                  aria-label={selected.status === "draft" ? "Review note" : "Scheduling note"}
                  value={reviewNote}
                  onChange={(event) => setReviewNote(event.target.value)}
                />
              </label>
              {selected.status === "approved" && (
                <label className="land-name">
                  Schedule under
                  <select
                    aria-label="Schedule under"
                    value={projectId}
                    onChange={(event) => setProjectId(event.target.value)}
                  >
                    <option value={`land:${land.id}`}>{land.name}</option>
                    {project && <option value={project.id}>{project.name}</option>}
                  </select>
                </label>
              )}
              <div className="land-actions">
                <button
                  disabled={
                    busy ||
                    Boolean(selected.staleReasons.length) ||
                    Boolean(selected.constraints?.some((constraint) => !constraint.resolved)) ||
                    !reviewNote.trim()
                  }
                >
                  {selected.status === "draft"
                    ? `Approve revision ${selected.revision}`
                    : "Schedule approved action"}
                </button>
              </div>
              <p className="land-footnote">
                {selected.status === "draft"
                  ? "Approval records your review of this version. Scheduling is a separate step."
                  : "Creates a private scheduled mission. Equipment dispatch remains a separate operation."}
              </p>
            </form>
          )}
          {mission.data && (
            <p role="status">
              Mission scheduled for {mission.data.startDate} through {mission.data.endDate}. Status:{" "}
              {mission.data.status}.
            </p>
          )}
          {mission.isError && <p role="alert">The scheduled mission could not be loaded.</p>}
          <details>
            <summary>Action revision history</summary>
            {history.isError && <p role="alert">History could not be loaded.</p>}
            {history.data?.map((version) => (
              <div key={version.revision}>
                <p>
                  Revision {version.revision} · {version.status} · {version.title}
                </p>
                <div className="land-actions">
                  <button type="button" onClick={() => exportAction(version)}>
                    Export revision {version.revision}
                  </button>
                  <button type="button" onClick={() => showMap(version)}>
                    Map revision {version.revision}
                  </button>
                </div>
              </div>
            ))}
          </details>
        </article>
      )}
    </section>
  );
}
