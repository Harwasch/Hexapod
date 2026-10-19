import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { ArrowLeft, Download, PenLine, Plus, Trash2, X } from "lucide-react";

import type { LandArea, LandEvidence } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";
import { GlassButton, GlassInput, GlassSelect, GlassTextarea } from "@twin/ui";

import { ApiError, api, unwrap } from "@/api/client";

import {
  actionDraftKey,
  actionFieldId,
  actionPayload,
  downloadActionDraft,
  serializeActionDraft,
  type ActionDraftSnapshot,
} from "../../../actionDraft";
import { EvidenceView } from "../../../EvidenceView";
import { useInvestigation, useInvestigations } from "../../../researchApi";
import { LensCard, LensDetails } from "../../LensCard";
import { useLensLayer, useLensPins } from "../../useLensMap";
import { PlanField } from "./PlanField";
import { PlanLegend } from "./PlanLegend";
import { PlanSummary } from "./PlanDetails";
import { usePlanProblems, PROBLEMS_ID, type PlanProblems } from "./planProblems";
import { useDraftAutosave, type PlanSeed } from "./planRecovery";
import { FieldProblem, PlanProblemList } from "./PlanSaveProblems";
import { planError, useLandFeatures, useRefreshPlans, useScenarios } from "./plansApi";
import {
  avoidLayer,
  cornersInside,
  drawingLayer,
  stepPins,
  stepsLayer,
  workAreaLayer,
} from "./plansMap";
import {
  AVOID_COLOR,
  newStep,
  stepColor,
  editablePlan,
  type Action,
  type ActionDraft,
  type Area,
  type Step,
} from "./planModel";
import { shortArea } from "./plansFormat";
import { usePlanDrawing, type PlanDrawing } from "./usePlanDrawing";

type Feature = NonNullable<ReturnType<typeof useLandFeatures>["data"]>[number];

function isArea(geometry: Feature["geometry"]): geometry is Area {
  return geometry.type === "Polygon" || geometry.type === "MultiPolygon";
}

const identities = new WeakMap<object, number>();
let nextIdentity = 1;

/** A key that changes only when an object in the list is replaced, so layers redraw on change. */
function identityKey(items: (object | null | undefined)[]): string {
  return items
    .map((item) => {
      if (!item) return "0";
      let id = identities.get(item);
      if (!id) {
        id = nextIdentity++;
        identities.set(item, id);
      }
      return String(id);
    })
    .join(",");
}

/** The bar shown while an area is being drawn on the map. */
function DrawBar({
  drawing,
  land,
  onFinish,
}: {
  drawing: PlanDrawing;
  land: LandArea;
  onFinish: () => void;
}) {
  const count = drawing.points.length;
  return (
    <div className="plans-drawbar" role="group" aria-label="Drawing an area on the map">
      <p aria-live="polite">
        {drawing.picking
          ? `Click the map to add corners · ${count} so far`
          : `${count} ${count === 1 ? "corner" : "corners"} · paused`}
      </p>
      {count > 0 && !cornersInside(drawing.points, land.boundary) && (
        <p className="land-muted">Some corners are outside your land. Keep the area inside it.</p>
      )}
      {drawing.error && (
        <p className="land-alert" role="alert">
          {drawing.error}
        </p>
      )}
      <div className="land-row">
        {!drawing.picking && (
          <GlassButton size="sm" onClick={drawing.resume}>
            Add corners
          </GlassButton>
        )}
        <GlassButton size="sm" disabled={!count} onClick={drawing.undo}>
          Undo corner
        </GlassButton>
        <GlassButton size="sm" variant="primary" disabled={count < 3} onClick={onFinish}>
          Finish area
        </GlassButton>
        <GlassButton size="sm" variant="ghost" onClick={drawing.cancel}>
          Cancel
        </GlassButton>
      </div>
    </div>
  );
}

function StepEditor({
  step,
  index,
  draft,
  areas,
  drawing,
  land,
  onChange,
  onArea,
  onRemove,
  onFinishDrawing,
  checklist,
}: {
  step: Step;
  index: number;
  draft: ActionDraft;
  areas: Feature[];
  drawing: PlanDrawing;
  land: LandArea;
  onChange: (change: Partial<Step>) => void;
  onArea: (area: Area | null, feature?: Feature) => void;
  onRemove: () => void;
  onFinishDrawing: () => void;
  checklist: PlanProblems;
}) {
  const n = index + 1;
  const drawingHere = drawing.target?.kind === "step" && drawing.target.id === step.id;
  const at = (field: Parameters<typeof actionFieldId.step>[1]) => actionFieldId.step(index, field);
  const problem = (field: Parameters<typeof actionFieldId.step>[1]) => (
    <FieldProblem id={at(field)} checklist={checklist} />
  );
  return (
    <li className="plans-edit-step" style={{ "--step": stepColor(index) } as CSSProperties}>
      <div className="plans-edit-step__head">
        <span className="plans-step-badge" aria-hidden="true">
          {n}
        </span>
        <GlassInput
          {...checklist.field(at("title"))}
          aria-label={`Step ${n} name`}
          placeholder="What happens, e.g. Mow the meadow"
          required
          maxLength={200}
          value={step.title}
          onChange={(event) => onChange({ title: event.target.value })}
        />
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label={`Remove step ${n}`}
          disabled={draft.steps.length === 1}
          onClick={onRemove}
        >
          <X size={14} aria-hidden="true" />
        </GlassButton>
      </div>
      {problem("title")}
      <div className="plans-edit-step__where">
        <span className="plans-edit-step__area">
          {step.footprint ? shortArea(footprintAreaM2(step.footprint)) : "Whole area"}
        </span>
        {!drawingHere && (
          <GlassButton
            size="sm"
            disabled={!drawing.canDraw}
            leadingIcon={<PenLine size={13} aria-hidden="true" />}
            onClick={() => drawing.start({ kind: "step", id: step.id })}
          >
            {step.footprint ? "Redraw" : "Draw on map"}
          </GlassButton>
        )}
        {areas.length > 0 && (
          <GlassSelect
            aria-label={`Step ${n} mapped area`}
            value=""
            onChange={(event) => {
              const feature = areas.find((item) => item.id === event.target.value);
              if (feature && isArea(feature.geometry)) onArea(feature.geometry, feature);
            }}
          >
            <option value="">Use a mapped area…</option>
            {areas.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </GlassSelect>
        )}
        {step.footprint && (
          <button
            type="button"
            className="land-link"
            {...checklist.control(at("area"))}
            onClick={() => onArea(null)}
          >
            Use the whole area
          </button>
        )}
      </div>
      {problem("area")}
      {drawingHere && <DrawBar drawing={drawing} land={land} onFinish={onFinishDrawing} />}
      <div className="plans-edit-grid">
        <PlanField label="Start day">
          <GlassInput
            {...checklist.field(at("start"))}
            aria-label={`Step ${n} start day`}
            type="number"
            min={1}
            max={36501}
            required
            value={Number.isFinite(step.startDay) ? step.startDay + 1 : ""}
            onChange={(event) => onChange({ startDay: event.target.valueAsNumber - 1 })}
          />
        </PlanField>
        <PlanField label="Days">
          <GlassInput
            {...checklist.field(at("days"))}
            aria-label={`Step ${n} duration (days)`}
            type="number"
            min={1}
            max={3650}
            required
            value={Number.isFinite(step.days) ? step.days : ""}
            onChange={(event) => onChange({ days: event.target.valueAsNumber })}
          />
        </PlanField>
        <PlanField label={<>Cost ({draft.currency})</>}>
          <GlassInput
            {...checklist.field(at("cost"))}
            aria-label={`Step ${n} cost`}
            type="number"
            min={0}
            step="any"
            placeholder="Optional"
            value={step.estimatedCost ?? ""}
            onChange={(event) =>
              onChange({
                estimatedCost: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </PlanField>
      </div>
      {problem("start")}
      {problem("days")}
      {problem("cost")}
      {step.estimatedCost != null && (
        <PlanField label="Where the cost comes from">
          <GlassInput
            {...checklist.field(at("cost-basis"))}
            aria-label={`Step ${n} cost basis`}
            required
            maxLength={2000}
            placeholder="A quote, a rate × area…"
            value={step.costBasis ?? ""}
            onChange={(event) => onChange({ costBasis: event.target.value })}
          />
        </PlanField>
      )}
      {problem("cost-basis")}
      <PlanField label="Done when">
        <GlassInput
          {...checklist.field(at("success"))}
          aria-label={`Step ${n} success measure`}
          required
          maxLength={2000}
          placeholder="How you will know it worked"
          value={step.successMeasure}
          onChange={(event) => onChange({ successMeasure: event.target.value })}
        />
      </PlanField>
      {problem("success")}
      <details className="plans-more">
        <summary>More about step {n}</summary>
        <PlanField label="What the work involves">
          <GlassTextarea
            {...checklist.field(at("detail"))}
            aria-label={`Step ${n} description`}
            rows={2}
            maxLength={1000}
            value={step.detail ?? ""}
            onChange={(event) => onChange({ detail: event.target.value })}
          />
        </PlanField>
        {problem("detail")}
        <PlanField label="Resources (one per line)">
          <GlassTextarea
            {...checklist.field(at("resources"))}
            aria-label={`Step ${n} resources`}
            rows={2}
            value={(step.resources ?? []).join("\n")}
            onChange={(event) => onChange({ resources: event.target.value.split("\n") })}
          />
        </PlanField>
        {problem("resources")}
        {draft.steps.length > 1 && (
          <fieldset className="plans-checks">
            <legend>Starts after</legend>
            {draft.steps
              .filter((other) => other.id !== step.id)
              .map((other) => (
                <label key={other.id} className="plans-check">
                  <input
                    type="checkbox"
                    checked={step.dependsOn?.includes(other.id) ?? false}
                    onChange={(event) =>
                      onChange({
                        dependsOn: event.target.checked
                          ? [...(step.dependsOn ?? []), other.id]
                          : step.dependsOn?.filter((id) => id !== other.id),
                      })
                    }
                  />
                  {other.title || "Untitled step"}
                </label>
              ))}
          </fieldset>
        )}
      </details>
    </li>
  );
}

/** Evidence from a research investigation, attached to the plan. */
function EvidenceChooser({
  land,
  draft,
  setDraft,
}: {
  land: LandArea;
  draft: ActionDraft;
  setDraft: (draft: ActionDraft) => void;
}) {
  const investigations = useInvestigations(land.id);
  const [investigationId, setInvestigationId] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [source, setSource] = useState<LandEvidence | null>(null);
  const evidence = useInvestigation(investigationId, offset);
  const total = evidence.data?.page.totals.evidence ?? 0;
  return (
    <div className="plans-group">
      <PlanField label="Research investigation">
        <GlassSelect
          aria-label="Research investigation"
          value={investigationId ?? ""}
          onChange={(event) => {
            setInvestigationId(event.target.value || null);
            setOffset(0);
          }}
        >
          <option value="">Choose sources that support this plan</option>
          {investigations.data?.map((item) => (
            <option key={item.id} value={item.id}>
              {item.title}
            </option>
          ))}
        </GlassSelect>
      </PlanField>
      {evidence.isError && <p role="alert">Those sources could not be loaded.</p>}
      {evidence.data?.evidence.map((item) => (
        <label className="plans-check" key={item.id}>
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
            {item.title}{" "}
            <button type="button" className="land-link" onClick={() => setSource(item)}>
              Inspect source{item.document ? ` · page ${item.document.page}` : ""}
            </button>
          </span>
        </label>
      ))}
      {source && <EvidenceView evidence={source} onClose={() => setSource(null)} />}
      {(offset > 0 || total > offset + 100) && (
        <div className="land-row">
          <GlassButton
            size="sm"
            disabled={!offset}
            onClick={() => setOffset(Math.max(0, offset - 100))}
          >
            Previous sources
          </GlassButton>
          <GlassButton
            size="sm"
            disabled={total <= offset + 100}
            onClick={() => setOffset(offset + 100)}
          >
            More sources
          </GlassButton>
        </div>
      )}
      <p className="land-muted">
        {draft.evidenceIds?.length ?? 0} sources attached. Each stays traceable to its research.
      </p>
    </div>
  );
}

/**
 * Making or revising a plan: its name and goal, then its steps, each with where it happens
 * (drawn on the map, a mapped area, or the whole area), when, what it costs and how to tell it
 * worked; areas to avoid; and, folded away, sources, references, questions and assumptions.
 * The draft is kept in this browser as it changes.
 */
export function PlanEditor({
  land,
  scope,
  seed,
  onSaved,
  onClose,
}: {
  land: LandArea;
  scope: string;
  seed: PlanSeed;
  onSaved: (id: string) => void;
  /** Back to the plans; `discarded` when the draft was thrown away. */
  onClose: (discarded: boolean) => void;
}) {
  const [draft, setDraft] = useState<ActionDraft>(seed.draft);
  const [editing, setEditing] = useState(seed.editing);
  const [conflict, setConflict] = useState<Action | null>(seed.conflict);
  const [notice, setNotice] = useState(seed.notice);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showSaved, setShowSaved] = useState(false);
  const ticket = useRef(0);
  const scenarios = useScenarios(land);
  const features = useLandFeatures(land, true);
  const refresh = useRefreshPlans(land);
  const drawing = usePlanDrawing();
  const checklist = usePlanProblems(draft);
  const ids = actionFieldId;
  const problem = (id: string) => <FieldProblem id={id} checklist={checklist} />;
  const areas = useMemo(
    () => (features.data ?? []).filter((item) => isArea(item.geometry)),
    [features.data],
  );
  useEffect(
    () => () => {
      ticket.current += 1;
    },
    [],
  );

  const snapshot: ActionDraftSnapshot = { version: 1, landId: land.id, draft, editing };
  const [initial] = useState(() =>
    serializeActionDraft({ version: 1, landId: land.id, draft: seed.draft, editing: seed.editing }),
  );
  const { storageError, clear } = useDraftAutosave(
    actionDraftKey(scope, land.id),
    serializeActionDraft(snapshot),
    4_000_000,
  );

  // The map: the draft's step areas and areas to avoid (or the saved version when comparing),
  // and the outline being drawn.
  const saved = showSaved ? conflict : null;
  const shownSteps = (saved ?? draft).steps;
  const shownExclusions = (saved ?? draft).exclusions ?? [];
  const stepsKey = identityKey(shownSteps.map((step) => step.footprint));
  const avoidKey = identityKey(shownExclusions);
  // Redraw when areas change, not on every keystroke in a step's name.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const stepLayer = useMemo(() => stepsLayer(shownSteps), [stepsKey]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const avoid = useMemo(() => avoidLayer(shownExclusions), [avoidKey]);
  const work = useMemo(
    () => (saved ? workAreaLayer(saved.effectiveBoundary, saved.title) : null),
    [saved],
  );
  const drawColor =
    drawing.target?.kind === "avoid"
      ? AVOID_COLOR
      : stepColor(
          Math.max(
            0,
            draft.steps.findIndex(
              (step) => drawing.target?.kind === "step" && step.id === drawing.target.id,
            ),
          ),
        );
  const drawn = useMemo(() => drawingLayer(drawing.points, drawColor), [drawing.points, drawColor]);
  useLensLayer(work);
  useLensLayer(stepLayer);
  useLensLayer(avoid);
  useLensLayer(drawn);
  const pins = useMemo(
    () => stepPins((saved ?? draft).steps, saved?.effectiveBoundary ?? land.boundary),
    [saved, draft, land.boundary],
  );
  useLensPins("plans", pins);

  const updateStep = (index: number, change: Partial<Step>) =>
    setDraft((current) => ({
      ...current,
      steps: current.steps.map((step, i) => (i === index ? { ...step, ...change } : step)),
    }));
  const setStepArea = (index: number, area: Area | null, feature?: Feature) =>
    setDraft((current) => ({
      ...current,
      steps: current.steps.map((step, i) => (i === index ? { ...step, footprint: area } : step)),
      features: feature
        ? [
            ...(current.features ?? []).filter((item) => item.id !== feature.id),
            { id: feature.id, revision: feature.revision },
          ]
        : current.features,
    }));
  const finishDrawing = () => {
    const done = drawing.finish();
    if (!done) return;
    const { target, area } = done;
    if (target.kind === "avoid")
      setDraft((current) => ({ ...current, exclusions: [...(current.exclusions ?? []), area] }));
    else
      setDraft((current) => ({
        ...current,
        steps: current.steps.map((step) =>
          step.id === target.id ? { ...step, footprint: area } : step,
        ),
      }));
  };

  const save = async () => {
    if (conflict) return;
    const current = ++ticket.current;
    drawing.cancel();
    setBusy(true);
    setError(null);
    try {
      const payload = actionPayload(draft);
      const result = editing
        ? await unwrap(
            api.PUT("/api/v1/land/{land_id}/actions/{action_id}", {
              params: { path: { land_id: land.id, action_id: editing.id } },
              body: {
                ...payload,
                expectedRevision: editing.revision,
                note: "Revised in the Plans lens",
              },
            }),
          )
        : await unwrap(
            api.POST("/api/v1/land/{land_id}/actions", {
              params: { path: { land_id: land.id } },
              body: payload,
            }),
          );
      await refresh();
      if (ticket.current !== current) return;
      clear();
      onSaved(result.id);
    } catch (cause) {
      if (ticket.current !== current) return;
      setError(planError(cause));
      if (editing && cause instanceof ApiError && cause.status === 409) {
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
          /* Keep the original error; the draft stays in this browser. */
        }
      }
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };

  const scenarioList = scenarios.data ?? [];
  const preservedScenario =
    draft.scenario &&
    !scenarioList.some(
      (item) => item.id === draft.scenario?.id && item.revision === draft.scenario?.revision,
    );
  const avoidDrawing = drawing.target?.kind === "avoid";
  const reapproval = editing && editing.status !== "draft";
  const saveBlocked = conflict
    ? "Choose how to handle the changed plan above before saving."
    : null;

  return (
    <LensCard
      lens="plans"
      headline={editing ? "Revise this plan" : "Make a plan"}
      action={
        <GlassButton
          size="sm"
          variant="ghost"
          aria-label="Back to plans"
          leadingIcon={<ArrowLeft size={14} aria-hidden="true" />}
          onClick={() => {
            drawing.cancel();
            // An untouched new or revised plan leaves nothing behind; any change is kept.
            const untouched = !seed.recovered && serializeActionDraft(snapshot) === initial;
            if (untouched) clear();
            onClose(untouched);
          }}
        >
          Plans
        </GlassButton>
      }
    >
      <p className="land-lede">
        Add the steps in order. Draw where each one happens; a step you don&apos;t draw uses the
        whole area.
      </p>
      {notice && (
        <p className="land-muted" role="status">
          {notice}
        </p>
      )}
      {storageError && (
        <p className="land-alert" role="alert">
          {storageError}
        </p>
      )}
      {conflict && (
        <div className="land-note plans-conflict" role="region" aria-label="Review changed plan">
          <p>
            This plan was saved again while you worked: it is now revision {conflict.revision} (
            {conflict.status}). Your draft started from revision {editing?.revision} (
            {editing?.status}).
          </p>
          <details className="plans-more">
            <summary>What the saved revision says</summary>
            <PlanSummary action={conflict} />
          </details>
          <div className="land-row">
            <GlassButton size="sm" active={showSaved} onClick={() => setShowSaved(!showSaved)}>
              {showSaved ? "Show my draft on the map" : "Show saved work areas on map"}
            </GlassButton>
          </div>
          <div className="land-row">
            <GlassButton
              size="sm"
              variant="primary"
              disabled={busy}
              onClick={() => {
                setDraft((current) => ({ ...current, requestKey: crypto.randomUUID() }));
                setEditing(null);
                setConflict(null);
                setShowSaved(false);
                setError(null);
              }}
            >
              Keep my work as a new plan
            </GlassButton>
            <GlassButton
              size="sm"
              disabled={busy}
              onClick={() => {
                setEditing({
                  id: conflict.id,
                  landId: conflict.landId,
                  revision: conflict.revision,
                  status: conflict.status,
                  missionId: conflict.missionId,
                });
                setDraft(editablePlan(conflict));
                checklist.reset();
                setConflict(null);
                setShowSaved(false);
                setNotice(null);
                setError(null);
              }}
            >
              Load the latest plan
            </GlassButton>
          </div>
        </div>
      )}
      {draft.boundaryRevision !== land.revision && (
        <p className="land-note">
          This plan was made for an earlier outline of the land.{" "}
          <button
            type="button"
            className="land-link"
            onClick={() => setDraft({ ...draft, boundaryRevision: land.revision, scenario: null })}
          >
            Use the current outline
          </button>
          {draft.scenario ? " (this detaches its scenario)" : ""}
        </p>
      )}
      <form
        className="plans-edit"
        aria-label={editing ? "Revise plan" : "New plan"}
        // The checklist above Save replaces the browser's bubbles, which cannot reach a field
        // in a folded section. `required` stays on the fields for assistive technology.
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          if (conflict || !checklist.check()) return;
          void save();
        }}
      >
        <fieldset disabled={busy}>
          <PlanField label="Plan name">
            <GlassInput
              {...checklist.field(ids.title)}
              required
              maxLength={200}
              placeholder="e.g. Meadow restoration"
              value={draft.title}
              onChange={(event) => setDraft({ ...draft, title: event.target.value })}
            />
          </PlanField>
          {problem(ids.title)}
          <PlanField label="Goal">
            <GlassTextarea
              {...checklist.field(ids.objective)}
              required
              rows={2}
              maxLength={2000}
              placeholder="What the land should look like when this is done"
              value={draft.objective}
              onChange={(event) => setDraft({ ...draft, objective: event.target.value })}
            />
          </PlanField>
          {problem(ids.objective)}
          <PlanField label="Start date">
            <GlassInput
              {...checklist.field(ids.start)}
              required
              type="date"
              value={draft.startDate}
              onChange={(event) => setDraft({ ...draft, startDate: event.target.value })}
            />
          </PlanField>
          {problem(ids.start)}

          <h4 className="plans-edit__heading">Steps</h4>
          <ol className="plans-edit-steps">
            {draft.steps.map((step, index) => (
              <StepEditor
                key={step.id}
                step={step}
                index={index}
                draft={draft}
                areas={areas}
                drawing={drawing}
                land={land}
                onChange={(change) => updateStep(index, change)}
                onArea={(area, feature) => setStepArea(index, area, feature)}
                onRemove={() => {
                  if (drawing.target?.kind === "step" && drawing.target.id === step.id)
                    drawing.cancel();
                  setDraft({
                    ...draft,
                    steps: draft.steps
                      .filter((_, i) => i !== index)
                      .map((other) => ({
                        ...other,
                        dependsOn: other.dependsOn?.filter((id) => id !== step.id),
                      })),
                  });
                }}
                onFinishDrawing={finishDrawing}
                checklist={checklist}
              />
            ))}
          </ol>
          <div className="land-row">
            <GlassButton
              size="sm"
              disabled={draft.steps.length >= 250}
              leadingIcon={<Plus size={14} aria-hidden="true" />}
              onClick={() =>
                setDraft({
                  ...draft,
                  steps: [
                    ...draft.steps,
                    newStep(
                      Math.max(
                        0,
                        ...draft.steps
                          .map((step) => step.startDay + step.days)
                          .filter(Number.isFinite),
                      ),
                    ),
                  ],
                })
              }
            >
              Add step
            </GlassButton>
          </div>

          <h4 className="plans-edit__heading">Areas to avoid</h4>
          {(draft.exclusions ?? []).length > 0 && (
            <ul className="plans-avoid-list">
              {(draft.exclusions ?? []).map((area, index) => (
                <li key={index}>
                  <i className="plans-legend__avoid" aria-hidden="true" />
                  Area to avoid {index + 1} · {shortArea(footprintAreaM2(area))}
                  <GlassButton
                    iconOnly
                    size="sm"
                    variant="ghost"
                    aria-label={`Remove area to avoid ${index + 1}`}
                    onClick={() =>
                      setDraft({
                        ...draft,
                        exclusions: draft.exclusions?.filter((_, i) => i !== index),
                      })
                    }
                  >
                    <Trash2 size={14} aria-hidden="true" />
                  </GlassButton>
                </li>
              ))}
            </ul>
          )}
          <div className="land-row">
            {!avoidDrawing && (
              <GlassButton
                size="sm"
                disabled={!drawing.canDraw || (draft.exclusions?.length ?? 0) >= 50}
                leadingIcon={<PenLine size={13} aria-hidden="true" />}
                onClick={() => drawing.start({ kind: "avoid" })}
              >
                Draw an area to avoid
              </GlassButton>
            )}
            {areas.length > 0 && (
              <GlassSelect
                aria-label="Avoid a mapped area"
                disabled={(draft.exclusions?.length ?? 0) >= 50}
                value=""
                onChange={(event) => {
                  const feature = areas.find((item) => item.id === event.target.value);
                  if (feature && isArea(feature.geometry)) {
                    const geometry = feature.geometry;
                    setDraft({ ...draft, exclusions: [...(draft.exclusions ?? []), geometry] });
                  }
                }}
              >
                <option value="">Avoid a mapped area…</option>
                {areas.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </GlassSelect>
            )}
            {(draft.exclusions?.length ?? 0) > 1 && (
              <button
                type="button"
                className="land-link"
                {...checklist.control(ids.exclusions)}
                onClick={() => setDraft({ ...draft, exclusions: [] })}
              >
                Clear all
              </button>
            )}
          </div>
          {problem(ids.exclusions)}
          {avoidDrawing && <DrawBar drawing={drawing} land={land} onFinish={finishDrawing} />}
          {features.data?.length === 500 && (
            <p className="land-muted">Showing the first 500 mapped assets.</p>
          )}
          <PlanLegend
            steps={draft.steps.some((step) => step.footprint)}
            avoid={(draft.exclusions?.length ?? 0) > 0}
          />

          <LensDetails summary="Money, sources and references">
            <PlanField label="Currency">
              <GlassInput
                {...checklist.field(ids.currency)}
                required
                pattern="[A-Z]{3}"
                maxLength={3}
                value={draft.currency}
                onChange={(event) =>
                  setDraft({ ...draft, currency: event.target.value.toUpperCase() })
                }
              />
            </PlanField>
            {problem(ids.currency)}
            <PlanField label="Scenario behind this plan">
              <GlassSelect
                aria-label="Supporting scenario"
                value={draft.scenario ? `${draft.scenario.id}:${draft.scenario.revision}` : ""}
                onChange={(event) => {
                  const chosen = scenarioList.find(
                    (item) => `${item.id}:${item.revision}` === event.target.value,
                  );
                  setDraft({
                    ...draft,
                    scenario: chosen ? { id: chosen.id, revision: chosen.revision } : null,
                    boundaryRevision: chosen?.boundaryRevision ?? draft.boundaryRevision,
                  });
                }}
              >
                <option value="">No scenario</option>
                {draft.scenario && preservedScenario && (
                  <option value={`${draft.scenario.id}:${draft.scenario.revision}`}>
                    Kept scenario revision {draft.scenario.revision}
                  </option>
                )}
                {scenarioList.map((item) => (
                  <option key={item.id} value={`${item.id}:${item.revision}`}>
                    {item.name} · revision {item.revision}
                  </option>
                ))}
              </GlassSelect>
            </PlanField>
            <p className="land-muted">
              Choosing a scenario uses the land outline it was made for. Check the areas after
              changing it. Boundary revision {draft.boundaryRevision}.
            </p>
            <EvidenceChooser land={land} draft={draft} setDraft={setDraft} />
            <div className="plans-group" role="group" aria-label="Mapped assets this plan uses">
              <span className="land-muted">Mapped assets this plan uses</span>
              {features.isError && <p role="alert">Mapped assets could not be loaded.</p>}
              {features.data?.map((feature) => {
                const reference = draft.features?.find((item) => item.id === feature.id);
                return (
                  <div key={feature.id}>
                    <label className="plans-check">
                      <input
                        type="checkbox"
                        checked={Boolean(reference)}
                        onChange={(event) =>
                          setDraft({
                            ...draft,
                            features: event.target.checked
                              ? [
                                  ...(draft.features ?? []).filter(
                                    (item) => item.id !== feature.id,
                                  ),
                                  { id: feature.id, revision: feature.revision },
                                ]
                              : draft.features?.filter((item) => item.id !== feature.id),
                          })
                        }
                      />
                      {feature.name}
                    </label>
                    {reference && reference.revision !== feature.revision && (
                      <p className="land-muted">
                        This plan uses an earlier version.{" "}
                        <button
                          type="button"
                          className="land-link"
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
                  </div>
                );
              })}
              {features.data?.length === 0 && (
                <p className="land-muted">No mapped assets on this land yet.</p>
              )}
            </div>
          </LensDetails>

          <LensDetails summary="Questions to settle and assumptions">
            <p className="land-muted">A plan can be approved once every question is settled.</p>
            {(draft.constraints ?? []).map((constraint, index) => (
              <div className="plans-group" key={index}>
                <PlanField label={<>Question {index + 1}</>}>
                  <GlassTextarea
                    {...checklist.field(ids.constraint(index, "text"))}
                    aria-label={`Question ${index + 1}`}
                    required
                    rows={2}
                    maxLength={2000}
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
                </PlanField>
                {problem(ids.constraint(index, "text"))}
                <label className="plans-check">
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
                  Settled
                </label>
                {constraint.resolved && (
                  <PlanField label="How it was settled">
                    <GlassTextarea
                      {...checklist.field(ids.constraint(index, "resolution"))}
                      aria-label={`How question ${index + 1} was settled`}
                      required
                      rows={2}
                      maxLength={2000}
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
                  </PlanField>
                )}
                {problem(ids.constraint(index, "resolution"))}
                <button
                  type="button"
                  className="land-link"
                  onClick={() =>
                    setDraft({
                      ...draft,
                      constraints: draft.constraints?.filter((_, i) => i !== index),
                    })
                  }
                >
                  Remove question {index + 1}
                </button>
              </div>
            ))}
            <div className="land-row">
              <GlassButton
                size="sm"
                leadingIcon={<Plus size={14} aria-hidden="true" />}
                onClick={() =>
                  setDraft({
                    ...draft,
                    constraints: [...(draft.constraints ?? []), { text: "", resolved: false }],
                  })
                }
              >
                Add a question
              </GlassButton>
            </div>
            <PlanField label="Assumptions (one per line)">
              <GlassTextarea
                {...checklist.field(ids.assumptions)}
                rows={3}
                value={(draft.assumptions ?? []).join("\n")}
                onChange={(event) =>
                  setDraft({ ...draft, assumptions: event.target.value.split("\n") })
                }
              />
            </PlanField>
            {problem(ids.assumptions)}
          </LensDetails>
        </fieldset>
        {error && (
          <p className="land-alert" role="alert">
            {error}
          </p>
        )}
        {checklist.shown && <PlanProblemList problems={checklist.problems} />}
        {saveBlocked && (
          <p className="land-muted" id="plans-save-blocked">
            {saveBlocked}
          </p>
        )}
        {reapproval && (
          <p className="land-muted">Saving makes a new revision that needs approval again.</p>
        )}
        <div className="land-row plans-edit__actions">
          <GlassButton
            type="submit"
            variant="primary"
            loading={busy}
            disabled={busy || Boolean(conflict)}
            aria-describedby={
              saveBlocked ? "plans-save-blocked" : checklist.shown ? PROBLEMS_ID : undefined
            }
          >
            {editing ? "Save changes" : "Save plan"}
          </GlassButton>
          <GlassButton
            variant="ghost"
            disabled={busy}
            onClick={() => {
              drawing.cancel();
              clear();
              onClose(true);
            }}
          >
            Discard draft
          </GlassButton>
          <GlassButton
            size="sm"
            variant="ghost"
            className="plans-edit__download"
            leadingIcon={<Download size={13} aria-hidden="true" />}
            onClick={() => downloadActionDraft(snapshot)}
          >
            Download draft
          </GlassButton>
        </div>
      </form>
    </LensCard>
  );
}
