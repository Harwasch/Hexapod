import { ArrowLeft, PenLine, Scan, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { PlanDraft } from "@twin/contracts";

import { usePlannerStatus } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { planningEdited } from "@/lib/planningMetrics";
import {
  areaFromMeasurement,
  isAreaZone,
  nextAreaId,
  viewFootprint,
  zoneFromFootprint,
} from "@/missions/areas";
import { describeConflict, planConflicts } from "@/missions/conflicts";
import { diffDrafts, idLabel, overlayFor } from "@/missions/planDraft";
import type { Project } from "@/missions/types";
import { useMeasurements } from "@/state/measurements";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";
import { useViewer } from "@/state/viewer";

import { PlanSchedule } from "./PlanSchedule";
import { approveDraft, startPlanDraft } from "./planDrafting";
import { useMissionActions } from "./useMissionActions";

const EXAMPLES = [
  "Clear the star thistle from Z-14 and Z-21 with two mowers before seed set",
  "Inspect the pipeline corridor monthly and flag erosion",
  "Cut fire breaks along the north fence this week",
];

/** Goal → draft → review → approve. The agent drafts; the operator decides. */
export function PlanComposer({ project }: { project: Project }) {
  const composer = useMission((s) => s.composer);
  const update = useMission((s) => s.updateComposer);
  const close = useMission((s) => s.closeComposer);
  const addArea = useMission((s) => s.addArea);
  const removeArea = useMission((s) => s.removeArea);
  const planner = usePlannerStatus();
  const scene = useScene();
  const metersPerPixel = useViewer((s) => s.camera.metersPerPixel);
  const measureMode = useUi((s) => s.measureMode);
  const setMeasureMode = useUi((s) => s.setMeasureMode);
  // Drawing an area borrows the measurement tool's polygon drawing; the finished polygon
  // becomes a zone and the measurement is discarded. The subscription lives in a ref so no
  // state is set from inside an effect.
  const drawing = useRef<{ since: number; stop: () => void } | null>(null);
  const [armed, setArmed] = useState(false);
  const areaCount = project.zones.filter(isAreaZone).length;
  const isDrawing = measureMode === "area" && armed;
  const adoptArea = (zone: (typeof project.zones)[number]) => {
    addArea(project.id, zone);
    const current = useMission.getState().composer;
    if (current && !current.zoneIds.includes(zone.id))
      update({ zoneIds: [...current.zoneIds, zone.id] });
  };
  const stopDrawing = () => {
    drawing.current?.stop();
    drawing.current = null;
    setArmed(false);
  };
  useEffect(() => stopDrawing, []);
  const startDrawing = () => {
    if (drawing.current) {
      stopDrawing();
      setMeasureMode(null);
      return;
    }
    const since = Date.now();
    const unsubscribeMeasurements = useMeasurements.subscribe((state) => {
      const done = state.items.find((m) => m.mode === "area" && m.complete && m.createdAt >= since);
      if (!done) return;
      const id = nextAreaId(useMission.getState().project?.zones ?? []);
      const zone = areaFromMeasurement(done, id, `Drawn area ${id.slice(2)}`);
      stopDrawing();
      useMeasurements.getState().remove(done.id);
      scene?.measurement.remove(done.id);
      setMeasureMode(null);
      if (zone) adoptArea(zone);
    });
    // Esc (or another tool) leaving area mode ends the drawing.
    const unsubscribeMode = useUi.subscribe((state) => {
      if (state.measureMode !== "area") stopDrawing();
    });
    drawing.current = {
      since,
      stop: () => {
        unsubscribeMeasurements();
        unsubscribeMode();
      },
    };
    setArmed(true);
    setMeasureMode("area");
  };
  const useView = () => {
    const center = scene?.camera.viewCenter();
    if (!center) return;
    const id = nextAreaId(project.zones);
    adoptArea(
      zoneFromFootprint(
        id,
        `View area ${id.slice(2)}`,
        viewFootprint(center, metersPerPixel, { width: center.width, height: center.height }),
      ),
    );
  };

  // The goal field takes focus when the composer opens (the operator came here to type).
  const goalRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    goalRef.current?.focus();
  }, []);
  if (!composer) return null;
  const drafting = composer.status === "drafting";
  const toggle = (key: "zoneIds" | "machineIds", id: string) => {
    const list = composer[key];
    update({ [key]: list.includes(id) ? list.filter((x) => x !== id) : [...list, id] });
  };
  const submit = () => {
    if (drafting) return;
    void startPlanDraft(composer.goal, {
      zoneIds: composer.zoneIds,
      machineIds: composer.machineIds,
    });
  };
  return (
    <div className="mc-composer" data-testid="plan-composer" aria-busy={drafting}>
      <p className="sr-only" aria-live="polite">
        {composer.status === "drafting"
          ? "Drafting the plan"
          : composer.status === "ready"
            ? "Draft ready for review"
            : composer.status === "error"
              ? "Drafting failed"
              : ""}
      </p>
      <div className="mc-window__head mc-window__head--column">
        <button type="button" className="mc-link" onClick={close}>
          <ArrowLeft size={12} aria-hidden="true" /> All plans
        </button>
        <div className="mc-row mc-row--between">
          <div>
            <div className="mc-window__title">
              {composer.replacePlanId ? "Revise plan" : "New plan"}
            </div>
            <div className="mc-window__sub">
              {planner.data?.provider === "claude"
                ? `Drafted by Claude · ${planner.data.model ?? ""}`
                : "Rule-based drafts · set ANTHROPIC_API_KEY on the API for Claude"}
            </div>
          </div>
        </div>
      </div>
      <div className="mc-window__body mc-composer__body">
        <form
          className="mc-composer__goal"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <label className="mc-eyebrow" htmlFor="plan-goal">
            GOAL
          </label>
          <textarea
            id="plan-goal"
            className="mc-textarea"
            rows={3}
            ref={goalRef}
            value={composer.goal}
            placeholder="What should the fleet achieve? Name zones, machines and a deadline if you have them."
            onChange={(e) => update({ goal: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            disabled={drafting}
            data-testid="plan-goal"
          />
          {composer.status === "idle" && (
            <div className="mc-tags" aria-label="Examples">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="mc-tag mc-tag--ghost"
                  onClick={() => update({ goal: example })}
                >
                  {example}
                </button>
              ))}
            </div>
          )}
          <div className="mc-composer__scope">
            <div>
              <div className="mc-row mc-row--between">
                <span className="mc-eyebrow">WHERE</span>
                <span className="mc-row" style={{ gap: "0.35rem" }}>
                  <button
                    type="button"
                    className="mc-btn mc-btn--sm"
                    onClick={useView}
                    disabled={drafting || !scene}
                    title="Make an area from the ground the view is looking at"
                    data-testid="area-from-view"
                  >
                    <Scan size={12} aria-hidden="true" /> Use current view
                  </button>
                  <button
                    type="button"
                    className={`mc-btn mc-btn--sm ${isDrawing ? "is-on" : ""}`}
                    onClick={startDrawing}
                    disabled={drafting || !scene}
                    aria-pressed={isDrawing}
                    title="Click corners on the map; double-click to finish"
                    data-testid="area-draw"
                  >
                    <PenLine size={12} aria-hidden="true" />{" "}
                    {isDrawing ? "Drawing… (Esc to cancel)" : "Draw area"}
                  </button>
                </span>
              </div>
              {project.zones.length === 0 && (
                <p className="mc-muted" style={{ fontSize: "12px", marginTop: "0.35rem" }}>
                  No zones here yet. Use the current view or draw an area on the map; plans can be
                  made anywhere.
                </p>
              )}
              <div className="mc-tags">
                {project.zones.map((zone) => (
                  <span key={zone.id} className="mc-tag-group">
                    <button
                      type="button"
                      className={`mc-tag ${composer.zoneIds.includes(zone.id) ? "is-on" : ""}`}
                      aria-pressed={composer.zoneIds.includes(zone.id)}
                      onClick={() => toggle("zoneIds", zone.id)}
                      disabled={drafting}
                      data-testid={`compose-zone-${zone.id}`}
                    >
                      <span className={`mc-dot mc-dot--${zone.tone}`} aria-hidden="true" />
                      {idLabel(zone.id, zone.name)}
                      {isAreaZone(zone) && (
                        <span className="mc-muted"> · {zone.acres.toLocaleString()} ac</span>
                      )}
                    </button>
                    {isAreaZone(zone) && (
                      <button
                        type="button"
                        className="mc-tag mc-tag--icon"
                        aria-label={`Remove ${zone.id}`}
                        onClick={() => {
                          removeArea(project.id, zone.id);
                          update({ zoneIds: composer.zoneIds.filter((z) => z !== zone.id) });
                        }}
                        disabled={drafting}
                      >
                        <X size={11} aria-hidden="true" />
                      </button>
                    )}
                  </span>
                ))}
              </div>
              {areaCount > 0 && (
                <p className="mc-muted" style={{ fontSize: "11.5px", marginTop: "0.3rem" }}>
                  Drawn areas are saved with the plans that cover them.
                </p>
              )}
            </div>
            <div>
              <span className="mc-eyebrow">MACHINES</span>
              {project.machines.length === 0 && (
                <p className="mc-muted" style={{ fontSize: "12px", marginTop: "0.35rem" }}>
                  No machines are registered for this site yet. The agent drafts the work and
                  schedule; assignments come once a robot bridge registers the fleet.
                </p>
              )}
              <div className="mc-tags">
                {project.machines.map((machine) => (
                  <button
                    key={machine.id}
                    type="button"
                    className={`mc-tag ${composer.machineIds.includes(machine.id) ? "is-on" : ""}`}
                    aria-pressed={composer.machineIds.includes(machine.id)}
                    onClick={() => toggle("machineIds", machine.id)}
                    disabled={drafting}
                    data-testid={`compose-machine-${machine.id}`}
                  >
                    <span className={`mc-dot mc-dot--${machine.status}`} aria-hidden="true" />
                    {machine.id}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="mc-actions">
            <button
              type="submit"
              className="mc-btn mc-btn--accent"
              disabled={drafting || composer.goal.trim().length < 3}
              data-testid="plan-draft-submit"
            >
              <Sparkles size={13} aria-hidden="true" />{" "}
              {drafting ? "Drafting…" : composer.draft ? "Draft again" : "Draft with the agent"}
            </button>
          </div>
        </form>
        {composer.status === "error" && (
          <div className="mc-note mc-note--warn" role="alert">
            {composer.error}
          </div>
        )}
        {composer.draft && composer.status !== "drafting" && (
          <DraftReview draft={composer.draft} project={project} goal={composer.goal} />
        )}
      </div>
    </div>
  );
}

function DraftReview({
  draft,
  project,
  goal,
}: {
  draft: PlanDraft;
  project: Project;
  goal: string;
}) {
  const composer = useMission((s) => s.composer);
  const update = useMission((s) => s.updateComposer);
  const scene = useScene();
  const { showPlanOnMap, clearPlanOverlay } = useMissionActions();
  const [refinement, setRefinement] = useState("");
  const changes = composer?.previousDraft ? diffDrafts(composer.previousDraft, draft) : [];
  const conflicts = planConflicts(
    {
      id: composer?.replacePlanId ?? undefined,
      startDate: draft.startDate,
      zoneIds: draft.zoneIds,
      machineIds: draft.machineIds,
      steps: draft.steps ?? [],
    },
    project.plans,
  );
  // The draft is on the map as soon as it exists: zones, passes and route, in machine colours.
  // The camera fits the zones when the set of zones changes; a redraft that only moves
  // machines redraws in place.
  const zoneKey = draft.zoneIds.join(",");
  const overlayKey = `${zoneKey}|${(draft.steps ?? []).map((s) => `${s.zoneId}:${s.machineIds.join("+")}`).join(";")}`;
  const flownFor = useRef<string | null>(null);
  useEffect(() => {
    if (flownFor.current === zoneKey) {
      scene?.mission.showPlan(
        overlayFor({ zoneIds: draft.zoneIds, machineIds: draft.machineIds, steps: draft.steps }),
      );
    } else {
      flownFor.current = zoneKey;
      showPlanOnMap(draft);
    }
    return () => clearPlanOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- redraw only when the scope changes
  }, [overlayKey, showPlanOnMap, clearPlanOverlay]);
  const colorFor = (id: string) => scene?.mission.machineColor(id).toCssColorString() ?? "#7fd8c0";
  const zoneName = (id: string | null | undefined) =>
    id ? (project.zones.find((z) => z.id === id)?.name ?? id) : null;
  const refine = () => {
    const text = refinement.trim();
    if (!text) return;
    setRefinement("");
    void startPlanDraft(goal, {
      zoneIds: composer?.zoneIds ?? draft.zoneIds,
      machineIds: composer?.machineIds ?? draft.machineIds,
      refinement: text,
    });
  };
  const [approving, setApproving] = useState(false);
  const onApprove = () => {
    if (approving) return;
    setApproving(true);
    void approveDraft({ ...draft, title: draft.title.trim() || "New plan" }, goal).finally(() =>
      setApproving(false),
    );
  };
  return (
    <section className="mc-review" data-testid="plan-review" aria-label="Plan draft">
      <div className="mc-source">
        <span
          className={`mc-dot mc-dot--glow ${draft.source === "claude" ? "mc-dot--run" : "mc-dot--neutral"}`}
          aria-hidden="true"
        />
        <span>{draft.note}</span>
      </div>
      <input
        className="mc-input mc-review__title"
        value={draft.title}
        aria-label="Plan title"
        onChange={(e) => {
          planningEdited();
          update({ draft: { ...draft, title: e.target.value } });
        }}
      />
      <textarea
        className="mc-textarea"
        rows={3}
        value={draft.objective}
        aria-label="Objective"
        onChange={(e) => {
          planningEdited();
          update({ draft: { ...draft, objective: e.target.value } });
        }}
      />
      <div className="mc-stats mc-stats--tight">
        <div className="mc-stat">
          <div className="mc-stat__value">{Math.round(draft.estimates.acres)}</div>
          <div className="mc-stat__label">Acres</div>
        </div>
        <div className="mc-stat">
          <div className="mc-stat__value">{Math.round(draft.estimates.machineHours)} h</div>
          <div className="mc-stat__label">Machine-hours</div>
        </div>
        <div className="mc-stat">
          <div className="mc-stat__value">
            {draft.cadence === "once" ? `${draft.estimates.calendarDays} d` : draft.cadence}
          </div>
          <div className="mc-stat__label">{draft.cadence === "once" ? "Duration" : "Cadence"}</div>
        </div>
      </div>
      <div className="mc-review__scope mc-mono">
        <span>{draft.zoneIds.join(", ") || "no zones"}</span>
        <span>{draft.machineIds.join(", ") || "no machines"}</span>
        <span>
          {draft.startDate}
          {draft.endDate ? ` → ${draft.endDate}` : " → ongoing"}
        </span>
      </div>
      {changes.length > 0 && (
        <section data-testid="plan-changes">
          <span className="mc-eyebrow">WHAT CHANGED</span>
          <ul className="mc-list">
            {changes.map((change) => (
              <li key={change} className="mc-note mc-note--agent">
                {change}
              </li>
            ))}
          </ul>
        </section>
      )}
      {(draft.steps ?? []).length > 0 && (
        <section>
          <span className="mc-eyebrow">SCHEDULE</span>
          <PlanSchedule steps={draft.steps ?? []} startDate={draft.startDate} colorFor={colorFor} />
        </section>
      )}
      <section>
        <span className="mc-eyebrow">STEPS</span>
        <ol className="mc-steps">
          {(draft.steps ?? []).map((step, i) => (
            <li key={`${i}-${step.title}`} className="mc-step">
              <div className="mc-step__title">{step.title}</div>
              <div className="mc-step__meta mc-mono">
                {[step.when, zoneName(step.zoneId), step.machineIds.join(" ")]
                  .filter(Boolean)
                  .join(" · ")}
              </div>
              {step.detail && <div className="mc-step__detail">{step.detail}</div>}
            </li>
          ))}
        </ol>
      </section>
      {(draft.assumptions ?? []).length > 0 && (
        <section>
          <span className="mc-eyebrow">ASSUMPTIONS BEHIND THE ESTIMATE</span>
          <ul className="mc-list mc-list--plain">
            {(draft.assumptions ?? []).map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </section>
      )}
      {conflicts.length > 0 && (
        <section data-testid="plan-conflicts">
          <span className="mc-eyebrow">CONFLICTS WITH OTHER PLANS</span>
          <ul className="mc-list">
            {conflicts.map((c) => (
              <li key={`${c.kind}-${c.id}-${c.planId}`} className="mc-note mc-note--warn">
                {describeConflict(c)}
              </li>
            ))}
          </ul>
        </section>
      )}
      {draft.risks.length > 0 && (
        <section>
          <span className="mc-eyebrow">RISKS</span>
          <ul className="mc-list">
            {draft.risks.map((risk) => (
              <li key={risk} className="mc-note mc-note--warn">
                {risk}
              </li>
            ))}
          </ul>
        </section>
      )}
      {draft.questions.length > 0 && (
        <section>
          <span className="mc-eyebrow">THE AGENT ASKS</span>
          <ul className="mc-list">
            {draft.questions.map((q) => (
              <li key={q} className="mc-note mc-note--agent">
                {q}
              </li>
            ))}
          </ul>
        </section>
      )}
      <form
        className="mc-refine"
        onSubmit={(e) => {
          e.preventDefault();
          refine();
        }}
      >
        <input
          className="mc-input"
          value={refinement}
          placeholder="Answer or adjust — “use three machines”, “skip Z-21”, “finish by October”"
          aria-label="Refine the draft"
          onChange={(e) => setRefinement(e.target.value)}
          data-testid="plan-refine"
        />
        <button type="submit" className="mc-btn mc-btn--sm" disabled={!refinement.trim()}>
          Redraft
        </button>
      </form>
      <div className="mc-actions">
        <button type="button" className="mc-btn" onClick={() => showPlanOnMap(draft)}>
          Fit on map
        </button>
        <button
          type="button"
          className="mc-btn mc-btn--accent"
          onClick={onApprove}
          disabled={draft.zoneIds.length === 0 || approving}
          data-testid="plan-approve"
        >
          {approving ? "Saving…" : conflicts.length ? "Approve despite conflicts" : "Approve plan"}
        </button>
      </div>
    </section>
  );
}
