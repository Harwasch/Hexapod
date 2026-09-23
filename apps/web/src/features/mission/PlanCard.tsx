import { ArrowLeft, Crosshair, PenLine, Scan } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { PlanAnswerValue, PlanClarification, PlanDraft } from "@twin/contracts";

import { usePlannerStatus } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { planningEdited } from "@/lib/planningMetrics";
import { areaFromMeasurement, isAreaZone, nextAreaId } from "@/missions/areas";
import { describeConflict, planConflicts } from "@/missions/conflicts";
import { diffDrafts, idLabel, overlayFor } from "@/missions/planDraft";
import type { Project, Zone } from "@/missions/types";
import { useMeasurements } from "@/state/measurements";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

import { PlanSchedule } from "./PlanSchedule";
import { approveDraft, startPlanDraft } from "./planDrafting";
import { groundFromView, waitForGround } from "./planFlow";
import { useMissionActions } from "./useMissionActions";

const EXAMPLES = [
  "3D scan this field into a gaussian splat",
  "Mow the orchard by Friday with two mowers",
  "Survey the lake shore for erosion every month",
];

/** How long after a corner is dropped the plan redrafts for the new outline. */
const RESHAPE_REDRAFT_MS = 800;

/** Puts words in the command bar; the bar listens. */
function sayInBar(text: string): void {
  window.dispatchEvent(new CustomEvent("twin:bar", { detail: text }));
}

/**
 * The plan as the agent made it, on a card beside the map: the ground it found (editable on
 * the map), the numbers, the schedule and steps, what it assumed (tap a chip to change), and
 * Approve. Everything else is said in the bar below.
 */
export function PlanCard({ project }: { project: Project }) {
  const composer = useMission((s) => s.composer);
  const update = useMission((s) => s.updateComposer);
  const close = useMission((s) => s.closeComposer);
  const addArea = useMission((s) => s.addArea);
  const planner = usePlannerStatus();
  const scene = useScene();
  const setMeasureMode = useUi((s) => s.setMeasureMode);

  // Drawing borrows the measurement area tool: the next completed area becomes the ground.
  const drawing = useRef<(() => void) | null>(null);
  const stopDrawing = () => {
    drawing.current?.();
    drawing.current = null;
  };
  useEffect(() => stopDrawing, []);
  const adopt = (zone: Zone, note: string) => {
    addArea(project.id, zone);
    const current = useMission.getState().composer;
    if (!current) return;
    update({
      zoneIds: [zone.id],
      ground: { source: "rough", note },
      status: current.goal.trim().length >= 3 ? current.status : "idle",
    });
    if (current.goal.trim().length >= 3) void startPlanDraft(current.goal, { zoneIds: [zone.id] });
  };
  const startDrawing = () => {
    stopDrawing();
    scene?.areas.cancelPick();
    const known = new Set(useMeasurements.getState().items.map((m) => m.id));
    const unsubscribe = useMeasurements.subscribe((state) => {
      const done = state.items.find((m) => m.mode === "area" && m.complete && !known.has(m.id));
      if (!done) return;
      const id = nextAreaId(useMission.getState().project?.zones ?? []);
      const zone = areaFromMeasurement(done, id, `Drawn area ${id.slice(2)}`);
      stopDrawing();
      useMeasurements.getState().remove(done.id);
      scene?.measurement.remove(done.id);
      setMeasureMode(null);
      if (zone) adopt(zone, `${zone.name}, ${zone.acres.toLocaleString()} ac, as you drew it.`);
    });
    const unsubscribeMode = useUi.subscribe((state) => {
      if (state.measureMode !== "area") stopDrawing();
    });
    drawing.current = () => {
      unsubscribe();
      unsubscribeMode();
    };
    setMeasureMode("area");
  };
  const useView = () => {
    if (!scene) return;
    scene.areas.cancelPick();
    const zone = groundFromView(scene);
    if (zone) adopt(zone, `${zone.acres.toLocaleString()} ac in view; drag the corners to fit.`);
  };
  const pickAgain = () => {
    const current = useMission.getState().composer;
    if (!scene || !current) return;
    update({ status: "awaiting-ground", zoneIds: [], ground: null });
    void waitForGround(current.goal, scene);
  };

  // The ground's corners are on the map for as long as the card is open; a dropped corner
  // redrafts the plan for the new outline.
  const groundId = composer?.zoneIds.find((id) => isAreaZone({ id })) ?? null;
  const groundZone = project.zones.find((z) => z.id === groundId) ?? null;
  const groundFootprint = groundZone?.footprint;
  useEffect(() => {
    if (!scene) return;
    if (!groundId || !groundFootprint) {
      scene.areas.edit(null);
      return;
    }
    if (scene.areas.editing === groundId) scene.areas.update(groundFootprint);
    else scene.areas.edit(groundId, groundFootprint);
  }, [scene, groundId, groundFootprint]);
  useEffect(() => {
    if (!scene) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const off = scene.events.on("area-edit", ({ zoneId }) => {
      const current = useMission.getState().composer;
      if (!current?.draft || !current.zoneIds.includes(zoneId)) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        void startPlanDraft(current.goal, { zoneIds: current.zoneIds, reshaped: true });
      }, RESHAPE_REDRAFT_MS);
    });
    return () => {
      off();
      if (timer) clearTimeout(timer);
    };
  }, [scene]);
  // Only the handles leave with the card; a pending click is the flow's to cancel (React's
  // development double-mount would otherwise cancel it at once).
  useEffect(() => () => scene?.areas.edit(null), [scene]);

  if (!composer) return null;
  const status = composer.status;
  const claude = planner.data?.provider === "claude";
  return (
    <div className="mc-composer" data-testid="plan-card" aria-busy={status === "drafting"}>
      <p className="sr-only" aria-live="polite">
        {status === "drafting"
          ? "Drafting the plan"
          : status === "ready"
            ? "Draft ready for review"
            : status === "error"
              ? "Drafting failed"
              : ""}
      </p>
      <div className="mc-window__head mc-window__head--column">
        <button
          type="button"
          className="mc-link"
          onClick={() => {
            scene?.areas.cancelPick();
            close();
          }}
        >
          <ArrowLeft size={12} aria-hidden="true" /> All plans
        </button>
        <div>
          <div className="mc-window__title">
            {composer.replacePlanId ? "Revise plan" : (composer.draft?.title ?? "New plan")}
          </div>
          <div className="mc-window__sub" data-testid="plan-source">
            {claude
              ? `Planned by Claude · ${planner.data?.model ?? ""}`
              : "Rule-based planner · the API can't see ANTHROPIC_API_KEY"}
          </div>
        </div>
      </div>
      <div className="mc-window__body mc-composer__body">
        {status === "idle" && !composer.draft && (
          <div className="mc-prompt" data-testid="plan-idle">
            <div className="mc-prompt__title">Tell me what to do and where.</div>
            <p className="mc-muted">
              One sentence in the bar below. I find the ground, draft the mission with sensible
              defaults, and you change what you disagree with.
            </p>
            <div className="mc-tags" aria-label="Examples">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="mc-tag mc-tag--ghost"
                  onClick={() => sayInBar(example)}
                >
                  {example}
                </button>
              ))}
            </div>
            {composer.goal.trim().length >= 3 && !groundZone && (
              <div className="mc-actions">
                <button
                  type="button"
                  className="mc-btn mc-btn--accent"
                  onClick={pickAgain}
                  data-testid="ground-pick"
                >
                  <Crosshair size={13} aria-hidden="true" /> Click the ground on the map
                </button>
              </div>
            )}
          </div>
        )}
        {status === "awaiting-ground" && (
          <div className="mc-prompt mc-prompt--live" data-testid="plan-awaiting-ground">
            <Crosshair size={22} aria-hidden="true" className="mc-prompt__icon" />
            <div className="mc-prompt__title">Click the ground you mean on the map.</div>
            <p className="mc-muted">
              I'll outline the field, pond or lot under your click and draft “{composer.goal}”. Esc
              to cancel.
            </p>
            <div className="mc-actions mc-actions--start">
              <button
                type="button"
                className="mc-btn mc-btn--sm"
                onClick={startDrawing}
                data-testid="ground-draw"
              >
                <PenLine size={12} aria-hidden="true" /> Draw it instead
              </button>
              <button
                type="button"
                className="mc-btn mc-btn--sm"
                onClick={useView}
                data-testid="ground-view"
              >
                <Scan size={12} aria-hidden="true" /> Use what's in view
              </button>
            </div>
          </div>
        )}
        {status === "locating" && (
          <div className="mc-prompt mc-prompt--live" data-testid="plan-locating">
            <span className="mc-spinner" aria-hidden="true" />
            <div className="mc-prompt__title">Looking at the ground there…</div>
            <p className="mc-muted">Map data first, then the imagery if nothing is mapped.</p>
          </div>
        )}
        {groundZone && status !== "awaiting-ground" && (
          <section className="mc-ground" data-testid="plan-ground">
            <span className="mc-eyebrow">GROUND</span>
            <div className="mc-ground__row">
              <span className="mc-ground__name">
                <span className={`mc-dot mc-dot--${groundZone.tone}`} aria-hidden="true" />
                {idLabel(groundZone.id, groundZone.name)}
                <span className="mc-muted"> · {groundZone.acres.toLocaleString()} ac</span>
              </span>
              <span className="mc-row" style={{ gap: "0.3rem" }}>
                <button
                  type="button"
                  className="mc-btn mc-btn--sm"
                  onClick={pickAgain}
                  disabled={!scene || status === "drafting"}
                  data-testid="ground-pick-again"
                >
                  <Crosshair size={12} aria-hidden="true" /> Pick again
                </button>
                <button
                  type="button"
                  className="mc-btn mc-btn--sm"
                  onClick={startDrawing}
                  disabled={!scene || status === "drafting"}
                  data-testid="ground-draw"
                >
                  <PenLine size={12} aria-hidden="true" /> Draw
                </button>
              </span>
            </div>
            <p className="mc-muted mc-ground__note">
              {composer.ground?.note ?? groundZone.note} Corners are on the map: drag them and the
              plan follows.
            </p>
          </section>
        )}
        {status === "drafting" && (
          <div className="mc-prompt mc-prompt--live" data-testid="plan-drafting">
            <span className="mc-spinner" aria-hidden="true" />
            <div className="mc-prompt__title">Drafting “{composer.goal}”…</div>
          </div>
        )}
        {status === "error" && (
          <div className="mc-note mc-note--warn" role="alert">
            {composer.error}
            <div className="mc-actions mc-actions--start">
              <button
                type="button"
                className="mc-btn mc-btn--sm"
                onClick={() => void startPlanDraft(composer.goal, { zoneIds: composer.zoneIds })}
              >
                Try again
              </button>
            </div>
          </div>
        )}
        {composer.draft && status === "ready" && (
          <DraftReview draft={composer.draft} project={project} goal={composer.goal} />
        )}
      </div>
    </div>
  );
}

/** The draft's open questions plus every earlier one that was answered, in first-seen order. */
function assumed(
  current: PlanClarification[],
  asked: Record<string, PlanClarification>,
  answers: Record<string, PlanAnswerValue>,
): PlanClarification[] {
  const byId = new Map<string, PlanClarification>();
  for (const [id, c] of Object.entries(asked)) if (id in answers) byId.set(id, c);
  for (const c of current) byId.set(c.id, c);
  return [...byId.values()];
}

/** What the agent assumed, as chips; tapping one opens its alternatives in place. */
function Assumptions({
  items,
  answers,
  onChange,
  disabled,
}: {
  items: PlanClarification[];
  answers: Record<string, PlanAnswerValue>;
  onChange: (id: string, value: PlanAnswerValue) => void;
  disabled: boolean;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [rangeValue, setRangeValue] = useState<number | null>(null);
  const shown = items.filter((c) => c.kind !== "area");
  if (shown.length === 0) return null;
  const open = shown.find((c) => c.id === openId) ?? null;
  const current = (c: PlanClarification): PlanAnswerValue | null =>
    answers[c.id] ?? c.default ?? null;
  const labelFor = (c: PlanClarification): string => {
    const value = current(c);
    if (c.kind === "range") return `${String(value ?? c.min ?? 0)} ${c.unit ?? ""}`.trim();
    return (c.options ?? []).find((o) => o.value === String(value))?.label ?? String(value ?? "");
  };
  return (
    <section data-testid="plan-assumed">
      <span className="mc-eyebrow">I ASSUMED · TAP TO CHANGE</span>
      <div className="mc-tags">
        {shown.map((c) => (
          <button
            key={c.id}
            type="button"
            className={`mc-tag mc-assume ${openId === c.id ? "is-on" : ""}`}
            aria-pressed={openId === c.id}
            title={c.question}
            onClick={() => {
              setOpenId(openId === c.id ? null : c.id);
              setRangeValue(null);
            }}
            disabled={disabled}
            data-testid={`assume-${c.id}`}
          >
            {labelFor(c)}
          </button>
        ))}
      </div>
      {open && (
        <div className="mc-assume__panel" data-testid={`assume-${open.id}-panel`}>
          <div className="mc-clarify__question">{open.question}</div>
          {open.why && <div className="mc-clarify__why">{open.why}</div>}
          {open.kind === "range" ? (
            <div className="mc-stepper__range">
              <label className="mc-clarify__range">
                <input
                  type="range"
                  min={open.min ?? 0}
                  max={open.max ?? 100}
                  step={open.step ?? 1}
                  value={rangeValue ?? Number(current(open) ?? open.min ?? 0)}
                  onChange={(e) => setRangeValue(Number(e.target.value))}
                  disabled={disabled}
                  aria-label={open.question}
                  data-testid={`assume-${open.id}-range`}
                />
                <span className="mc-mono mc-clarify__value">
                  {rangeValue ?? Number(current(open) ?? open.min ?? 0)} {open.unit ?? ""}
                </span>
              </label>
              <button
                type="button"
                className="mc-btn mc-btn--sm mc-btn--accent"
                onClick={() => {
                  onChange(open.id, rangeValue ?? Number(current(open) ?? open.min ?? 0));
                  setOpenId(null);
                }}
                disabled={disabled}
                data-testid={`assume-${open.id}-apply`}
              >
                Update the plan
              </button>
            </div>
          ) : (
            <div className="mc-tags">
              {(open.options ?? []).map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`mc-tag ${String(current(open)) === option.value ? "is-on" : ""}`}
                  aria-pressed={String(current(open)) === option.value}
                  onClick={() => {
                    onChange(open.id, option.value);
                    setOpenId(null);
                  }}
                  disabled={disabled}
                  data-testid={`assume-${open.id}-${option.value}`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
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
  // The draft is on the map as soon as it exists; the camera fits the ground when it changes.
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
  const [approving, setApproving] = useState(false);
  const onApprove = () => {
    if (approving) return;
    setApproving(true);
    void approveDraft({ ...draft, title: draft.title.trim() || "New plan" }, goal).finally(() =>
      setApproving(false),
    );
  };
  const drafting = composer?.status === "drafting";
  return (
    <section className="mc-review" data-testid="plan-review" aria-label="Plan draft">
      <input
        className="mc-input mc-review__title"
        value={draft.title}
        aria-label="Plan title"
        onChange={(e) => {
          planningEdited();
          update({ draft: { ...draft, title: e.target.value } });
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
        <span>{draft.machineIds.join(", ") || "no machines yet"}</span>
        <span>
          {draft.startDate}
          {draft.endDate ? ` → ${draft.endDate}` : " → ongoing"}
        </span>
      </div>
      <Assumptions
        items={assumed(draft.clarifications ?? [], composer?.asked ?? {}, composer?.answers ?? {})}
        answers={composer?.answers ?? {}}
        disabled={drafting}
        onChange={(id, value) =>
          void startPlanDraft(goal, {
            zoneIds: composer?.zoneIds ?? draft.zoneIds,
            answers: { [id]: value },
          })
        }
      />
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
      {(draft.risks.length > 0 || draft.questions.length > 0) && (
        <section>
          <span className="mc-eyebrow">WORTH KNOWING</span>
          <ul className="mc-list">
            {draft.risks.map((risk) => (
              <li key={risk} className="mc-note mc-note--warn">
                {risk}
              </li>
            ))}
            {draft.questions.map((q) => (
              <li key={q} className="mc-note mc-note--agent">
                {q}
              </li>
            ))}
          </ul>
        </section>
      )}
      {(draft.assumptions ?? []).length > 0 && (
        <details className="mc-details">
          <summary className="mc-eyebrow">HOW I ESTIMATED</summary>
          <ul className="mc-list mc-list--plain">
            {(draft.assumptions ?? []).map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </details>
      )}
      <div className="mc-actions">
        <button type="button" className="mc-btn" onClick={() => showPlanOnMap(draft)}>
          Fit on map
        </button>
        <button
          type="button"
          className="mc-btn mc-btn--accent"
          onClick={onApprove}
          disabled={draft.zoneIds.length === 0 || approving || drafting}
          data-testid="plan-approve"
        >
          {approving ? "Saving…" : conflicts.length ? "Approve despite conflicts" : "Approve plan"}
        </button>
      </div>
      <p className="mc-muted mc-review__hint">
        Change anything by telling me in the bar below: “two drones”, “finish by Friday”, “finer
        detail”.
      </p>
    </section>
  );
}
