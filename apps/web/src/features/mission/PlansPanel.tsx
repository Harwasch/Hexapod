import { ArrowLeft } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect } from "react";

import { GlassPanel } from "@twin/ui";

import type { Plan } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

import { useScene } from "@/cesium/SceneContext";

import { PlanComposer } from "./PlanComposer";
import { PlanSchedule } from "./PlanSchedule";
import { useMissionActions } from "./useMissionActions";

/** Plans list and plan detail window (design: Plan view). */
export function PlansPanel() {
  const project = useMission((s) => s.project);
  const view = useMission((s) => s.view);
  const planId = useMission((s) => s.planId);
  const openPlan = useMission((s) => s.openPlan);
  const composer = useMission((s) => s.composer);
  const openComposer = useMission((s) => s.openComposer);
  const setAddDataOpen = useUi((s) => s.setAddDataOpen);
  const plan = project?.plans.find((p) => p.id === planId) ?? null;
  const open = view === "plan";
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="plans"
          className="mc-window-wrap"
          initial={{ opacity: 0, y: 10, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.985 }}
          transition={{ type: "spring", stiffness: 360, damping: 32 }}
        >
          <GlassPanel
            strong
            className="mc-window"
            role="region"
            aria-label="Plans"
            data-testid="plans-panel"
          >
            {!project && (
              <div className="mc-window__empty">
                <div className="mc-window__title">Plans</div>
                <p className="mc-muted">
                  No project is loaded. Fly to a site with mission data, or add one.
                </p>
                <button
                  type="button"
                  className="mc-btn mc-btn--accent"
                  onClick={() => setAddDataOpen(true)}
                >
                  Add a site
                </button>
              </div>
            )}
            {project && composer && <PlanComposer project={project} />}
            {project && !composer && !plan && (
              <>
                <div className="mc-window__head">
                  <div>
                    <div className="mc-window__title">Plans</div>
                    <div className="mc-window__sub">
                      {project.name} · {project.plans.length} active ·{" "}
                      {project.plans.filter((p) => p.ongoing).length} recurring
                      {project.simulated ? " · simulated" : ""}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="mc-btn mc-btn--accent"
                    onClick={() => openComposer()}
                    data-testid="plan-new"
                  >
                    + New plan
                  </button>
                </div>
                <div className="mc-window__body mc-plan-list">
                  {project.plans.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      className="mc-plan"
                      onClick={() => openPlan(p.id)}
                      data-testid={`plan-${p.id}`}
                    >
                      <div className="mc-row">
                        <span
                          className={`mc-dot mc-dot--glow mc-dot--${p.status}`}
                          aria-hidden="true"
                        />
                        <span className="mc-plan__title">{p.title}</span>
                        <span className={`mc-plan__state mc-tone--${p.status}`}>{p.state}</span>
                      </div>
                      <div className="mc-row mc-plan__progress">
                        <div
                          className={`mc-progress mc-progress--${p.status}`}
                          role="progressbar"
                          aria-valuenow={p.progressPct}
                          aria-valuemin={0}
                          aria-valuemax={100}
                        >
                          <div
                            className="mc-progress__bar"
                            style={{ width: `${p.progressPct}%` }}
                          />
                        </div>
                        <span className="mc-mono mc-plan__label">{p.progressLabel}</span>
                      </div>
                      <div className="mc-row mc-plan__meta">
                        <span>{p.meta}</span>
                        <span className={p.ongoing ? "mc-muted" : ""}>
                          {p.ongoing ? "Ongoing" : `Ends ${p.end}`}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </>
            )}
            {project && !composer && plan && (
              <PlanDetail plan={plan} onBack={() => openPlan(null)} />
            )}
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function PlanDetail({ plan, onBack }: { plan: Plan; onBack: () => void }) {
  const { showPlanOnMap, clearPlanOverlay } = useMissionActions();
  const appendLog = useMission((s) => s.appendLog);
  const setStreamOpen = useMission((s) => s.setStreamOpen);
  const openComposer = useMission((s) => s.openComposer);
  const updatePlan = useMission((s) => s.updatePlan);
  const scene = useScene();
  const colorFor = (id: string) => scene?.mission.machineColor(id).toCssColorString() ?? "#7fd8c0";
  const owned = plan.goal !== undefined;
  // An open plan is drawn on the map for as long as its detail is open.
  useEffect(() => {
    scene?.mission.showPlan(
      plan.zoneIds.length
        ? {
            zones: plan.zoneIds.map((zoneId) => ({
              zoneId,
              machineIds:
                plan.steps?.find((s) => s.zoneId === zoneId)?.machineIds ?? plan.machineIds ?? [],
            })),
          }
        : null,
    );
    return () => clearPlanOverlay();
  }, [scene, plan.id, plan.zoneIds, plan.steps, plan.machineIds, clearPlanOverlay]);
  const lifecycle = () => {
    if (plan.status === "run") {
      updatePlan(plan.id, { status: "idle", state: "Paused", action: "Resume" });
      appendLog("agent", `“${plan.title}” paused. The fleet finishes its current pass and holds.`);
    } else {
      updatePlan(plan.id, { status: "run", state: "Dispatched", action: "Pause plan" });
      appendLog(
        "agent",
        `“${plan.title}” dispatched to the fleet${useMission.getState().project?.simulated ? " (simulated: the robot bridge will pick this up here)" : ""}.`,
      );
    }
    setStreamOpen(true);
  };
  return (
    <div className="mc-plan-detail" data-testid="plan-detail">
      <div className="mc-window__head mc-window__head--column">
        <button type="button" className="mc-link" onClick={onBack}>
          <ArrowLeft size={12} aria-hidden="true" /> All plans
        </button>
        <div className="mc-row mc-row--between">
          <div>
            <div className="mc-window__title">{plan.title}</div>
            <div className="mc-window__sub">{plan.meta}</div>
          </div>
          <span className="mc-pill-state">
            <span className={`mc-dot mc-dot--glow mc-dot--${plan.status}`} aria-hidden="true" />
            {plan.state}
          </span>
        </div>
      </div>
      <div className="mc-window__body mc-plan-detail__body">
        <p className="mc-objective">{plan.objective}</p>
        <div className="mc-stats mc-stats--tight">
          {plan.stats.map((stat) => (
            <div key={stat.label} className="mc-stat">
              <div className={`mc-stat__value ${stat.accent ? `mc-tone--${plan.status}` : ""}`}>
                {stat.value}
              </div>
              <div className="mc-stat__label">{stat.label}</div>
            </div>
          ))}
        </div>
        <section>
          <div className="mc-row mc-row--between">
            <span className="mc-eyebrow">PROGRESS</span>
            <span className={`mc-mono mc-tone--${plan.status}`}>{plan.progressLabel}</span>
          </div>
          <div
            className={`mc-progress mc-progress--${plan.status}`}
            role="progressbar"
            aria-valuenow={plan.progressPct}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="mc-progress__bar" style={{ width: `${plan.progressPct}%` }} />
          </div>
        </section>
        <section>
          <div className="mc-row mc-row--between">
            <span className="mc-eyebrow">TIMELINE</span>
            <span className="mc-muted">{plan.span}</span>
          </div>
          {plan.ongoing ? (
            <div className="mc-note">
              <span className={`mc-dot mc-dot--${plan.status}`} aria-hidden="true" />
              Ongoing — repeats on cadence, no end date
            </div>
          ) : (
            <div className="mc-timeline">
              <div className={`mc-progress mc-progress--${plan.status}`}>
                <div
                  className="mc-progress__bar"
                  style={{ width: `${plan.progressPct}%`, opacity: 0.85 }}
                />
                <div className="mc-timeline__today" style={{ left: `${plan.todayPct}%` }} />
              </div>
              <div className="mc-timeline__ticks mc-mono">
                <span>{plan.start}</span>
                <span>today</span>
                <span>{plan.end}</span>
              </div>
            </div>
          )}
        </section>
        <section>
          <span className="mc-eyebrow">SCHEDULE &amp; SCOPE</span>
          <dl className="mc-facts">
            {plan.facts.map((fact) => (
              <div key={fact.k} className="mc-facts__row">
                <dt>{fact.k}</dt>
                <dd>{fact.v}</dd>
              </div>
            ))}
          </dl>
        </section>
        {plan.steps && plan.steps.length > 0 && (
          <section>
            <span className="mc-eyebrow">SCHEDULE</span>
            <PlanSchedule
              steps={plan.steps}
              startDate={
                plan.steps.length ? (plan.facts.find((f) => f.k === "Starts")?.v ?? "") : ""
              }
              colorFor={colorFor}
            />
          </section>
        )}
        {plan.steps && plan.steps.length > 0 && (
          <section>
            <span className="mc-eyebrow">STEPS</span>
            <ol className="mc-steps">
              {plan.steps.map((step, i) => (
                <li key={`${i}-${step.title}`} className="mc-step">
                  <div className="mc-step__title">{step.title}</div>
                  <div className="mc-step__meta mc-mono">
                    {[step.when, step.zoneId, step.machineIds.join(" ")]
                      .filter(Boolean)
                      .join(" · ")}
                  </div>
                </li>
              ))}
            </ol>
          </section>
        )}
        {plan.assumptions && plan.assumptions.length > 0 && (
          <section>
            <span className="mc-eyebrow">ASSUMPTIONS</span>
            <ul className="mc-list mc-list--plain">
              {plan.assumptions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </section>
        )}
        <div className="mc-agent-note">
          <span className="mc-dot mc-dot--glow mc-dot--run" aria-hidden="true" />
          <span>{plan.agentNote}</span>
        </div>
        <div className="mc-actions">
          <button
            type="button"
            className="mc-btn"
            onClick={() => {
              if (plan.goal !== undefined) {
                openComposer({
                  goal: plan.goal,
                  zoneIds: plan.zoneIds,
                  machineIds: plan.machineIds ?? [],
                  replacePlanId: plan.id,
                });
                return;
              }
              // Provider plans are owned by the fleet backend; a revision starts from their objective.
              openComposer({ goal: plan.objective, zoneIds: plan.zoneIds, replacePlanId: plan.id });
              appendLog(
                "agent",
                `Revising “${plan.title}”: adjust the goal and I'll draft the new version for your approval.`,
              );
              setStreamOpen(true);
            }}
          >
            Revise with agent
          </button>
          {owned ? (
            <button
              type="button"
              className={`mc-btn ${plan.status === "run" ? "" : "mc-btn--accent"}`}
              onClick={lifecycle}
              data-testid="plan-lifecycle"
            >
              {plan.status === "run"
                ? "Pause plan"
                : plan.state === "Paused"
                  ? "Resume"
                  : "Dispatch to fleet"}
            </button>
          ) : (
            <button
              type="button"
              className="mc-btn"
              onClick={() => {
                appendLog("you", `${plan.action}: ${plan.title}`);
                appendLog("agent", `${plan.action} acknowledged for “${plan.title}” (simulated).`);
                setStreamOpen(true);
              }}
            >
              {plan.action}
            </button>
          )}
          <button
            type="button"
            className={`mc-btn ${owned ? "" : "mc-btn--accent"}`}
            onClick={() => showPlanOnMap(plan)}
            data-testid="plan-show-on-map"
          >
            Fit on map
          </button>
        </div>
      </div>
    </div>
  );
}
