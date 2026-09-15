import { ArrowLeft } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { GlassPanel } from "@twin/ui";

import type { Plan } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

import { useMissionActions } from "./useMissionActions";

/** Plans list and plan detail window (design: Plan view). */
export function PlansPanel() {
  const project = useMission((s) => s.project);
  const view = useMission((s) => s.view);
  const planId = useMission((s) => s.planId);
  const openPlan = useMission((s) => s.openPlan);
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
            {project && !plan && (
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
                    onClick={() => {
                      useMission
                        .getState()
                        .appendLog(
                          "agent",
                          "New plans are authored in the plan editor, which is next on the roadmap. Describe the goal in the command bar and I'll draft it.",
                        );
                      useMission.getState().setStreamOpen(true);
                    }}
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
            {project && plan && <PlanDetail plan={plan} onBack={() => openPlan(null)} />}
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function PlanDetail({ plan, onBack }: { plan: Plan; onBack: () => void }) {
  const { showPlanOnMap } = useMissionActions();
  const appendLog = useMission((s) => s.appendLog);
  const setStreamOpen = useMission((s) => s.setStreamOpen);
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
        <div className="mc-agent-note">
          <span className="mc-dot mc-dot--glow mc-dot--run" aria-hidden="true" />
          <span>{plan.agentNote}</span>
        </div>
        <div className="mc-actions">
          <button
            type="button"
            className="mc-btn"
            onClick={() => {
              appendLog(
                "agent",
                `Plan editing for “${plan.title}” opens here once plans are persisted; for now tell me the change in the command bar.`,
              );
              setStreamOpen(true);
            }}
          >
            Edit plan
          </button>
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
          <button
            type="button"
            className="mc-btn mc-btn--accent"
            onClick={() => showPlanOnMap(plan)}
            data-testid="plan-show-on-map"
          >
            Show on map
          </button>
        </div>
      </div>
    </div>
  );
}
