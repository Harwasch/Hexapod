import { AnimatePresence, motion } from "motion/react";

import { GlassPanel } from "@twin/ui";

import { useMission } from "@/state/mission";
import { useUi } from "@/state/ui";

import { useMissionActions } from "./useMissionActions";

const TONE_CLASS = {
  teal: "mc-dot--teal",
  amber: "mc-dot--amber",
  blue: "mc-dot--blue",
  neutral: "mc-dot--neutral",
} as const;

/** Fleet table with stats, attention note and the treatment log (design: Fleet view + DATA WINDOW). */
export function FleetPanel() {
  const project = useMission((s) => s.project);
  const view = useMission((s) => s.view);
  const selection = useMission((s) => s.selection);
  const workLogOpen = useMission((s) => s.workLogOpen);
  const setWorkLogOpen = useMission((s) => s.setWorkLogOpen);
  const setAddDataOpen = useUi((s) => s.setAddDataOpen);
  const { selectMachine } = useMissionActions();
  const open = view === "fleet";
  const totalAcres = project?.workLog.reduce((sum, row) => sum + (row.acres ?? 0), 0) ?? 0;
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="fleet"
          className="mc-window-wrap mc-window-wrap--wide"
          initial={{ opacity: 0, y: 10, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.985 }}
          transition={{ type: "spring", stiffness: 360, damping: 32 }}
        >
          <GlassPanel
            strong
            className="mc-window"
            role="region"
            aria-label="Fleet"
            data-testid="fleet-panel"
          >
            {!project && (
              <div className="mc-window__empty">
                <div className="mc-window__title">Fleet</div>
                <p className="mc-muted">
                  No machines are assigned here yet. Load a project with mission data or add a site.
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
            {project && (
              <>
                <div className="mc-window__head">
                  <div>
                    <div className="mc-window__title">Fleet</div>
                    <div className="mc-window__sub">
                      {project.machines.length} machines assigned to this project · click a row to
                      locate it{project.simulated ? " · simulated" : ""}
                    </div>
                  </div>
                  <button
                    type="button"
                    className="mc-btn"
                    onClick={() => {
                      useMission
                        .getState()
                        .appendLog(
                          "agent",
                          "Machines are registered by the robot bridge; there is nothing to pair in this simulated fleet.",
                        );
                      useMission.getState().setStreamOpen(true);
                    }}
                  >
                    Add machine
                  </button>
                </div>
                <div className="mc-stats mc-stats--row">
                  {project.fleetStats.map((stat) => (
                    <div key={stat.label} className="mc-stat">
                      <div
                        className={`mc-stat__value mc-stat__value--lg ${stat.accent ? "mc-tone--warn" : ""}`}
                      >
                        {stat.value}
                      </div>
                      <div className="mc-stat__label">{stat.label}</div>
                    </div>
                  ))}
                </div>
                <div className="mc-window__body mc-table-wrap">
                  <div className="mc-table mc-table--fleet" role="table" aria-label="Machines">
                    <div className="mc-table__head" role="row">
                      <span role="columnheader">MACHINE</span>
                      <span role="columnheader">TASK</span>
                      <span role="columnheader">ZONE</span>
                      <span role="columnheader" className="mc-right">
                        BATT
                      </span>
                      <span role="columnheader" className="mc-right">
                        HRS
                      </span>
                    </div>
                    {project.machines.map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        role="row"
                        className={`mc-table__row ${selection?.kind === "machine" && selection.id === m.id ? "is-selected" : ""}`}
                        onClick={() => selectMachine(m.id, { fly: true })}
                        data-testid={`fleet-row-${m.id}`}
                      >
                        <span role="cell" className="mc-row">
                          <span
                            className={`mc-dot mc-dot--sq mc-dot--${m.status}`}
                            aria-hidden="true"
                          />
                          <span>
                            <span className="mc-table__primary">{m.name}</span>
                            <span className="mc-table__secondary">{m.model}</span>
                          </span>
                        </span>
                        <span role="cell" className={`mc-status-text mc-status-text--${m.status}`}>
                          {m.task}
                        </span>
                        <span role="cell">{m.zoneLabel}</span>
                        <span role="cell" className="mc-mono mc-right">
                          {m.batteryPct}%
                        </span>
                        <span role="cell" className="mc-mono mc-right mc-muted">
                          {m.shift}
                        </span>
                      </button>
                    ))}
                  </div>
                  {workLogOpen && (
                    <div
                      className="mc-table mc-table--log"
                      role="table"
                      aria-label="Treatment log"
                      data-testid="work-log"
                    >
                      <div className="mc-table__title">Treatment log</div>
                      <div className="mc-table__head" role="row">
                        <span role="columnheader">DATE</span>
                        <span role="columnheader">MACHINE</span>
                        <span role="columnheader">TASK</span>
                        <span role="columnheader" className="mc-right">
                          ACRES
                        </span>
                        <span role="columnheader" className="mc-right">
                          HRS
                        </span>
                        <span role="columnheader" className="mc-right">
                          L/AC
                        </span>
                        <span role="columnheader">ZONE</span>
                      </div>
                      {project.workLog.map((row, i) => (
                        <div key={i} className="mc-table__row mc-table__row--static" role="row">
                          <span role="cell" className="mc-mono mc-muted">
                            {row.date}
                          </span>
                          <span role="cell" className="mc-row">
                            <span className={`mc-dot ${TONE_CLASS[row.tone]}`} aria-hidden="true" />
                            {row.machine}
                          </span>
                          <span role="cell">{row.task}</span>
                          <span role="cell" className="mc-mono mc-right">
                            {row.acres ?? "—"}
                          </span>
                          <span role="cell" className="mc-mono mc-right mc-muted">
                            {row.hours}
                          </span>
                          <span role="cell" className="mc-mono mc-right mc-muted">
                            {row.rate}
                          </span>
                          <span role="cell">{row.zone}</span>
                        </div>
                      ))}
                      <div className="mc-table__row mc-table__row--total" role="row">
                        <span role="cell" className="mc-eyebrow">
                          TOTAL
                        </span>
                        <span role="cell" />
                        <span role="cell" />
                        <span role="cell" className="mc-mono mc-right mc-tone--teal">
                          {totalAcres.toLocaleString()}
                        </span>
                        <span role="cell" />
                        <span role="cell" />
                        <span role="cell" />
                      </div>
                    </div>
                  )}
                </div>
                <div className="mc-window__foot">
                  <span className="mc-dot mc-dot--glow mc-dot--warn" aria-hidden="true" />
                  <span className="mc-window__foot-text">{project.fleetNote}</span>
                  <button
                    type="button"
                    className="mc-btn mc-btn--sm"
                    onClick={() => setWorkLogOpen(!workLogOpen)}
                    aria-pressed={workLogOpen}
                    data-testid="toggle-work-log"
                  >
                    {workLogOpen ? "Hide log" : "Service log"}
                  </button>
                </div>
              </>
            )}
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
