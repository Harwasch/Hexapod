import { X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { GlassPanel } from "@twin/ui";

import { useMission } from "@/state/mission";

import { useMissionActions } from "./useMissionActions";

/** Bottom-left machine / zone card (design: selection cards). */
export function SelectionCard() {
  const project = useMission((s) => s.project);
  const selection = useMission((s) => s.selection);
  const view = useMission((s) => s.view);
  const openPlan = useMission((s) => s.openPlan);
  const setFeedsOpen = useMission((s) => s.setFeedsOpen);
  const feedsOpen = useMission((s) => s.feedsOpen);
  const appendLog = useMission((s) => s.appendLog);
  const setStreamOpen = useMission((s) => s.setStreamOpen);
  const { clearSelection, runMachineAction } = useMissionActions();

  const machine =
    selection?.kind === "machine"
      ? project?.machines.find((m) => m.id === selection.id)
      : undefined;
  const zone =
    selection?.kind === "zone" ? project?.zones.find((z) => z.id === selection.id) : undefined;
  const show = view === "map" && Boolean(machine ?? zone);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key={selection?.id}
          initial={{ opacity: 0, y: 10, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 380, damping: 32 }}
        >
          <GlassPanel
            className="mc-card"
            role="region"
            aria-label={machine ? machine.name : zone?.name}
            data-testid="selection-card"
          >
            {machine && (
              <>
                <div className="mc-card__head">
                  <div className="mc-card__title">{machine.name}</div>
                  <div className="mc-row">
                    <span className={`mc-status mc-status--${machine.status}`}>
                      <span className="mc-dot" aria-hidden="true" />
                      {machine.task}
                    </span>
                    <button
                      type="button"
                      className="mc-close"
                      onClick={clearSelection}
                      aria-label="Close"
                    >
                      <X size={13} aria-hidden="true" />
                    </button>
                  </div>
                </div>
                <div className="mc-card__meta">
                  {machine.model} · {machine.zoneLabel}
                </div>
                <div className="mc-stats">
                  <div className="mc-stat">
                    <div className="mc-stat__value">{machine.batteryPct}%</div>
                    <div className="mc-stat__label">Battery</div>
                  </div>
                  <div className="mc-stat">
                    <div className="mc-stat__value">{machine.acresDone ?? "—"}</div>
                    <div className="mc-stat__label">Acres done</div>
                  </div>
                  <div className="mc-stat">
                    <div className="mc-stat__value">{machine.shift}</div>
                    <div className="mc-stat__label">Shift</div>
                  </div>
                </div>
                <div className="mc-actions">
                  <button
                    type="button"
                    className="mc-btn"
                    onClick={() => runMachineAction(machine)}
                  >
                    {machine.primaryAction}
                  </button>
                  <button
                    type="button"
                    className={`mc-btn ${feedsOpen ? "is-on" : ""}`}
                    onClick={() => setFeedsOpen(!feedsOpen)}
                    aria-pressed={feedsOpen}
                  >
                    Camera
                  </button>
                </div>
              </>
            )}
            {zone && (
              <>
                <div className="mc-card__head">
                  <div className="mc-card__title">{zone.name}</div>
                  <div className="mc-row">
                    <span className={`mc-mono mc-tone--${zone.tone}`}>{zone.acres} ac</span>
                    <button
                      type="button"
                      className="mc-close"
                      onClick={clearSelection}
                      aria-label="Close"
                    >
                      <X size={13} aria-hidden="true" />
                    </button>
                  </div>
                </div>
                <div className="mc-card__meta">
                  {zone.task} · {zone.machines}
                </div>
                <div
                  className={`mc-progress mc-progress--${zone.tone}`}
                  role="progressbar"
                  aria-valuenow={zone.progressPct}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div className="mc-progress__bar" style={{ width: `${zone.progressPct}%` }} />
                </div>
                <div className="mc-card__note">{zone.note}</div>
                <div className="mc-actions">
                  <button
                    type="button"
                    className="mc-btn"
                    onClick={() => {
                      const plan = project?.plans.find((p) => p.zoneIds.includes(zone.id));
                      if (plan) openPlan(plan.id);
                      else useMission.getState().openComposer({ zoneIds: [zone.id] });
                    }}
                  >
                    {project?.plans.some((p) => p.zoneIds.includes(zone.id))
                      ? "Open plan"
                      : "Plan here"}
                  </button>
                  <button
                    type="button"
                    className="mc-btn"
                    onClick={() => {
                      appendLog("you", `Reassign ${zone.id}`);
                      appendLog(
                        "agent",
                        `I can move ${zone.machines} off ${zone.name} — tell me which machine should take it and I'll re-sequence the passes.`,
                      );
                      setStreamOpen(true);
                    }}
                  >
                    Reassign
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
