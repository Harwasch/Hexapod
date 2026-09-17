import { ChevronDown, ChevronUp } from "lucide-react";

import { useMission } from "@/state/mission";

/** Bottom-right agent activity: collapsed status pill that expands into the thread list (design: agent stream). */
export function AgentStream() {
  const project = useMission((s) => s.project);
  const open = useMission((s) => s.streamOpen);
  const setOpen = useMission((s) => s.setStreamOpen);
  const log = useMission((s) => s.log);
  const drafting = useMission((s) => s.composer?.status === "drafting");
  const actions = project?.agent.actions ?? [];
  const running = actions.filter((a) => a.status === "run").length + (drafting ? 1 : 0);
  // An idle agent has no card: the stream appears when something runs, when a command was
  // answered, or when the operator opens it (key `a`).
  if (!project || (running === 0 && log.length === 0 && !open)) return null;
  const headline = drafting ? "Drafting a plan…" : project.agent.headline;
  return (
    <div className={`mc-stream ${open ? "is-open" : ""}`} data-testid="agent-stream">
      <div className="glass mc-stream__panel" aria-hidden={!open} id="agent-stream-panel">
        <div className="mc-stream__head">
          <span className="mc-stream__title">
            {drafting ? "Planning with the agent" : project.agent.summary}
          </span>
          <span className="mc-eyebrow">{running} THREADS</span>
        </div>
        <ul className="mc-stream__list" aria-label="Agent actions">
          {drafting && (
            <li className="mc-stream__item is-run">
              <span className="mc-dot mc-stream__dot" aria-hidden="true" />
              <span className="mc-stream__text">Drafting the plan from your goal</span>
              <Blink />
            </li>
          )}
          {actions.map((action, i) => (
            <li
              key={action.id}
              className={`mc-stream__item is-${action.status}`}
              style={{ transitionDelay: open ? `${0.08 + i * 0.035}s` : "0s" }}
            >
              <span className="mc-dot mc-stream__dot" aria-hidden="true" />
              <span className="mc-stream__text">{action.text}</span>
              {action.status === "run" && <Blink />}
            </li>
          ))}
          {log.map((entry) => (
            <li key={entry.id} className={`mc-stream__item is-${entry.role}`}>
              <span className="mc-dot mc-stream__dot" aria-hidden="true" />
              <span className="mc-stream__text">
                <span className="mc-stream__role">{entry.role === "you" ? "You" : "Agent"}</span>{" "}
                {entry.text}
              </span>
            </li>
          ))}
        </ul>
        <div className="mc-stream__foot">
          {project.agent.footer}
          {project.simulated ? " · simulated activity" : ""}
        </div>
      </div>
      <button
        type="button"
        className="glass mc-stream__pill"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="agent-stream-panel"
        data-testid="agent-stream-toggle"
      >
        <span className="mc-spinner" aria-hidden="true" />
        <span className="mc-stream__headline">{headline}</span>
        {running > 0 && <Blink />}
        {open ? (
          <ChevronDown size={13} aria-hidden="true" />
        ) : (
          <ChevronUp size={13} aria-hidden="true" />
        )}
      </button>
    </div>
  );
}

function Blink() {
  return (
    <span className="mc-blink" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}
