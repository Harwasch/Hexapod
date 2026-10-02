import { TriangleAlert } from "lucide-react";

import { useMission } from "@/state/mission";

import type { ConnectionIssue } from "../notices/connection";

/**
 * The agent's activity log (design: agent stream): what it is running, what is queued, what it
 * has done, and the conversation with the operator. It opens above the status line, which
 * carries the one line of it that matters now; degraded connections lead it with their
 * sentence, since the status line only has room for their name.
 */
export function AgentActivityLog({ id, issues }: { id: string; issues: ConnectionIssue[] }) {
  const project = useMission((s) => s.project);
  const log = useMission((s) => s.log);
  const drafting = useMission((s) => s.composer?.status === "drafting");
  const actions = project?.agent.actions ?? [];
  const running = actions.filter((a) => a.status === "run").length + (drafting ? 1 : 0);
  const empty = actions.length === 0 && log.length === 0 && !drafting;
  return (
    <div
      className="glass glass--strong status-log"
      id={id}
      role="region"
      aria-label="Agent activity"
      data-hud-popover=""
      data-testid="agent-stream"
    >
      {issues.length > 0 && (
        <ul className="status-log__issues" aria-label="Connection">
          {issues.map((issue) => (
            <li key={issue.id} className="status-log__issue">
              <TriangleAlert size={14} aria-hidden="true" />
              <span>
                <strong>{issue.title}.</strong> {issue.detail}
              </span>
            </li>
          ))}
        </ul>
      )}
      <div className="mc-stream__head">
        <span className="mc-stream__title">
          {drafting ? "Planning with the agent" : (project?.agent.summary ?? "Agent")}
        </span>
        {running > 0 && <span className="mc-eyebrow">{running} running</span>}
      </div>
      {empty ? (
        <p className="status-log__empty">
          Nothing yet. Ask from the command box: a place, a layer, or the work to plan.
        </p>
      ) : (
        <ul className="mc-stream__list" aria-label="Agent actions">
          {drafting && (
            <li className="mc-stream__item is-run">
              <span className="mc-dot mc-stream__dot" aria-hidden="true" />
              <span className="mc-stream__text">Drafting the plan from your goal</span>
              <Blink />
            </li>
          )}
          {actions.map((action) => (
            <li key={action.id} className={`mc-stream__item is-${action.status}`}>
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
      )}
      {project && (
        <div className="mc-stream__foot">
          {project.agent.footer}
          {project.simulated ? " · simulated activity" : ""}
        </div>
      )}
    </div>
  );
}

export function Blink() {
  return (
    <span className="mc-blink" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}
