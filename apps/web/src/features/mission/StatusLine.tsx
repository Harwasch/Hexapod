import { ChevronDown, ChevronUp, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useSites as useSiteCatalog } from "@/api/queries";
import { DemoMissionProvider } from "@/missions/demo";
import { useMission } from "@/state/mission";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

import { useConnectionIssues } from "../notices/connection";
import { AgentActivityLog, Blink } from "./AgentStream";
import {
  REPLY_HOLD_MS,
  agentLine,
  fleetCounts,
  fleetLine,
  globeLine,
  sumFleet,
  type FleetCounts,
} from "./statusSummary";

const LOG_ID = "agent-activity";

/**
 * Every catalog site's fleet, for the summary at globe scale. Read through the same mission
 * provider the scene uses (`SceneBridge`), so a site counts the machines it would show when
 * visited — including before the first visit.
 */
function useFleetOverview(): { sites: number; counts: FleetCounts; simulated: boolean } {
  const catalog = useSiteCatalog();
  return useMemo(() => {
    const provider = new DemoMissionProvider();
    const projects = (catalog.data ?? [])
      .map((site) => provider.projectForSite(site.id, site.slug))
      .filter((project) => project !== null);
    return {
      sites: catalog.data?.length ?? 0,
      counts: sumFleet(projects.map(fleetCounts)),
      simulated: projects.some((project) => project.simulated),
    };
  }, [catalog.data]);
}

/**
 * One line, bottom left: the fleet in view, what the agent is doing, and the connection when
 * it is degraded. It replaced the agent's thirteen-line panel, the camera readouts and the
 * notice stack; the panel is one click away (the chevron, or `a`), the readouts are behind
 * Settings › Advanced, and a healthy connection says nothing.
 *
 * Polite live region: a new reply or a dropped connection is announced without interrupting.
 */
export function StatusLine() {
  const project = useMission((s) => s.project);
  const log = useMission((s) => s.log);
  const drafting = useMission((s) => s.composer?.status === "drafting");
  const activeSiteId = useSites((s) => s.activeSiteId);
  const open = useUi((s) => s.activityOpen);
  const setOpen = useUi((s) => s.setActivityOpen);
  const issues = useConnectionIssues();
  const overview = useFleetOverview();
  const root = useRef<HTMLDivElement>(null);

  // A reply holds the agent's line for a few seconds; this marks the moment it lets go.
  const last = log.at(-1);
  const [expired, setExpired] = useState<string | null>(null);
  useEffect(() => {
    if (!last) return;
    const timer = setTimeout(
      () => setExpired(last.id),
      Math.max(0, last.at + REPLY_HOLD_MS - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [last]);
  const agent = agentLine(project, log, drafting, Boolean(last && expired !== last.id));

  // Flows that want the operator to read the agent (`streamOpen`) get its line lit up, not a
  // panel over the map; the request is lowered once answered.
  const [flash, setFlash] = useState(0);
  useEffect(() => {
    const answer = () => {
      useMission.getState().setStreamOpen(false);
      setFlash((n) => n + 1);
    };
    if (useMission.getState().streamOpen) queueMicrotask(answer);
    return useMission.subscribe((state, prev) => {
      if (state.streamOpen && !prev.streamOpen) answer();
    });
  }, []);

  // The log is a popover: a press anywhere else closes it.
  useEffect(() => {
    if (!open) return;
    const onDown = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open, setOpen]);

  const site = activeSiteId && project?.siteId === activeSiteId ? project : null;
  const counts = site ? fleetCounts(site) : overview.counts;
  const fleetText = site ? fleetLine(counts) : globeLine(overview.sites, counts);
  const simulated = site ? site.simulated : overview.simulated && counts.machines > 0;
  const fleetTone =
    counts.attention > 0 ? "mc-dot--amber" : counts.working > 0 ? "mc-dot--teal" : "";

  return (
    <div className="status-line-wrap" ref={root}>
      {open && <AgentActivityLog id={LOG_ID} issues={issues} />}
      <div className="glass glass--strong status-line" data-testid="status-line">
        <div className="status-line__text" role="status" aria-live="polite">
          {issues.map((issue) => (
            <span
              key={issue.id}
              className="status-line__seg status-line__seg--warn"
              title={issue.detail}
              data-testid={issue.testId}
            >
              <TriangleAlert size={13} aria-hidden="true" />
              {issue.title}
            </span>
          ))}
          <span className="status-line__seg" data-testid="status-fleet">
            <span className={`mc-dot ${fleetTone}`} aria-hidden="true" />
            <span className="status-line__clip">{fleetText}</span>
            {simulated && <span className="status-line__tag">simulated</span>}
          </span>
        </div>
        <button
          type="button"
          className={`status-line__agent ${agent.idle ? "is-idle" : ""}`}
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-controls={open ? LOG_ID : undefined}
          title={agent.text}
          data-testid="agent-stream-toggle"
        >
          <span
            key={flash}
            className={`status-line__agent-text ${flash ? "is-flash" : ""}`}
            aria-live="polite"
            data-testid="status-agent"
          >
            <span className={`mc-spinner ${agent.busy ? "" : "is-static"}`} aria-hidden="true" />
            <span className="status-line__clip">
              <span className="status-line__label">Agent:</span> {agent.text}
            </span>
            {agent.busy && <Blink />}
          </span>
          {agent.more > 0 && (
            <span className="status-line__more">
              +{agent.more} {agent.more === 1 ? "task" : "tasks"}
            </span>
          )}
          <span className="sr-only">{open ? "Hide agent activity" : "Show agent activity"}</span>
          {open ? (
            <ChevronDown size={14} aria-hidden="true" />
          ) : (
            <ChevronUp size={14} aria-hidden="true" />
          )}
        </button>
      </div>
    </div>
  );
}
