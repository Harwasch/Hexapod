import { useEffect, useId, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, CalendarCheck, Check, Pencil } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton, GlassSelect, GlassTextarea } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useSites } from "@/api/queries";
import { missionProjectName, openLandMission } from "@/features/mission/landPlanning";
import { DEMO_PROJECT } from "@/missions/demo";
import { groundModeFor, landIdOfProject, landProjectId } from "@/missions/landGround";
import { useMission } from "@/state/mission";

import { approvalBlockers } from "../../../actionDraft";
import { PlanField } from "./PlanField";
import { planError, useActionMission, useRefreshPlans } from "./plansApi";
import type { Action } from "./planModel";
import { dayLabel, planState, staleReasonText } from "./plansFormat";

/**
 * The one thing to do next with a plan. A draft is reviewed and approved; an approved plan is
 * scheduled as a robot mission (under this land's own plans, or under the open mission project
 * when its fleet works this land), and a scheduled one opens in Plan; a plan made before the
 * land changed, or with questions still open, is revised first. Approval and scheduling are
 * separate, deliberate steps, each with a note, and a disabled button always says why.
 */
export function PlanNextStep({
  land,
  action,
  canEdit,
  onRevise,
}: {
  land: LandArea;
  action: Action;
  canEdit: boolean;
  onRevise: () => void;
}) {
  const cache = useQueryClient();
  const project = useMission((state) => state.project);
  const sites = useSites().data;
  const mission = useActionMission(land, action.id, action.missionId);
  const refresh = useRefreshPlans(land);
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  // The note was visited and left: its reason turns from a hint into a warning.
  const [noteVisited, setNoteVisited] = useState(false);
  const ownProject = landProjectId(land.id);
  // A fleet's project that works this land keeps its missions together; otherwise the land's
  // own plans, the project "Plan work on this land" uses.
  const [projectId, setProjectId] = useState(() =>
    project && groundModeFor(project, land) === "layer" ? project.id : ownProject,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ticket = useRef(0);
  const reasonId = useId();
  const state = planState(action);
  useEffect(
    () => () => {
      ticket.current += 1;
    },
    [],
  );

  // The project in view can file this land's mission unless it is another land's own project
  // (a mission is filed under one land; the server refuses that).
  const otherLandProject = project !== null && (landIdOfProject(project.id) ?? land.id) !== land.id;
  // What the choice says now: a project since left (or never offered) falls back to this
  // land's own plans rather than filing under it.
  const scheduleUnder = projectId === project?.id && !otherLandProject ? projectId : ownProject;
  // A project whose fleet is simulated says so wherever it is named.
  const fleet = (id: string) =>
    (id === project?.id ? project.simulated : id === DEMO_PROJECT.id) ? " (simulated fleet)" : "";
  /** A project's name in the choice, never its id. */
  const projectLabel = (id: string) =>
    id === ownProject
      ? `${land.name} · this land's plans`
      : id === project?.id
        ? `${project.name} · current mission project${fleet(id)}`
        : (missionProjectName(id, sites) ?? "another mission project");
  /** Where a mission is filed, in a sentence. */
  const filedUnder = (id: string) => {
    if (id === ownProject) return "this land's own plans";
    const name = missionProjectName(id, sites);
    return name ? `the mission project ${name}${fleet(id)}` : "another mission project";
  };

  const submit = async () => {
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      if (action.status === "draft")
        await unwrap(
          api.POST("/api/v1/land/{land_id}/actions/{action_id}/approve", {
            params: { path: { land_id: land.id, action_id: action.id } },
            body: { expectedRevision: action.revision, note },
          }),
        );
      else {
        await unwrap(
          api.POST("/api/v1/land/{land_id}/actions/{action_id}/mission", {
            params: { path: { land_id: land.id, action_id: action.id } },
            body: { expectedRevision: action.revision, note, projectId: scheduleUnder },
          }),
        );
        // Plan and Fleet list scheduled missions: they fetch the new one now.
        await cache.invalidateQueries({ queryKey: ["workspace-plans"] });
      }
      await refresh();
      if (ticket.current === current) {
        setNote("");
        setNoteVisited(false);
        setOpen(false);
      }
    } catch (cause) {
      if (ticket.current === current) setError(planError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };

  if (state === "scheduled") {
    const record = mission.data;
    return (
      <div className="plans-next">
        {record && (
          <p role="status" className="plans-next__status" data-testid="land-mission-status">
            <CalendarCheck size={15} aria-hidden="true" />
            <span>
              Mission scheduled for {dayLabel(record.startDate)}
              {record.endDate && record.endDate !== record.startDate
                ? ` through ${dayLabel(record.endDate)}`
                : ""}{" "}
              under {filedUnder(record.projectId)}. Status: {record.status}.
            </span>
          </p>
        )}
        {mission.isPending && <p className="land-muted">Loading the mission…</p>}
        {mission.isError && <p role="alert">The scheduled mission could not be loaded.</p>}
        {(Boolean(record) || canEdit) && (
          <div className="land-row">
            {record && (
              <GlassButton
                size="sm"
                variant="primary"
                leadingIcon={<ArrowUpRight size={14} aria-hidden="true" />}
                data-testid="land-mission-open-plan"
                onClick={() => void openLandMission(land, record, cache)}
              >
                Open in Plan
              </GlassButton>
            )}
            {canEdit && (
              <GlassButton
                size="sm"
                variant="ghost"
                leadingIcon={<Pencil size={14} aria-hidden="true" />}
                onClick={onRevise}
              >
                Revise plan
              </GlassButton>
            )}
          </div>
        )}
      </div>
    );
  }

  if (state === "stale" || state === "blocked") {
    const questions = action.constraints?.filter((item) => !item.resolved).length ?? 0;
    const counted = questions === 1 ? "One question" : `${questions} questions`;
    return (
      <div className="plans-next">
        <div className="land-note">
          {state === "stale" ? (
            <>
              <strong>The land changed since this plan was made.</strong>
              <ul className="plans-reasons">
                {staleReasonText(action.staleReasons).map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
              {questions > 0 && `${counted} also to settle. `}
              Revise it before approving or scheduling.
            </>
          ) : (
            <>
              {counted} to settle before this plan can be approved. Revise it, tick Settled and say
              how.
            </>
          )}
        </div>
        {canEdit && (
          <div className="land-row">
            <GlassButton
              variant="primary"
              leadingIcon={<Pencil size={15} aria-hidden="true" />}
              onClick={onRevise}
            >
              Revise plan
            </GlassButton>
          </div>
        )}
      </div>
    );
  }

  if (!canEdit)
    return (
      <p className="land-muted plans-next">
        {state === "approved" ? "Approved, waiting to be scheduled." : "Waiting for review."}
      </p>
    );

  const approving = action.status === "draft";
  const label = approving ? "Review note" : "Scheduling note";
  // Why the button is disabled, in the order to fix them. A stale plan or open questions are
  // answered above by revising; should a reason of that kind reach this form (an approved
  // plan whose questions reopened), the revision is offered beside it.
  const blockers = approvalBlockers(action, note, approving ? "approve" : "schedule");
  const reviseBlocker = blockers.find((blocker) => blocker.revise);
  const noteLeftEmpty = noteVisited && !note.trim();
  const blockerId = (kind: string) => `${reasonId}-${kind}`;
  return (
    <div className="plans-next">
      {!open ? (
        <div className="land-row">
          <GlassButton
            variant="primary"
            leadingIcon={
              approving ? (
                <Check size={15} aria-hidden="true" />
              ) : (
                <CalendarCheck size={15} aria-hidden="true" />
              )
            }
            onClick={() => {
              setNoteVisited(false);
              setOpen(true);
            }}
          >
            {approving ? "Review and approve" : "Schedule as a mission"}
          </GlassButton>
          <GlassButton
            variant="ghost"
            leadingIcon={<Pencil size={14} aria-hidden="true" />}
            onClick={onRevise}
          >
            Revise
          </GlassButton>
        </div>
      ) : (
        <form
          className="plans-next__form"
          aria-label={approving ? "Approve this plan" : "Schedule this plan"}
          onSubmit={(event) => {
            event.preventDefault();
            if (!blockers.length) void submit();
          }}
        >
          <PlanField label={approving ? "What did you check?" : "Anything the crew should know?"}>
            <GlassTextarea
              aria-label={label}
              required
              rows={2}
              maxLength={2000}
              invalid={noteLeftEmpty}
              aria-describedby={noteLeftEmpty ? blockerId("note") : undefined}
              value={note}
              onChange={(event) => setNote(event.target.value)}
              onBlur={() => setNoteVisited(true)}
            />
          </PlanField>
          {!approving && (
            <PlanField label="Schedule under">
              <GlassSelect
                aria-label="Schedule under"
                value={scheduleUnder}
                onChange={(event) => setProjectId(event.target.value)}
              >
                <option value={ownProject}>{projectLabel(ownProject)}</option>
                {project && project.id !== ownProject && !otherLandProject && (
                  <option value={project.id}>{projectLabel(project.id)}</option>
                )}
              </GlassSelect>
            </PlanField>
          )}
          <div className="land-row">
            <GlassButton
              type="submit"
              variant="primary"
              loading={busy}
              disabled={busy || blockers.length > 0}
              // The reasons' own words, without the Revise button beside one of them.
              aria-describedby={
                blockers.map((blocker) => blockerId(blocker.kind)).join(" ") || undefined
              }
            >
              {approving ? `Approve revision ${action.revision}` : "Schedule mission"}
            </GlassButton>
            <GlassButton variant="ghost" disabled={busy} onClick={() => setOpen(false)}>
              Not yet
            </GlassButton>
          </div>
          {blockers.length > 0 && (
            <ul
              className="plans-blockers"
              aria-label={approving ? "Before approving" : "Before scheduling"}
              data-testid="plans-approval-blockers"
            >
              {blockers.map((blocker) => (
                <li
                  key={blocker.kind}
                  // Quiet until the note was visited and left empty: a fresh review does not
                  // open on a warning, yet the disabled button is never without a reason.
                  className={blocker.kind === "note" && !noteLeftEmpty ? "is-hint" : undefined}
                >
                  <span id={blockerId(blocker.kind)}>{blocker.message}</span>
                  {blocker === reviseBlocker && (
                    <>
                      {" "}
                      <button
                        type="button"
                        className="land-link"
                        disabled={busy}
                        onClick={onRevise}
                      >
                        Revise plan
                      </button>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
          <p className="land-muted">
            {approving
              ? "Approval records your review of this revision. Scheduling comes next."
              : "Creates a private scheduled mission. Sending machines out stays a separate step."}
          </p>
        </form>
      )}
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
