import { useEffect, useRef, useState } from "react";
import { CalendarCheck, Check, Pencil } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton, GlassSelect, GlassTextarea } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { useMission } from "@/state/mission";

import { PlanField } from "./PlanField";
import { planError, useActionMission, useRefreshPlans } from "./plansApi";
import type { Action } from "./planModel";
import { dayLabel, planState, staleReasonText } from "./plansFormat";

/**
 * The one thing to do next with a plan. A draft is reviewed and approved; an approved plan is
 * scheduled as a robot mission (under this land, or under the open mission project); a plan
 * made before the land changed, or with questions still open, is revised first. Approval and
 * scheduling are separate, deliberate steps, each with a note.
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
  const project = useMission((state) => state.project);
  const mission = useActionMission(land, action.id, action.missionId);
  const refresh = useRefreshPlans(land);
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [projectId, setProjectId] = useState(`land:${land.id}`);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ticket = useRef(0);
  const state = planState(action);
  useEffect(
    () => () => {
      ticket.current += 1;
    },
    [],
  );

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
      else
        await unwrap(
          api.POST("/api/v1/land/{land_id}/actions/{action_id}/mission", {
            params: { path: { land_id: land.id, action_id: action.id } },
            body: { expectedRevision: action.revision, note, projectId },
          }),
        );
      await refresh();
      if (ticket.current === current) {
        setNote("");
        setOpen(false);
      }
    } catch (cause) {
      if (ticket.current === current) setError(planError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };

  if (state === "scheduled")
    return (
      <div className="plans-next">
        {mission.data && (
          <p role="status" className="plans-next__status">
            <CalendarCheck size={15} aria-hidden="true" /> Mission scheduled for{" "}
            {dayLabel(mission.data.startDate)}
            {mission.data.endDate ? ` through ${dayLabel(mission.data.endDate)}` : ""}. Status:{" "}
            {mission.data.status}.
          </p>
        )}
        {mission.isPending && <p className="land-muted">Loading the mission…</p>}
        {mission.isError && <p role="alert">The scheduled mission could not be loaded.</p>}
        {canEdit && (
          <div className="land-row">
            <GlassButton
              size="sm"
              variant="ghost"
              leadingIcon={<Pencil size={14} aria-hidden="true" />}
              onClick={onRevise}
            >
              Revise plan
            </GlassButton>
          </div>
        )}
      </div>
    );

  if (state === "stale" || state === "blocked") {
    const open = action.constraints?.filter((item) => !item.resolved).length ?? 0;
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
              Revise it before approving or scheduling.
            </>
          ) : (
            <>
              {open === 1 ? "One question" : `${open} questions`} to settle before this plan can be
              approved.
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
            onClick={() => setOpen(true)}
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
            void submit();
          }}
        >
          <PlanField label={approving ? "What did you check?" : "Anything the crew should know?"}>
            <GlassTextarea
              aria-label={label}
              required
              rows={2}
              maxLength={2000}
              value={note}
              onChange={(event) => setNote(event.target.value)}
            />
          </PlanField>
          {!approving && (
            <PlanField label="Schedule under">
              <GlassSelect
                aria-label="Schedule under"
                value={projectId}
                onChange={(event) => setProjectId(event.target.value)}
              >
                <option value={`land:${land.id}`}>{land.name}</option>
                {project && <option value={project.id}>{project.name}</option>}
              </GlassSelect>
            </PlanField>
          )}
          <div className="land-row">
            <GlassButton
              type="submit"
              variant="primary"
              loading={busy}
              disabled={busy || !note.trim()}
            >
              {approving ? `Approve revision ${action.revision}` : "Schedule mission"}
            </GlassButton>
            <GlassButton variant="ghost" disabled={busy} onClick={() => setOpen(false)}>
              Not yet
            </GlassButton>
          </div>
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
