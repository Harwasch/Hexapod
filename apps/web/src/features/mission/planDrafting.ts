import type { PlanAnswerValue, PlanDraft, PlanRecordStatus } from "@twin/contracts";

import { isOffline } from "@/api/client";
import { draftPlan, plansApi } from "@/api/queries";
import { describeError } from "@/lib/log";
import { planningApproved, planningDrafted, planningOpened } from "@/lib/planningMetrics";
import {
  buildDraftRequest,
  describeDraft,
  planFromDraft,
  planFromRecord,
  presentStatus,
  recordBodyFromDraft,
} from "@/missions/planDraft";
import type { Plan } from "@/missions/types";
import { useMission } from "@/state/mission";

/** Invalidated after every write so an open Plans window refetches. */
export const plansInvalidate: { current: (() => void) | null } = { current: null };

/**
 * Runs one drafting round: opens the composer (or reuses it), asks the planner, and narrates
 * the result in the agent stream. Shared by the composer's button and the command bar.
 */
export async function startPlanDraft(
  goal: string,
  options: {
    zoneIds?: string[];
    machineIds?: string[];
    refinement?: string;
    answers?: Record<string, PlanAnswerValue>;
  } = {},
): Promise<void> {
  const state = useMission.getState();
  const project = state.project;
  if (!project) {
    state.appendLog("agent", "Load a project with zones and machines first, then I can plan.");
    state.setStreamOpen(true);
    return;
  }
  const text = goal.trim();
  if (text.length < 3) {
    state.appendLog("agent", "Tell me the goal in a sentence and I'll draft the plan.");
    state.setStreamOpen(true);
    return;
  }
  const existing = state.composer;
  if (!existing) {
    state.openComposer({ goal: text, ...options });
    planningOpened();
  }
  const composer = useMission.getState().composer;
  const zoneIds = options.zoneIds ?? composer?.zoneIds ?? [];
  const machineIds = options.machineIds ?? composer?.machineIds ?? [];
  const answers = { ...(composer?.answers ?? {}), ...(options.answers ?? {}) };
  useMission.getState().updateComposer({
    goal: text,
    zoneIds,
    machineIds,
    answers,
    status: "drafting",
    error: null,
  });
  useMission.getState().appendLog("you", options.refinement ?? text);
  try {
    const request = buildDraftRequest(project, {
      goal: text,
      zoneIds,
      machineIds,
      replacePlanId: composer?.replacePlanId ?? null,
      answers,
      ...(options.refinement ? { refinement: options.refinement } : {}),
      previous: options.refinement ? (composer?.draft ?? null) : null,
    });
    const draft = await draftPlan(request);
    if (!useMission.getState().composer) return; // closed while drafting
    planningDrafted(Boolean(options.refinement));
    // The chips stay the operator's pre-selection; the draft carries its own scope. The
    // previous draft is kept so the review can say what the redraft changed.
    useMission.getState().updateComposer({
      status: "ready",
      draft,
      previousDraft: options.refinement || options.answers ? (composer?.draft ?? null) : null,
    });
    useMission.getState().appendLog("agent", describeDraft(draft));
  } catch (error) {
    const message = describeError(error);
    useMission.getState().updateComposer({ status: "error", error: message });
    useMission.getState().appendLog("agent", `I couldn't draft that: ${message}`);
  }
}

/**
 * Approves a reviewed draft: the API stores it (a new plan, or a new revision of the plan
 * being revised) and the console shows the stored record. When the API is offline the plan
 * is kept in this browser and says so.
 */
export async function approveDraft(draft: PlanDraft, goal: string): Promise<Plan | null> {
  const state = useMission.getState();
  const project = state.project;
  if (!project) return null;
  const replaceId = state.composer?.replacePlanId ?? null;
  const existing = replaceId ? project.plans.find((p) => p.id === replaceId) : undefined;
  const body = recordBodyFromDraft(draft, goal, project.id, project.siteId, project.zones);
  let plan: Plan;
  try {
    // A revision carries the plan body only: identity (project, site) never changes.
    const { projectId: _projectId, siteId: _siteId, ...revision } = body;
    const record = existing?.persisted
      ? await plansApi.revise(existing.id, { ...revision, note: `Revised: ${goal.slice(0, 120)}` })
      : await plansApi.create(body);
    plan = planFromRecord(record, project);
    plansInvalidate.current?.();
  } catch (error) {
    if (!isOffline(error)) {
      useMission.getState().appendLog("agent", `I couldn't save the plan: ${describeError(error)}`);
      useMission.getState().setStreamOpen(true);
      return null;
    }
    plan = planFromDraft(draft, project, { goal, ...(replaceId ? { id: replaceId } : {}) });
    plan.agentNote = `${plan.agentNote} Saved in this browser only: the API is offline.`;
  }
  useMission.getState().approvePlan(plan);
  planningApproved();
  useMission
    .getState()
    .appendLog(
      "agent",
      `“${plan.title}” is approved and scheduled${plan.persisted ? ` (revision ${plan.revision ?? 1})` : ""}. ${plan.facts[2]?.v ?? ""}`,
    );
  return plan;
}

/** Lifecycle changes show at once and go to the API; the stored state replaces them when it lands. */
export async function setPlanStatus(plan: Plan, status: PlanRecordStatus): Promise<void> {
  const presentation = presentStatus(status);
  // The console reflects the change at once; the stored record replaces it when it lands.
  useMission.getState().updatePlan(plan.id, presentation);
  if (plan.persisted) {
    try {
      const record = await plansApi.setStatus(plan.id, status);
      const project = useMission.getState().project;
      if (project) useMission.getState().updatePlan(plan.id, planFromRecord(record, project));
      plansInvalidate.current?.();
      return;
    } catch (error) {
      if (!isOffline(error)) {
        // Roll the optimistic change back to what the console showed before.
        useMission.getState().updatePlan(plan.id, {
          status: plan.status,
          state: plan.state,
          action: plan.action,
        });
        useMission
          .getState()
          .appendLog("agent", `I couldn't update the plan: ${describeError(error)}`);
        useMission.getState().setStreamOpen(true);
      }
    }
  }
}
