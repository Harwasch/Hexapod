import { draftPlan } from "@/api/queries";
import { describeError } from "@/lib/log";
import { buildDraftRequest, describeDraft } from "@/missions/planDraft";
import { useMission } from "@/state/mission";

/**
 * Runs one drafting round: opens the composer (or reuses it), asks the planner, and narrates
 * the result in the agent stream. Shared by the composer's button and the command bar.
 */
export async function startPlanDraft(
  goal: string,
  options: { zoneIds?: string[]; machineIds?: string[]; refinement?: string } = {},
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
  if (!existing) state.openComposer({ goal: text, ...options });
  const composer = useMission.getState().composer;
  const zoneIds = options.zoneIds ?? composer?.zoneIds ?? [];
  const machineIds = options.machineIds ?? composer?.machineIds ?? [];
  useMission.getState().updateComposer({
    goal: text,
    zoneIds,
    machineIds,
    status: "drafting",
    error: null,
  });
  useMission.getState().appendLog("you", options.refinement ?? text);
  try {
    const request = buildDraftRequest(project, {
      goal: text,
      zoneIds,
      machineIds,
      ...(options.refinement ? { refinement: options.refinement } : {}),
      previous: options.refinement ? (composer?.draft ?? null) : null,
    });
    const draft = await draftPlan(request);
    if (!useMission.getState().composer) return; // closed while drafting
    // The chips stay the operator's pre-selection; the draft carries its own scope. The
    // previous draft is kept so the review can say what the redraft changed.
    useMission.getState().updateComposer({
      status: "ready",
      draft,
      previousDraft: options.refinement ? (composer?.draft ?? null) : null,
    });
    useMission.getState().appendLog("agent", describeDraft(draft));
  } catch (error) {
    const message = describeError(error);
    useMission.getState().updateComposer({ status: "error", error: message });
    useMission.getState().appendLog("agent", `I couldn't draft that: ${message}`);
  }
}
