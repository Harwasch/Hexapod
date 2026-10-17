/**
 * The checklist that stops Save and Approve/Schedule in LandActions
 * (features/land/actionDraft.ts), mirroring apps/api/app/schemas/land_actions.py and the
 * review checks in apps/api/app/services/land_actions.py.
 */

import { describe, expect, it } from "vitest";

import {
  actionDraftProblems,
  approvalBlockers,
  type ActionDraft,
} from "@/features/land/actionDraft";

type Step = ActionDraft["steps"][number];
const step = (change: Partial<Step> = {}): Step => ({
  id: "s1",
  title: "Clear brush",
  detail: "",
  startDay: 0,
  days: 3,
  successMeasure: "No brush taller than 30 cm",
  ...change,
});
// What LandActions' fresh() opens: today's date, USD and one empty step.
const fresh = (): ActionDraft => ({
  requestKey: "00000000-0000-4000-8000-000000000000",
  title: "",
  objective: "",
  boundaryRevision: 1,
  startDate: "2026-10-17",
  currency: "USD",
  steps: [step({ title: "", successMeasure: "", days: 1 })],
  constraints: [],
  evidenceIds: [],
  features: [],
  exclusions: [],
  assumptions: [],
});
const complete = (change: Partial<ActionDraft> = {}): ActionDraft => ({
  ...fresh(),
  title: "Fence line brush",
  objective: "Keep the fence line clear",
  steps: [step()],
  ...change,
});
const ids = (draft: ActionDraft) => actionDraftProblems(draft).map((problem) => problem.id);
const messages = (draft: ActionDraft) =>
  actionDraftProblems(draft).map((problem) => problem.message);

describe("actionDraftProblems", () => {
  it("names the four empty required fields of a fresh draft, in form order", () => {
    expect(actionDraftProblems(fresh())).toEqual([
      { id: "action-title", message: "Add an action title." },
      { id: "action-objective", message: "Describe the outcome to achieve." },
      { id: "action-step-0-title", message: "Step 1: add a step name." },
      { id: "action-step-0-success", message: "Step 1: add a success measure." },
    ]);
  });

  it("finds nothing in a draft the server accepts", () => {
    expect(actionDraftProblems(complete())).toEqual([]);
    // The server strips whitespace before it checks a length.
    expect(ids(complete({ title: `  ${"a".repeat(200)}  ` }))).toEqual([]);
  });

  it("treats whitespace as empty, as the server does", () => {
    expect(ids(complete({ title: "   ", objective: "\n" }))).toEqual([
      "action-title",
      "action-objective",
    ]);
  });

  it("asks for a start date and a three-letter currency", () => {
    expect(actionDraftProblems(complete({ startDate: "", currency: "us" }))).toEqual([
      { id: "action-start", message: "Choose a start date." },
      { id: "action-currency", message: "Use a three-letter currency code, like USD." },
    ]);
    expect(messages(complete({ startDate: "9999-12-01", steps: [step({ days: 60 })] }))).toEqual([
      "Choose an earlier start date. This schedule would run past the year 9999.",
    ]);
  });

  it("asks a costed step for its basis, including a zero estimate", () => {
    expect(actionDraftProblems(complete({ steps: [step({ estimatedCost: 450 })] }))).toEqual([
      {
        id: "action-step-0-cost-basis",
        message: "Step 1: explain the basis for its 450 USD estimate.",
      },
    ]);
    expect(ids(complete({ steps: [step({ estimatedCost: 0, costBasis: " " })] }))).toEqual([
      "action-step-0-cost-basis",
    ]);
    expect(
      ids(complete({ steps: [step({ estimatedCost: 450, costBasis: "Contractor quote" })] })),
    ).toEqual([]);
    expect(messages(complete({ steps: [step({ estimatedCost: -1, costBasis: "x" })] }))).toEqual([
      "Step 1: enter an estimated cost of 0 or more, or leave it blank.",
    ]);
  });

  it("names the prerequisite and the earliest day a dependent step can start", () => {
    const after = (startDay: number, title = "Clear brush") =>
      complete({
        steps: [step({ title }), step({ id: "s2", title: "Reseed", startDay, dependsOn: ["s1"] })],
      });
    expect(actionDraftProblems(after(2))).toEqual([
      {
        id: "action-step-1-start",
        message:
          "Step 2 starts on day 3, but “Clear brush” finishes on day 3. Start it on day 4 or later.",
      },
    ]);
    expect(ids(after(3))).toEqual([]);
    expect(messages(after(0, "  ")).at(-1)).toBe(
      "Step 2 starts on day 1, but “its prerequisite” finishes on day 3. Start it on day 4 or later.",
    );
  });

  it("asks for whole, in-range step timing", () => {
    expect(messages(complete({ steps: [step({ startDay: NaN, days: 0 })] }))).toEqual([
      "Step 1: enter a start day (1 or later).",
      "Step 1: enter a duration of at least 1 day.",
    ]);
    expect(messages(complete({ steps: [step({ startDay: 1.5, days: 2.5 })] }))).toEqual([
      "Step 1: enter the start day as a whole number.",
      "Step 1: enter the duration in whole days.",
    ]);
    expect(messages(complete({ steps: [step({ startDay: 36501, days: 3651 })] }))).toEqual([
      "Step 1: start on day 36,501 or earlier.",
      "Step 1: keep the duration to 3,650 days or fewer.",
    ]);
  });

  it("flags constraints with no text, and resolved ones with no resolution", () => {
    const draft = complete({
      constraints: [
        { text: "", resolved: false },
        { text: "Confirm grazing lease", resolved: true, resolution: " " },
        { text: "Check nesting season", resolved: true, resolution: "Surveyed in March" },
      ],
    });
    expect(actionDraftProblems(draft)).toEqual([
      { id: "action-constraint-0-text", message: "Constraint 1: describe it or remove it." },
      {
        id: "action-constraint-1-resolution",
        message: "Constraint 2 is marked resolved: record how.",
      },
    ]);
  });

  it("holds the text limits a scenario-derived draft can exceed", () => {
    // A scenario's assumptions (up to 5,000 characters) arrive as one assumption, and its
    // treatment objectives (up to 2,000) become step descriptions (up to 1,000).
    const draft = complete({
      assumptions: ["a".repeat(2001)],
      steps: [step({ detail: "d".repeat(1001) })],
    });
    expect(ids(draft)).toEqual(["action-step-0-detail", "action-assumptions"]);
    expect(ids(complete({ assumptions: ["a".repeat(2000), "", "  "] }))).toEqual([]);
  });

  it("counts only the resource lines actionPayload sends", () => {
    expect(ids(complete({ steps: [step({ resources: ["Mower", "", " "] })] }))).toEqual([]);
    expect(messages(complete({ steps: [step({ resources: ["r".repeat(201)] })] }))).toEqual([
      "Step 1: keep each resource to 200 characters.",
    ]);
  });
});

describe("approvalBlockers", () => {
  const action = {
    staleReasons: ["The land boundary changed after this revision."],
    constraints: [
      { text: "Confirm access", resolved: false },
      { text: "Check permits", resolved: false },
      { text: "Done", resolved: true, resolution: "Called the county" },
    ],
  };

  it("lists stale, unresolved and note blockers, with the structural ones fixed by revising", () => {
    expect(approvalBlockers(action, " ", "approve")).toEqual([
      {
        kind: "stale",
        message:
          "Supporting information changed since this revision. Revise the action to review it against current data.",
        revise: true,
      },
      {
        kind: "constraints",
        message: "Resolve 2 recorded constraints: revise the action, tick Resolved and record how.",
        revise: true,
      },
      { kind: "note", message: "Add a review note. It is stored with the approval." },
    ]);
  });

  it("words the note for scheduling and counts one constraint in the singular", () => {
    const one = { staleReasons: [], constraints: [{ text: "Confirm access", resolved: false }] };
    expect(approvalBlockers(one, "", "schedule").map((blocker) => blocker.message)).toEqual([
      "Resolve 1 recorded constraint: revise the action, tick Resolved and record how.",
      "Add a scheduling note. It is stored with the scheduled mission.",
    ]);
  });

  it("clears once the revision is current, every constraint is resolved and a note is written", () => {
    expect(approvalBlockers({ staleReasons: [] }, "Reviewed on site", "approve")).toEqual([]);
  });
});
