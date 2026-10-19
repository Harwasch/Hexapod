import { useMemo, useState } from "react";

import { actionDraftProblems, type DraftProblem } from "../../../actionDraft";
import type { ActionDraft } from "./planModel";

/** The save checklist's id: Save points at it while it shows. */
export const PROBLEMS_ID = "plans-draft-problems";

/** A marked field's reason, under it: its id is the field's id plus this. */
export const problemId = (id: string) => `${id}-problem`;

/** Opens every folded section around a field, then focuses it. */
export function focusField(id: string): void {
  const target = document.getElementById(id);
  if (!target) return;
  let fold = target.closest("details");
  while (fold) {
    fold.open = true;
    fold = fold.parentElement?.closest("details") ?? null;
  }
  target.focus();
}

export interface PlanProblems {
  /** What the server would refuse, in form order, each keyed to the field that fixes it. */
  problems: DraftProblem[];
  /** Save was pressed with problems: the list shows and the named fields are marked. */
  shown: boolean;
  /** A field the list can name: its id and, once marked, aria-invalid and its reason. */
  field: (id: string) => { id: string; "aria-invalid"?: true; "aria-describedby"?: string };
  /** A button that fixes a problem (clearing areas, using the whole area): id and reason. */
  control: (id: string) => { id: string; "aria-describedby"?: string };
  /** The marked field's reason in words, or null while it is not marked. */
  reason: (id: string) => string | null;
  /** Save pressed. False when something must be fixed first: the list shows, focus moves. */
  check: () => boolean;
  /** Another draft is open: nothing is marked until Save is pressed again. */
  reset: () => void;
}

/**
 * The plan editor's save checklist (`actionDraftProblems`, which mirrors the server's
 * validators). Nothing is marked while the plan is written; Save with problems lists them,
 * marks each field and focuses the first, and the list follows the draft as it is fixed.
 */
export function usePlanProblems(draft: ActionDraft): PlanProblems {
  const [shown, setShown] = useState(false);
  const problems = useMemo(() => actionDraftProblems(draft), [draft]);
  const flagged = new Set(shown ? problems.map((problem) => problem.id) : []);
  const describedBy = (id: string) => (flagged.has(id) ? problemId(id) : undefined);
  return {
    problems,
    shown: shown && problems.length > 0,
    field: (id) => ({
      id,
      "aria-invalid": flagged.has(id) || undefined,
      "aria-describedby": describedBy(id),
    }),
    control: (id) => ({ id, "aria-describedby": describedBy(id) }),
    reason: (id) =>
      flagged.has(id)
        ? problems
            .filter((problem) => problem.id === id)
            .map((problem) => problem.message)
            .join(" ")
        : null,
    check: () => {
      const first = problems[0];
      if (!first) return true;
      setShown(true);
      focusField(first.id);
      return false;
    },
    reset: () => setShown(false),
  };
}
