import type { DraftProblem } from "../../../actionDraft";
import { focusField, problemId, PROBLEMS_ID, type PlanProblems } from "./planProblems";

/** A marked field's reason, under it: focus lands there, often a screen from the list. */
export function FieldProblem({ id, checklist }: { id: string; checklist: PlanProblems }) {
  const reason = checklist.reason(id);
  if (!reason) return null;
  return (
    <p className="plans-field-problem" id={problemId(id)}>
      {reason}
    </p>
  );
}

/** "Before saving, fix 2 things:", each a way back to its field. */
export function PlanProblemList({ problems }: { problems: DraftProblem[] }) {
  const count = problems.length;
  return (
    <div
      className="land-alert plans-problems"
      role="alert"
      id={PROBLEMS_ID}
      data-testid="plans-draft-problems"
    >
      <p>
        Before saving, fix {count} thing{count === 1 ? "" : "s"}:
      </p>
      <ul>
        {problems.map((problem, index) => (
          <li key={`${problem.id}:${index}`}>
            <button type="button" className="land-link" onClick={() => focusField(problem.id)}>
              {problem.message}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
