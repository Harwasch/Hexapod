import type { components } from "@twin/contracts";

type Action = components["schemas"]["LandActionRead"];
export function ActionPlanSummary({ action }: { action: Action }) {
  return (
    <div className="land-action-saved-summary">
      <strong>{action.title}</strong>
      <p>{action.objective}</p>
      <p>
        Starts {action.startDate} · boundary revision {action.boundaryRevision}
      </p>
      <ol>
        {action.steps.map((step) => (
          <li key={step.id}>
            <strong>{step.title}</strong>
            <p>
              Starts day {step.startDay + 1} · {step.days} {step.days === 1 ? "day" : "days"}
            </p>
            {step.detail && <p>{step.detail}</p>}
            <p>Success measure: {step.successMeasure}</p>
            {step.estimatedCost != null && (
              <p>
                Estimated cost: {step.estimatedCost.toLocaleString()} {action.currency}.{" "}
                {step.costBasis}
              </p>
            )}
            {!!step.resources?.length && <p>Resources: {step.resources.join(", ")}</p>}
            {!!step.dependsOn?.length && (
              <p>
                After:{" "}
                {step.dependsOn
                  .map((id) => action.steps.find((row) => row.id === id)?.title ?? id)
                  .join(", ")}
              </p>
            )}
            {step.footprint && <p>Has a mapped work area.</p>}
          </li>
        ))}
      </ol>
      {!!action.constraints?.length && (
        <>
          <h4>Constraints</h4>
          <ul>
            {action.constraints.map((item, index) => (
              <li key={index}>
                {item.text} · {item.resolved ? "Resolved" : "Unresolved"}
                {item.resolution ? `: ${item.resolution}` : ""}
              </li>
            ))}
          </ul>
        </>
      )}
      {!!action.assumptions?.length && (
        <>
          <h4>Assumptions</h4>
          <ul>
            {action.assumptions.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>
        </>
      )}
      <p>
        {action.evidenceIds?.length ?? 0} evidence references · {action.features?.length ?? 0} asset
        references · {action.exclusions?.length ?? 0} excluded areas
      </p>
      {action.scenario && <p>Based on saved scenario revision {action.scenario.revision}.</p>}
    </div>
  );
}
