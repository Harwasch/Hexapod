import { useState } from "react";
import {
  mergeAssetDraft,
  type AssetDraft,
  type AssetField,
  type AssetRecord,
} from "./inventoryDraft";
const labels: Record<AssetField, string> = {
  name: "Name",
  category: "Category",
  status: "Identity status",
  description: "Notes",
  geometry: "Geometry",
  source: "Source",
  attributes: "Attributes",
  evidenceIds: "Evidence",
  externalRef: "Dataset identity",
};
export function InventoryConflictReview({
  baseline,
  draft,
  current,
  working,
  onResolve,
}: {
  baseline: AssetRecord;
  draft: AssetDraft;
  current: AssetRecord;
  working: boolean;
  onResolve: (draft: AssetDraft, keepWorking: boolean) => void;
}) {
  const [choices, setChoices] = useState<Partial<Record<AssetField, "current" | "draft">>>({});
  const { merged, conflicts } = mergeAssetDraft(baseline, draft, current, working);
  return (
    <section className="inventory-conflict" aria-label="Review newer asset revision">
      <h4>This asset has a newer saved revision</h4>
      <p>
        Your draft started from revision {baseline.revision}; revision {current.revision} is now
        saved. Fields you did not change will use the saved values. Choose how to resolve each
        conflicting field.
      </p>
      {conflicts.map((field) => (
        <fieldset key={field}>
          <legend>{labels[field]}</legend>
          <details>
            <summary>Compare {labels[field].toLowerCase()}</summary>
            <strong>Saved now</strong>
            <pre>{JSON.stringify(current[field], null, 2)?.slice(0, 5000)}</pre>
            <strong>Your draft</strong>
            <pre>{JSON.stringify(draft[field], null, 2)?.slice(0, 5000)}</pre>
            {field === "geometry" && (
              <p>
                Geometry text is limited to 5,000 characters per version. Keeping your unfinished
                geometry requires another preview before it can be applied.
              </p>
            )}
          </details>
          <label>
            <input
              type="radio"
              name={`conflict-${field}`}
              checked={choices[field] === "current"}
              onChange={() => setChoices({ ...choices, [field]: "current" })}
            />
            Use saved {labels[field].toLowerCase()}
          </label>
          <label>
            <input
              type="radio"
              name={`conflict-${field}`}
              checked={choices[field] === "draft"}
              onChange={() => setChoices({ ...choices, [field]: "draft" })}
            />
            Keep my {labels[field].toLowerCase()}
          </label>
        </fieldset>
      ))}
      {!conflicts.length && <p>Your changes do not conflict with the newer saved fields.</p>}
      <button
        type="button"
        disabled={conflicts.some((field) => !choices[field])}
        onClick={() => {
          const next = { ...merged };
          for (const field of conflicts)
            if (choices[field] === "current") Object.assign(next, { [field]: current[field] });
          onResolve(next, choices.geometry !== "current");
        }}
      >
        Apply reviewed merge to draft
      </button>
    </section>
  );
}
