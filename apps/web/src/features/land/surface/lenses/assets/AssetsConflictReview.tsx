import { useState } from "react";

import { GlassButton } from "@twin/ui";

import { mergeAssetDraft, type AssetDraft, type AssetField, type AssetRecord } from "./assetsDraft";

const LABELS: Record<AssetField, string> = {
  name: "Name",
  category: "Category",
  status: "Status",
  description: "Notes",
  geometry: "Shape",
  source: "Origin",
  attributes: "Attributes",
  evidenceIds: "Linked evidence",
  externalRef: "Dataset record",
};

/**
 * Someone saved this asset after the draft began. Fields only one side changed merge on their
 * own; for each field both changed, the person picks the saved value or their own.
 */
export function AssetsConflictReview({
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
    <section className="assets-conflict assets-tool" aria-label="Newer saved version">
      <h4>Someone saved a newer version</h4>
      <p className="land-muted">
        You started from version {baseline.revision}; version {current.revision} is saved now.
        {conflicts.length
          ? " Choose which value to keep where you both made changes."
          : " Your changes do not clash with it."}
      </p>
      {conflicts.map((field) => (
        <fieldset key={field}>
          <legend>{LABELS[field]}</legend>
          <details>
            <summary>Compare</summary>
            <strong>Saved now</strong>
            <pre>{JSON.stringify(current[field], null, 2)?.slice(0, 5000)}</pre>
            <strong>Yours</strong>
            <pre>{JSON.stringify(draft[field], null, 2)?.slice(0, 5000)}</pre>
            {field === "geometry" && (
              <p className="land-muted">
                Shapes are cut at 5,000 characters. Keeping your unfinished shape means checking it
                again before it can be used.
              </p>
            )}
          </details>
          <label className="assets-check">
            <input
              type="radio"
              name={`conflict-${field}`}
              checked={choices[field] === "current"}
              onChange={() => setChoices({ ...choices, [field]: "current" })}
            />
            Use saved {LABELS[field].toLowerCase()}
          </label>
          <label className="assets-check">
            <input
              type="radio"
              name={`conflict-${field}`}
              checked={choices[field] === "draft"}
              onChange={() => setChoices({ ...choices, [field]: "draft" })}
            />
            Keep my {LABELS[field].toLowerCase()}
          </label>
        </fieldset>
      ))}
      <GlassButton
        size="sm"
        disabled={conflicts.some((field) => !choices[field])}
        onClick={() => {
          const next = { ...merged };
          for (const field of conflicts)
            if (choices[field] === "current") Object.assign(next, { [field]: current[field] });
          onResolve(next, choices.geometry !== "current");
        }}
      >
        Use these choices
      </GlassButton>
    </section>
  );
}
