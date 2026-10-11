import { Check, Info, Pencil } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";

import { footprintAreaM2 } from "@twin/geo";
import { GlassButton, GlassInput, GlassPanel, GlassTextarea } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { useLand, type BoundaryOperation } from "@/state/land";

import { BoundaryReferences } from "../BoundaryReferences";
import { areaLabel, beginLand, frameBoundary, saveDraft, useLandFlow } from "./landFlow";
import { sourceLabel, sourceMeaning } from "./sourceLabels";

const SHAPE_TOOLS: readonly [BoundaryOperation, string][] = [
  ["union", "Add an area"],
  ["difference", "Cut out an area"],
  ["intersection", "Keep only an overlap"],
];

/** A drawn, picked or imported outline before it is saved: name it, check it, save it. */
export function LandReview() {
  const scene = useScene();
  const queryClient = useQueryClient();
  const draft = useLand((s) => s.draft);
  const active = useLand((s) => s.active);
  const mode = useLand((s) => s.mode);
  const busy = useLandFlow((s) => s.busy);
  if (!draft) return null;
  const editing = mode === "edit";
  return (
    <GlassPanel strong className="land-card" data-testid="land-review">
      <div className="land-eyebrow-row">
        <span className="land-lens-label">{active ? "Revise boundary" : "New land"}</span>
        <button
          type="button"
          className="land-icon-button"
          aria-label="Show boundary on map"
          onClick={() => frameBoundary(scene, draft.boundary)}
        >
          <Info size={16} aria-hidden="true" />
        </button>
      </div>
      <label className="land-field" htmlFor="land-draft-name">
        Name this land
        <GlassInput
          id="land-draft-name"
          autoComplete="off"
          maxLength={200}
          value={draft.name}
          onChange={(event) => useLand.getState().updateDraft({ name: event.target.value })}
        />
      </label>
      <p className="land-review__area">
        <strong>≈ {areaLabel(footprintAreaM2(draft.boundary))}</strong>
        <span title={sourceMeaning(draft.source.meaning)}>{sourceLabel(draft.source)}</span>
      </p>
      <div className="land-row">
        <GlassButton
          variant="primary"
          disabled={busy || !draft.name.trim()}
          leadingIcon={<Check size={16} aria-hidden="true" />}
          onClick={() => void saveDraft(queryClient)}
        >
          {busy ? "Saving…" : active ? "Save revision" : "Save this land"}
        </GlassButton>
        <GlassButton
          disabled={!scene || busy}
          active={editing}
          leadingIcon={<Pencil size={16} aria-hidden="true" />}
          onClick={() => beginLand(scene, editing ? "browse" : "edit")}
        >
          {editing ? "Done adjusting" : "Adjust corners"}
        </GlassButton>
        <GlassButton variant="ghost" disabled={busy} onClick={() => useLand.getState().cancel()}>
          Discard
        </GlassButton>
      </div>
      <details className="land-details">
        <summary>Shape tools</summary>
        <div className="land-details__body">
          <div className="land-row">
            {SHAPE_TOOLS.map(([operation, label]) => (
              <GlassButton
                key={operation}
                size="sm"
                disabled={!scene || busy}
                onClick={() => {
                  useLand.getState().setBoundaryOperation(operation);
                  beginLand(scene, "draw");
                }}
              >
                {label}
              </GlassButton>
            ))}
            <GlassButton
              size="sm"
              disabled={!scene || busy}
              onClick={() => beginLand(scene, "split")}
            >
              Split with a line
            </GlassButton>
          </div>
        </div>
      </details>
      <details className="land-details">
        <summary>Notes and source</summary>
        <div className="land-details__body">
          <label className="land-field" htmlFor="land-draft-notes">
            Notes
            <GlassTextarea
              id="land-draft-notes"
              rows={2}
              maxLength={5000}
              value={draft.description ?? ""}
              placeholder="What would you like to understand about this place?"
              onChange={(event) =>
                useLand.getState().updateDraft({ description: event.target.value })
              }
            />
          </label>
          <p>{draft.source.label}</p>
          <p className="land-muted">{sourceMeaning(draft.source.meaning)}</p>
          {draft.source.attribution && <p className="land-muted">{draft.source.attribution}</p>}
          <BoundaryReferences sources={draft.source.references ?? []} />
        </div>
      </details>
    </GlassPanel>
  );
}
