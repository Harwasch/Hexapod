import { useId } from "react";
import { Check, Download, PenLine } from "lucide-react";

import type { LandArea } from "@twin/contracts";
import { GlassButton, GlassInput, GlassTextarea } from "@twin/ui";

import { useSettings } from "@/state/settings";

import { LandEvidencePicker } from "../../../LandEvidencePicker";
import { AssetsAttributes } from "./AssetsAttributes";
import { AssetsConflictReview } from "./AssetsConflictReview";
import { downloadAssetDraft } from "./assetsDraft";
import { AssetsGeometryEditor, type GeometryPreview } from "./AssetsGeometryEditor";
import {
  CATEGORIES,
  CATEGORY_ORDER,
  SHAPE_NAMES,
  sizeLabel,
  STATUSES,
  whereLabel,
  type AssetStatus,
} from "./assetsModel";
import type { AssetDraftState } from "./useAssetDraft";

/** Where the shape sits against the land, from the server's check of it. */
export type Placement = Pick<GeometryPreview, "intersectsLand" | "distanceM">;

/**
 * The asset being added or changed: its name, what kind of thing it is and whether it has been
 * confirmed, with its shape shown on the map. Notes, attributes, evidence and exact
 * coordinates are under "More details". A change to a saved asset says what changed.
 */
export function AssetsEditor({
  land,
  state,
  placement,
  active,
  onPlacement,
  onRedraw,
}: {
  land: LandArea;
  state: AssetDraftState;
  placement: Placement | null;
  /** False while the land panel is closed. */
  active: boolean;
  onPlacement: (placement: Placement) => void;
  onRedraw: () => void;
}) {
  const id = useId();
  const units = useSettings((s) => s.units);
  const { draft, editing } = state;
  if (!draft) return null;
  const status: AssetStatus = draft.status ?? "candidate";
  const size = sizeLabel(draft.geometry, units);
  const shapeLine = [
    SHAPE_NAMES[draft.geometry.type],
    size,
    placement && whereLabel(placement, units),
  ]
    .filter(Boolean)
    .join(" · ");
  const blocked =
    state.busy ||
    state.geometryEditing ||
    Boolean(state.concurrent) ||
    !draft.name.trim() ||
    (Boolean(editing) && !state.revisionNote.trim());

  return (
    <form
      className="assets-form"
      aria-label={editing ? "Edit asset" : "New asset"}
      onSubmit={(event) => {
        event.preventDefault();
        void state.save();
      }}
    >
      {state.error && (
        <p className="land-alert" role="alert">
          {state.error}
        </p>
      )}
      {state.notice && (
        <p className="land-muted" role="status">
          {state.notice}
        </p>
      )}
      {state.storageError && (
        <p className="land-note" role="alert">
          This browser could not keep a copy of this draft. Keep the page open, or download it under
          More details.
        </p>
      )}
      <fieldset disabled={state.busy} className="assets-form__fields">
        <label className="land-field" htmlFor={`${id}-name`}>
          Name
          <GlassInput
            id={`${id}-name`}
            required
            autoComplete="off"
            maxLength={200}
            value={draft.name}
            placeholder="North gate"
            onChange={(event) => state.update({ name: event.target.value })}
          />
        </label>
        <fieldset className="assets-chips">
          <legend className="land-field">Kind</legend>
          {CATEGORY_ORDER.map((category) => (
            <label key={category} className="assets-chip">
              <input
                type="radio"
                name={`${id}-category`}
                value={category}
                checked={draft.category === category}
                onChange={() => state.update({ category })}
              />
              <i style={{ background: CATEGORIES[category].color }} aria-hidden="true" />
              {CATEGORIES[category].label}
            </label>
          ))}
        </fieldset>
        <fieldset className="assets-chips">
          <legend className="land-field">Status</legend>
          {(editing
            ? (["candidate", "confirmed", "retired"] as const)
            : (["candidate", "confirmed"] as const)
          ).map((value) => (
            <label key={value} className="assets-chip">
              <input
                type="radio"
                name={`${id}-status`}
                value={value}
                checked={status === value}
                onChange={() => state.update({ status: value })}
              />
              {STATUSES[value]}
            </label>
          ))}
        </fieldset>
        <div className="assets-shape">
          <span>{shapeLine}</span>
          {!state.geometryEditing && (
            <div className="land-row">
              <GlassButton
                size="sm"
                disabled={!active}
                leadingIcon={<PenLine size={14} aria-hidden="true" />}
                onClick={onRedraw}
              >
                Redraw on the map
              </GlassButton>
              <GlassButton
                size="sm"
                variant="ghost"
                onClick={() => {
                  state.setWorking(null);
                  state.setGeometryEditing(true);
                }}
              >
                Edit coordinates
              </GlassButton>
            </div>
          )}
        </div>
        {state.geometryEditing && (
          <AssetsGeometryEditor
            key={draft.requestKey}
            landId={land.id}
            boundaryRevision={land.revision}
            geometry={draft.geometry}
            initialShape={state.working}
            active={active}
            onDraft={state.setWorking}
            onApply={(geometry, preview) => {
              state.update({ geometry });
              state.setGeometryEditing(false);
              state.setWorking(null);
              onPlacement(preview);
            }}
            onCancel={() => {
              state.setGeometryEditing(false);
              state.setWorking(null);
            }}
          />
        )}
        {state.concurrent && editing && (
          <AssetsConflictReview
            key={state.concurrent.revision}
            baseline={editing}
            draft={draft}
            current={state.concurrent}
            working={Boolean(state.working)}
            onResolve={state.resolveConflict}
          />
        )}
        {editing && (
          <label className="land-field" htmlFor={`${id}-note`}>
            What changed?
            <GlassTextarea
              id={`${id}-note`}
              required
              rows={2}
              maxLength={1000}
              value={state.revisionNote}
              placeholder="Moved the gate 10 m east after walking the fence"
              onChange={(event) => state.setRevisionNote(event.target.value)}
            />
          </label>
        )}
        <details className="land-details">
          <summary>More details</summary>
          <div className="land-details__body">
            <label className="land-field" htmlFor={`${id}-notes`}>
              Notes
              <GlassTextarea
                id={`${id}-notes`}
                rows={3}
                maxLength={5000}
                value={draft.description ?? ""}
                onChange={(event) => state.update({ description: event.target.value })}
              />
            </label>
            <AssetsAttributes
              key={state.version}
              value={draft.attributes}
              onChange={(attributes) => state.update({ attributes })}
            />
            <div className="assets-tool">
              <LandEvidencePicker
                key={state.version}
                landId={land.id}
                value={draft.evidenceIds ?? []}
                onChange={(evidenceIds) => state.update({ evidenceIds })}
                label="Linked evidence"
                limit={100}
              />
            </div>
            <p className="land-muted">
              From: {draft.source.label}. Confirming what something is does not establish who owns
              it.
            </p>
            <div className="land-row">
              {editing && (
                <GlassButton size="sm" variant="ghost" onClick={() => void state.checkLatest()}>
                  Check for a newer version
                </GlassButton>
              )}
              <GlassButton
                size="sm"
                variant="ghost"
                leadingIcon={<Download size={14} aria-hidden="true" />}
                onClick={() =>
                  downloadAssetDraft({
                    version: 1,
                    landId: land.id,
                    boundaryRevision: land.revision,
                    draft,
                    editing,
                    revisionNote: state.revisionNote,
                    working: state.working,
                  })
                }
              >
                Download draft
              </GlassButton>
            </div>
          </div>
        </details>
        {state.geometryEditing && (
          <p className="land-muted">Use or discard the coordinate changes before saving.</p>
        )}
        <div className="land-row">
          <GlassButton
            type="submit"
            variant="primary"
            disabled={blocked}
            loading={state.busy}
            leadingIcon={<Check size={16} aria-hidden="true" />}
          >
            {editing ? "Save changes" : "Add asset"}
          </GlassButton>
          <GlassButton variant="ghost" onClick={state.discard}>
            Discard
          </GlassButton>
        </div>
      </fieldset>
    </form>
  );
}
