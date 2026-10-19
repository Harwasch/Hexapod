import type { components } from "@twin/contracts";

import { alignmentDraftKey } from "./recordsModel";

type Point = components["schemas"]["ImageControlPoint"];
type Request = components["schemas"]["ImageRegistrationCreate"];

/** Unfinished point matches for one map image, kept in this browser. */
export interface AlignmentDraft {
  name: string;
  notes: string;
  points: Point[];
  requestKey: string;
}

export const MAX_POINTS = 30;

/** The saved draft, or a fresh one when there is none or it does not read back cleanly. */
export function restoreAlignmentDraft(key: string, title: string): AlignmentDraft {
  const empty = {
    name: title.slice(0, 200),
    notes: "",
    points: [],
    requestKey: crypto.randomUUID(),
  };
  try {
    const raw = localStorage.getItem(key);
    if (!raw || raw.length > 32000) return empty;
    const value = JSON.parse(raw) as Partial<AlignmentDraft>;
    if (
      typeof value.name !== "string" ||
      value.name.length > 200 ||
      typeof value.notes !== "string" ||
      value.notes.length > 2000 ||
      !Array.isArray(value.points) ||
      value.points.length > MAX_POINTS ||
      typeof value.requestKey !== "string" ||
      !/^[a-f0-9-]{36}$/.test(value.requestKey)
    )
      return empty;
    const valid = value.points.every(
      (p) =>
        p &&
        typeof p.label === "string" &&
        p.label.length <= 100 &&
        [p.imageX, p.imageY, p.longitude, p.latitude].every(Number.isFinite) &&
        p.imageX >= 0 &&
        p.imageX <= 8192 &&
        p.imageY >= 0 &&
        p.imageY <= 8192 &&
        Math.abs(p.longitude) <= 180 &&
        Math.abs(p.latitude) <= 90,
    );
    return valid ? (value as AlignmentDraft) : empty;
  } catch {
    return empty;
  }
}

/** Keeps a draft (or forgets an untouched one); false when the browser refused. */
export function persistAlignmentDraft(key: string, draft: AlignmentDraft, title: string): boolean {
  try {
    if (draft.points.length || draft.notes || draft.name !== title.slice(0, 200))
      localStorage.setItem(key, JSON.stringify(draft));
    else localStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

/** Starts a new draft from a saved alignment ("Refine a copy") for the editor to pick up. */
export function seedAlignmentDraft(scope: string, evidenceId: string, request: Request): boolean {
  return persistAlignmentDraft(
    alignmentDraftKey(scope, evidenceId, request.imageSha256),
    {
      name: request.name,
      notes: request.notes ?? "",
      points: request.points,
      requestKey: crypto.randomUUID(),
    },
    "",
  );
}
