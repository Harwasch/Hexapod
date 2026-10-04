import { create } from "zustand";

import type { components, UploadedPart } from "@twin/contracts";

/** A minted phone handoff: the QR code and the link it encodes, and until when it works. */
export type CaptureHandoff = components["schemas"]["CaptureHandoff"];

/**
 * Client-side upload state: the bytes this browser has actually put on the wire.
 *
 * Deliberately separate from TanStack Query. The server knows a file's part count and
 * its `parts-completed` watermark, and that is all it can know — the parts themselves
 * go straight to object storage, so byte-level progress exists only here, and only for
 * as long as the tab is open. Everything durable (the capture, its files, its jobs)
 * stays in the query cache. Shaped after `state/sites.ts`: one record per thing,
 * patched immutably.
 */
export type UploadPhase =
  "queued" | "registering" | "uploading" | "completing" | "complete" | "error" | "cancelled";

export interface UploadItem {
  /** Client-side id: a file has one of these before the API has ever heard of it. */
  id: string;
  captureId: string;
  /** The API's file id, once the row exists. Null until `register` returns. */
  fileId: string | null;
  filename: string;
  bytes: number;
  /** Bytes acknowledged by storage, plus the live part's `loaded`. */
  uploaded: number;
  partsCompleted: number;
  partsTotal: number | null;
  phase: UploadPhase;
  error: string | null;
  /** ETags collected so far. A retry resumes from here rather than from part 1. */
  parts: UploadedPart[];
  startedAt: number;
}

/**
 * Making a capture -- from dropped files, or an empty one for a phone -- and how the last one
 * went: what the Add panel's drop zone and phone button say. Here, not in the panel, for the
 * same reason as `phone` below.
 */
export interface CaptureCreation {
  /** A capture is being made (and, for dropped files, uploaded): the panel's controls wait. */
  busy: boolean;
  /** Why the last one failed, in words, or null. */
  error: string | null;
  /** The last drop can be tried again as it was (after a refused write token). */
  canRetryDrop: boolean;
}

interface UploadsState {
  items: Record<string, UploadItem>;
  begin: (item: Pick<UploadItem, "id" | "captureId" | "filename" | "bytes">) => void;
  update: (id: string, patch: Partial<Omit<UploadItem, "id">>) => void;
  remove: (id: string) => void;
  /** Drops finished rows; anything still moving is left alone. */
  clearSettled: () => void;
  creation: CaptureCreation;
  setCreation: (patch: Partial<CaptureCreation>) => void;
  /**
   * "New capture from phone": the capture made for a phone to upload into, and its handoff
   * (the QR code) once minted; null when none is showing.
   *
   * Kept here rather than in the Add panel, which unmounts when Add closes or its tab changes:
   * the Captures panel before it stayed mounted and kept them. A handoff in progress has a
   * phone pointing at its code, and minting another on the way back would replace that code.
   */
  phone: { captureId: string; handoff: CaptureHandoff | null } | null;
  setPhone: (captureId: string | null) => void;
  /** The handoff minted for `captureId`, if that is still the phone capture showing. */
  setPhoneHandoff: (captureId: string, handoff: CaptureHandoff) => void;
}

export const useUploads = create<UploadsState>()((set) => ({
  items: {},
  creation: { busy: false, error: null, canRetryDrop: false },
  setCreation: (patch) => set((s) => ({ creation: { ...s.creation, ...patch } })),
  phone: null,
  setPhone: (captureId) => set({ phone: captureId ? { captureId, handoff: null } : null }),
  setPhoneHandoff: (captureId, handoff) =>
    set((s) => (s.phone?.captureId === captureId ? { phone: { captureId, handoff } } : s)),
  begin: (item) =>
    set((s) => ({
      items: {
        ...s.items,
        [item.id]: {
          ...item,
          fileId: null,
          uploaded: 0,
          partsCompleted: 0,
          partsTotal: null,
          phase: "queued",
          error: null,
          parts: [],
          startedAt: Date.now(),
        },
      },
    })),
  update: (id, patch) =>
    set((s) => {
      const current = s.items[id];
      if (!current) return s;
      return { items: { ...s.items, [id]: { ...current, ...patch } } };
    }),
  remove: (id) =>
    set((s) => ({
      items: Object.fromEntries(Object.entries(s.items).filter(([key]) => key !== id)),
    })),
  clearSettled: () =>
    set((s) => ({
      items: Object.fromEntries(
        Object.entries(s.items).filter(
          ([, item]) => item.phase !== "complete" && item.phase !== "cancelled",
        ),
      ),
    })),
}));

/** The uploads belonging to one capture, oldest first — the order they were dropped in. */
export function uploadsForCapture(
  items: Record<string, UploadItem>,
  captureId: string,
): UploadItem[] {
  return Object.values(items)
    .filter((item) => item.captureId === captureId)
    .sort((a, b) => a.startedAt - b.startedAt);
}

/** True while any upload is still moving: the panel uses it to keep polling honest. */
export function isSettled(item: UploadItem): boolean {
  return item.phase === "complete" || item.phase === "error" || item.phase === "cancelled";
}
