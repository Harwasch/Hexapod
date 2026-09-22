import { create } from "zustand";

import type { UploadedPart } from "@twin/contracts";

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

interface UploadsState {
  items: Record<string, UploadItem>;
  begin: (item: Pick<UploadItem, "id" | "captureId" | "filename" | "bytes">) => void;
  update: (id: string, patch: Partial<Omit<UploadItem, "id">>) => void;
  remove: (id: string) => void;
  /** Drops finished rows; anything still moving is left alone. */
  clearSettled: () => void;
}

export const useUploads = create<UploadsState>()((set) => ({
  items: {},
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
