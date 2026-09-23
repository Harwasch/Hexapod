import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { capturesApi, queryKeys } from "@/api/queries";
import { abortCaptureFile, isAbort, uploadCaptureFile } from "@/api/uploads";
import { describeError } from "@/lib/log";
import { useUploads } from "@/state/uploads";
import { useViewer } from "@/state/viewer";

import { captureName, classify } from "./recipes";

/** Six decimals is ~0.1 m: finer than a hand placement means anything. */
function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

/**
 * The point on the ground directly under the camera, as the capture's placement guess.
 *
 * A splat file carries no EXIF and no poses -- it is geometry with no idea where on Earth
 * it belongs -- so without this the pipeline runs correctly and lands the site at (0, 0),
 * in the Gulf of Guinea.  Where the camera was when you started the capture is the best
 * guess available, and the manifest records it as `manual`, uncertainty 10 m,
 * `scaleSource: unresolved`, so nothing downstream mistakes it for a survey. B4's
 * placement editor is what replaces the guess.
 */
function cameraPlacement(): { lat: number; lon: number } {
  const camera = useViewer.getState().camera;
  return { lat: round6(camera.latitude), lon: round6(camera.longitude) };
}

/** "Phone capture 14:05" -- a name that says where it came from and when. */
function phoneCaptureName(now: Date = new Date()): string {
  const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `Phone capture ${time}`;
}

/**
 * `File` handles for uploads in flight, outside React and outside the store.
 *
 * A `File` is a live handle to something on disk, not data: it cannot be serialised, and
 * putting it in the store would mean the store no longer holds plain state. Keeping it
 * here lets a retry resume the same file without re-prompting for it.
 */
const handles = new Map<string, File>();
const controllers = new Map<string, AbortController>();

let counter = 0;
function nextUploadId(): string {
  counter += 1;
  return `upload-${counter}-${Date.now()}`;
}

function messageOf(error: unknown): string {
  if (error instanceof ApiError) {
    const fields = error.fieldErrors;
    return fields.length ? fields.join("; ") : error.message;
  }
  return describeError(error);
}

export interface CaptureUploads {
  /** Create a capture from these files and upload every one of them. */
  start: (files: File[]) => Promise<void>;
  /**
   * Create an empty capture for a phone to upload into, and return its id -- or null if
   * the API refused, in which case `error` says why.
   */
  startFromPhone: () => Promise<string | null>;
  /** Resume one failed upload from the parts it already has. */
  retry: (uploadId: string) => Promise<void>;
  /** Stop one upload and abandon its multipart upload server-side. */
  cancel: (uploadId: string) => void;
  /** Run the last drop again — what the token affordance offers after a 401. */
  retryLastDrop: () => Promise<void>;
  canRetryDrop: boolean;
  error: string | null;
  clearError: () => void;
  busy: boolean;
}

/** Drives the upload lifecycle: create the capture, register each file, move the bytes. */
export function useCaptureUploads(): CaptureUploads {
  const client = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [canRetryDrop, setCanRetryDrop] = useState(false);
  const lastDrop = useRef<File[]>([]);

  const invalidate = useCallback(() => {
    void client.invalidateQueries({ queryKey: queryKeys.captures });
    void client.invalidateQueries({ queryKey: queryKeys.jobs });
  }, [client]);

  const runOne = useCallback(
    async (uploadId: string, captureId: string, file: File) => {
      const store = useUploads.getState();
      const controller = new AbortController();
      controllers.set(uploadId, controller);
      const existing = store.items[uploadId];
      const resume = existing?.fileId
        ? { fileId: existing.fileId, parts: existing.parts }
        : undefined;
      store.update(uploadId, {
        phase: resume ? "uploading" : "registering",
        error: null,
      });
      try {
        const completed = await uploadCaptureFile({
          captureId,
          file,
          resume,
          signal: controller.signal,
          onRegistered: (fileId, partsTotal) =>
            useUploads.getState().update(uploadId, { fileId, partsTotal, phase: "uploading" }),
          // Kept so a retry resumes from the next part rather than re-registering the
          // file and re-uploading everything that already landed.
          onPart: (part) => {
            const current = useUploads.getState().items[uploadId];
            if (current)
              useUploads.getState().update(uploadId, { parts: [...current.parts, part] });
          },
          onProgress: ({ uploaded, partsCompleted, partsTotal }) =>
            useUploads.getState().update(uploadId, { uploaded, partsCompleted, partsTotal }),
        });
        useUploads.getState().update(uploadId, {
          phase: "complete",
          uploaded: completed.bytes ?? file.size,
          partsCompleted: completed.partsCompleted,
          partsTotal: completed.partsTotal,
        });
        handles.delete(uploadId);
      } catch (cause) {
        if (isAbort(cause)) {
          useUploads.getState().update(uploadId, { phase: "cancelled" });
        } else {
          useUploads.getState().update(uploadId, { phase: "error", error: messageOf(cause) });
          throw cause;
        }
      } finally {
        controllers.delete(uploadId);
        invalidate();
      }
    },
    [invalidate],
  );

  const start = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      lastDrop.current = files;
      setBusy(true);
      setError(null);
      setCanRetryDrop(false);
      try {
        const proposal = classify(files);
        const capture = await capturesApi.create({
          name: captureName(files),
          kind: proposal.kind,
          // The proposed recipe rides along so the card can offer it later without
          // re-deriving it from filenames the API has already stored.
          metadata: { recipe: proposal.recipe, origin: "console", ...cameraPlacement() },
        });
        invalidate();
        for (const file of files) {
          const uploadId = nextUploadId();
          handles.set(uploadId, file);
          useUploads.getState().begin({
            id: uploadId,
            captureId: capture.id,
            filename: file.name,
            bytes: file.size,
          });
          // Sequential on purpose: parallel uploads of a 12 GB video would fight each
          // other for the same uplink and make every progress bar a lie.
          await runOne(uploadId, capture.id, file);
        }
      } catch (cause) {
        setError(messageOf(cause));
        setCanRetryDrop(true);
      } finally {
        setBusy(false);
      }
    },
    [invalidate, runOne],
  );

  const startFromPhone = useCallback(async () => {
    setBusy(true);
    setError(null);
    setCanRetryDrop(false);
    try {
      const capture = await capturesApi.create({
        name: phoneCaptureName(),
        // Nothing has been picked yet, so this is a guess, and "video" is the likeliest
        // thing a phone sends. It is display-only: the card reads the files once they
        // land, and no recipe is stored, so the card proposes one from what arrived
        // rather than from this.
        kind: "video",
        metadata: { origin: "phone", ...cameraPlacement() },
      });
      invalidate();
      return capture.id;
    } catch (cause) {
      setError(messageOf(cause));
      return null;
    } finally {
      setBusy(false);
    }
  }, [invalidate]);

  const retry = useCallback(
    async (uploadId: string) => {
      const item = useUploads.getState().items[uploadId];
      const file = handles.get(uploadId);
      if (!item || !file) return;
      setError(null);
      try {
        await runOne(uploadId, item.captureId, file);
      } catch (cause) {
        setError(messageOf(cause));
      }
    },
    [runOne],
  );

  const cancel = useCallback((uploadId: string) => {
    controllers.get(uploadId)?.abort();
    const item = useUploads.getState().items[uploadId];
    if (item?.fileId) {
      void abortCaptureFile(item.captureId, item.fileId).catch(() => {
        // Storage keeps the orphaned parts until its lifecycle rule sweeps them; the
        // upload is over either way, and nothing here is worth a second error banner.
      });
    }
  }, []);

  const retryLastDrop = useCallback(() => start(lastDrop.current), [start]);

  return {
    start,
    startFromPhone,
    retry,
    cancel,
    retryLastDrop,
    canRetryDrop,
    error,
    clearError: () => setError(null),
    busy,
  };
}
