/**
 * The chunked uploader.
 *
 * `openapi-fetch` (and `fetch`) cannot report upload progress, so the part PUTs run on
 * `XMLHttpRequest`, which can. Two consequences worth stating:
 *
 * - The part PUTs go to object storage, not the API. They carry **no** `Authorization`
 *   header: the presigned URL is the credential, and an extra auth header is exactly the
 *   kind of thing that makes a SigV4 signature stop matching. Because they never go
 *   through `api`, the write-token middleware cannot reach them by accident.
 * - Failures are still raised as `ApiError`, so `errorMessage()` and `fieldErrors` in the
 *   UI keep working over storage errors and API errors alike.
 *
 * Parts are presigned a window at a time (32 by default, `PRESIGN_WINDOW_PARTS`): a 12 GB
 * video is 1536 parts, whose URLs would be ~590 KB of JSON and would start expiring long
 * before a phone on cellular reached the end of them.
 */
import type { CaptureFile, CaptureFileUpload, UploadedPart, UploadWindow } from "@twin/contracts";

import { ApiError, api, unwrap } from "./client";

export interface UploadProgress {
  /** Bytes storage has taken, including the part in flight. */
  uploaded: number;
  partsCompleted: number;
  partsTotal: number | null;
}

export interface UploadOptions {
  captureId: string;
  file: File;
  /** Resume state from an earlier attempt: the row already exists and these parts are done. */
  resume?: { fileId: string; parts: UploadedPart[] } | undefined;
  onRegistered?: (fileId: string, partsTotal: number | null) => void;
  /** Each part as storage acknowledges it — the caller keeps these so a retry can resume. */
  onPart?: (part: UploadedPart) => void;
  onProgress?: (progress: UploadProgress) => void;
  signal?: AbortSignal | undefined;
}

/** Raised when the caller aborts; callers distinguish it from a real failure. */
export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/** PUT one part straight to storage, reporting bytes as they leave. Resolves to the part's ETag. */
export function putPart(
  url: string,
  body: Blob,
  options: { onProgress?: (loaded: number) => void; signal?: AbortSignal | undefined } = {},
): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url, true);
    xhr.responseType = "text";
    const onAbort = () => xhr.abort();
    options.signal?.addEventListener("abort", onAbort, { once: true });
    const done = () => options.signal?.removeEventListener("abort", onAbort);
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) options.onProgress?.(event.loaded);
    });
    xhr.addEventListener("load", () => {
      done();
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(
          new ApiError(xhr.status, undefined, `Object storage refused the part (${xhr.status})`),
        );
        return;
      }
      const etag = xhr.getResponseHeader("ETag");
      if (!etag) {
        // Not a hypothetical: without ExposeHeaders: ["ETag"] on the bucket's CORS rule
        // the browser is handed the header and refuses to show it, and completion — which
        // is a list of part ETags — becomes impossible.
        reject(
          new ApiError(
            xhr.status,
            undefined,
            "Object storage did not expose an ETag for this part. The bucket's CORS rule " +
              'needs ExposeHeaders: ["ETag"] (infra/cors/upload.json).',
          ),
        );
        return;
      }
      resolve(etag.replace(/"/g, ""));
    });
    xhr.addEventListener("error", () => {
      done();
      reject(new ApiError(0, undefined, "Could not reach object storage to upload this part."));
    });
    xhr.addEventListener("timeout", () => {
      done();
      reject(new ApiError(0, undefined, "Uploading this part timed out."));
    });
    xhr.addEventListener("abort", () => {
      done();
      reject(new DOMException("Upload cancelled", "AbortError"));
    });
    xhr.send(body);
  });
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new DOMException("Upload cancelled", "AbortError");
}

/** Register the file (or pick up an existing one) and walk its presigned windows to completion. */
export async function uploadCaptureFile(options: UploadOptions): Promise<CaptureFile> {
  const { captureId, file, signal } = options;
  const parts: UploadedPart[] = [...(options.resume?.parts ?? [])];
  let fileId = options.resume?.fileId ?? null;
  let window: UploadWindow;

  throwIfAborted(signal);
  if (fileId === null) {
    const registered = await unwrap<CaptureFileUpload>(
      api.POST("/api/v1/captures/{capture_id}/files", {
        params: { path: { capture_id: captureId } },
        body: {
          filename: file.name,
          contentType: file.type || null,
          bytes: file.size,
        },
      }),
    );
    fileId = registered.file.id;
    window = registered.upload;
    options.onRegistered?.(fileId, registered.file.partsTotal);
  } else {
    // The resume path is just the next window: re-presigning never rewinds the server's
    // progress watermark, so a retry costs one request, not the upload so far.
    window = await presignFrom(captureId, fileId, parts.length + 1);
  }

  let uploaded = parts.reduce(
    (sum, part) => sum + partSize(file, window.partSize, part.partNumber),
    0,
  );
  for (;;) {
    for (const part of window.parts) {
      throwIfAborted(signal);
      if (parts.some((done) => done.partNumber === part.partNumber)) continue;
      const start = (part.partNumber - 1) * window.partSize;
      const blob = file.slice(start, Math.min(start + window.partSize, file.size));
      const base = uploaded;
      const etag = await putPart(part.url, blob, {
        signal,
        onProgress: (loaded) =>
          options.onProgress?.({
            uploaded: base + loaded,
            partsCompleted: parts.length,
            partsTotal: window.partsTotal,
          }),
      });
      const done = { partNumber: part.partNumber, etag };
      parts.push(done);
      options.onPart?.(done);
      uploaded = base + blob.size;
      options.onProgress?.({
        uploaded,
        partsCompleted: parts.length,
        partsTotal: window.partsTotal,
      });
    }
    if (window.nextPartNumber === null) break;
    window = await presignFrom(captureId, fileId, window.nextPartNumber);
    // A window with nothing in it would otherwise spin forever.
    if (window.parts.length === 0) break;
  }

  throwIfAborted(signal);
  return unwrap<CaptureFile>(
    api.POST("/api/v1/captures/{capture_id}/files/{file_id}/complete", {
      params: { path: { capture_id: captureId, file_id: fileId } },
      body: { parts: [...parts].sort((a, b) => a.partNumber - b.partNumber) },
    }),
  );
}

function presignFrom(
  captureId: string,
  fileId: string,
  firstPartNumber: number,
): Promise<UploadWindow> {
  return unwrap<UploadWindow>(
    api.POST("/api/v1/captures/{capture_id}/files/{file_id}/parts", {
      params: { path: { capture_id: captureId, file_id: fileId } },
      body: { firstPartNumber },
    }),
  );
}

/** Bytes in a given part of this file — the last one is short. */
function partSize(file: File, size: number, partNumber: number): number {
  const start = (partNumber - 1) * size;
  return Math.max(0, Math.min(start + size, file.size) - start);
}

/** Abandon a half-finished upload server-side, so storage is not left holding the parts. */
export function abortCaptureFile(captureId: string, fileId: string): Promise<CaptureFile> {
  return unwrap<CaptureFile>(
    api.POST("/api/v1/captures/{capture_id}/files/{file_id}/abort", {
      params: { path: { capture_id: captureId, file_id: fileId } },
    }),
  );
}
