/**
 * PUT one part straight to object storage, reporting bytes as they leave.
 *
 * This is the piece worth sharing between the console and the standalone page a phone
 * opens, and it is deliberately free of every console dependency — no stores, no
 * `openapi-fetch` client, no token. Two reasons, and both are load-bearing:
 *
 * - **It must not carry `Authorization`.** The presigned URL is the credential; an extra
 *   auth header is exactly the kind of thing that makes a SigV4 signature stop matching.
 *   Because this never touches `client.ts`, the write-token middleware cannot reach it
 *   even by accident.
 * - **It must not drag the console into the phone's bundle.** `upload.html` exists so a
 *   phone does not download CesiumJS to pick one file; importing through `client.ts`
 *   would quietly pull the stores back in.
 *
 * `XMLHttpRequest` rather than `fetch` because only XHR reports upload progress, which is
 * the whole point of a progress bar over a multi-gigabyte video.
 */
import { ApiError } from "./error";

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

/** The part size the API derives from a declared file size, mirrored for the client. */
export function partSizeFor(fileBytes: number, partBytes: number, partNumber: number): number {
  const start = (partNumber - 1) * partBytes;
  return Math.min(partBytes, Math.max(0, fileBytes - start));
}
