/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- The labeled scroll region needs keyboard focus to pan a zoomed page. */
import { useEffect, useRef, useState } from "react";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";

export function LandDocumentImage({
  landId,
  documentId,
  page,
}: {
  landId: string;
  documentId: string;
  page: number;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null),
    objectUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      controller.current?.abort();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [],
  );
  const open = async () => {
    const request = new AbortController();
    controller.current = request;
    setBusy(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/land/{land_id}/documents/{document_id}/pages/{page}/image", {
          params: { path: { land_id: landId, document_id: documentId, page } },
          parseAs: "blob",
          signal: request.signal,
        }),
      );
      if (!request.signal.aborted) {
        const next = URL.createObjectURL(blob);
        objectUrl.current = next;
        setUrl(next);
      }
    } catch (cause) {
      if (!request.signal.aborted) setError(describeError(cause));
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  return (
    <div className="land-document-image">
      {url ? (
        <figure>
          <label className="land-name">
            Page zoom · {Math.round(zoom * 100)}%
            <input
              type="range"
              aria-label="Page zoom"
              min={1}
              max={4}
              step={0.25}
              value={zoom}
              onChange={(event) => setZoom(Number(event.target.value))}
            />
          </label>
          <div
            className="land-page-viewport"
            tabIndex={0}
            role="region"
            aria-label="Original page image"
          >
            <img
              src={url}
              alt={`Original document page ${page}; compare with the extracted text below`}
              style={{ width: `${zoom * 100}%`, maxWidth: "none" }}
            />
          </div>
          <figcaption>
            Original page {page}, rendered for inspection. Download the original for full detail.
          </figcaption>
        </figure>
      ) : (
        <div className="land-actions">
          <button type="button" disabled={busy} onClick={() => void open()}>
            {busy ? "Rendering original page…" : "View original page"}
          </button>
        </div>
      )}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
