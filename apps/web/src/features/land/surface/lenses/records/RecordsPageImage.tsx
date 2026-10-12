/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Keyboard users pan a zoomed page by focusing it. */
import { useEffect, useRef, useState } from "react";
import { Minus, Plus } from "lucide-react";

import { GlassButton } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";

const ZOOMS = [1, 1.5, 2, 3, 4];

/** The original page of a PDF, rendered on request, to compare with its extracted text. */
export function RecordsPageImage({
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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const objectUrl = useRef<string | null>(null);
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
      if (request.signal.aborted) return;
      const next = URL.createObjectURL(blob);
      objectUrl.current = next;
      setUrl(next);
    } catch (cause) {
      if (!request.signal.aborted)
        setError(`The page could not be rendered. ${describeError(cause)}`);
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };
  const index = ZOOMS.indexOf(zoom);
  if (!url)
    return (
      <div className="land-row">
        <GlassButton size="sm" variant="ghost" loading={busy} onClick={() => void open()}>
          {busy ? "Rendering the page…" : "View original page"}
        </GlassButton>
        {error && (
          <p className="land-alert" role="alert">
            {error}
          </p>
        )}
      </div>
    );
  return (
    <figure className="records-page-image">
      <div className="records-align__zoom">
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Zoom out of the page"
          disabled={index <= 0}
          onClick={() => setZoom(ZOOMS[Math.max(0, index - 1)] ?? 1)}
        >
          <Minus size={14} />
        </GlassButton>
        <span className="land-muted">Page zoom {Math.round(zoom * 100)}%</span>
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Zoom into the page"
          disabled={index >= ZOOMS.length - 1}
          onClick={() => setZoom(ZOOMS[Math.min(ZOOMS.length - 1, index + 1)] ?? 1)}
        >
          <Plus size={14} />
        </GlassButton>
      </div>
      <div
        className="records-page-image__viewport"
        tabIndex={0}
        role="region"
        aria-label="Original page image"
      >
        <img
          src={url}
          alt={`Original page ${page}`}
          style={{ width: `${zoom * 100}%`, maxWidth: "none" }}
        />
      </div>
      <figcaption className="land-muted">
        Page {page} as rendered for reading. Download the original for full detail.
      </figcaption>
    </figure>
  );
}
