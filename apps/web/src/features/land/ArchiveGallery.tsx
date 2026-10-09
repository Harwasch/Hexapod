import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { LandEvidence } from "@twin/contracts";
import { boundsOf } from "@twin/geo";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { useLandContext } from "@/state/landContext";
import { useLandAccessReady, useLandScope } from "@/state/landIdentity";
import { useArchiveImage } from "./useArchiveImage";

function ArchiveCard({
  id,
  cached,
  onAsk,
}: {
  id: string;
  cached?: LandEvidence;
  onAsk?: () => void;
}) {
  const scope = useLandScope();
  const ready = useLandAccessReady();
  const scene = useScene();
  const [expanded, setExpanded] = useState(false);
  const [failedImage, setFailedImage] = useState<string | null>(null);
  const layerId = `archive:${id}`;
  const shown = useLandContext((state) => Boolean(state.layers[layerId]));
  const query = useQuery({
    queryKey: ["land-research", scope, "evidence", id],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/research/evidence/{evidence_id}", {
          params: { path: { evidence_id: id } },
        }),
      ),
    enabled: ready && !cached,
    retry: false,
  });
  const evidence = cached ?? query.data;
  const media = evidence?.media;
  const archived = useArchiveImage(id, Boolean(media));
  if (!evidence)
    return query.isError ? (
      <p className="land-error" role="alert">
        This source could not be loaded.{" "}
        <button type="button" onClick={() => void query.refetch()}>
          Retry source
        </button>
      </p>
    ) : (
      <p role="status">Loading archive source…</p>
    );
  if (!media) return <p className="land-notice">This source has no reusable image preview.</p>;
  const imageUrl = archived.url ?? media.previewUrl;
  const imageFailed = failedImage === imageUrl;
  const toggleLocation = () => {
    if (shown) {
      useLandContext.getState().removeLayer(layerId);
      return;
    }
    const geometry = media.location;
    if (!geometry) return;
    useLandContext.getState().setLayer({
      id: layerId,
      title: media.title,
      features: [{ id, label: `${media.title} · ${media.locationMeaning}`, geometry }],
    });
    if (geometry.type === "Point") {
      const [x = 0, y = 0] = geometry.coordinates;
      scene?.camera.flyToRectangle(x - 0.002, y - 0.002, x + 0.002, y + 0.002);
    } else if (geometry.type !== "LineString") {
      const b = boundsOf(geometry);
      scene?.camera.flyToRectangle(b.west, b.south, b.east, b.north);
    }
  };
  return (
    <article className="land-archive-card" aria-label={media.title}>
      <h5>{media.title}</h5>
      <p className="land-archive-date">{media.sourceDate ?? "Date not specified in catalog"}</p>
      <figure>
        {imageFailed ? (
          <p className="land-notice">
            {archived.url
              ? "The saved preview could not be displayed."
              : "The remote preview is unavailable."}{" "}
            You can still open the original source below.
          </p>
        ) : (
          <img
            src={imageUrl}
            alt={media.title}
            loading="lazy"
            decoding="async"
            referrerPolicy="no-referrer"
            className={expanded ? "is-expanded" : ""}
            onError={() => setFailedImage(imageUrl)}
          />
        )}
        <figcaption>
          {media.creator} ·{" "}
          <a href={media.licenseUrl} target="_blank" rel="noopener noreferrer">
            {media.license}
          </a>
        </figcaption>
      </figure>
      <div className="land-actions">
        {!imageFailed && (
          <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
            {expanded ? "Reduce preview" : "Enlarge preview"}
          </button>
        )}
        <a href={media.sourceUrl} target="_blank" rel="noopener noreferrer">
          Open original source
        </a>
        {media.downloadUrl && (
          <a href={media.downloadUrl} target="_blank" rel="noopener noreferrer">
            Open map file
          </a>
        )}
      </div>
      <div className="land-archive-storage">
        {archived.metadata ? (
          <>
            <p className="land-footnote">
              Image saved {new Date(archived.metadata.createdAt).toLocaleDateString()} ·{" "}
              {archived.metadata.width} × {archived.metadata.height} pixels.
            </p>
            <div className="land-actions">
              <button
                type="button"
                disabled={archived.busy}
                onClick={() => void archived.download()}
              >
                {archived.busy ? "Preparing image…" : "Download saved source image"}
              </button>
            </div>
            <details>
              <summary>Saved image provenance</summary>
              <p className="land-footnote">
                The downloaded archive preview is preserved byte for byte. The display copy applies
                its orientation and removes embedded metadata. This is a preview snapshot, not the
                full-resolution archive master.
              </p>
              <p className="land-footnote">
                Source SHA-256: <code>{archived.metadata.sourceSha256}</code>
              </p>
              <p className="land-footnote">
                Display SHA-256: <code>{archived.metadata.sha256}</code>
              </p>
            </details>
          </>
        ) : (
          <div className="land-actions">
            <button
              type="button"
              disabled={archived.busy || archived.statusPending || !archived.canEdit}
              onClick={() => void archived.capture()}
            >
              {archived.busy ? "Saving image…" : "Save image to workspace"}
            </button>
          </div>
        )}
        {(archived.error !== null || archived.statusError) && (
          <p role="alert" className="land-error">
            {archived.error ?? "Saved image status could not be loaded."}{" "}
            <button type="button" onClick={archived.retry}>
              Retry saved image
            </button>
          </p>
        )}
      </div>
      <p>{media.relevance}</p>
      <div className="land-actions">
        {media.location && (
          <button type="button" disabled={!scene} aria-pressed={shown} onClick={toggleLocation}>
            {shown
              ? "Hide catalog location"
              : media.locationMeaning === "catalog-footprint"
                ? "Show sheet footprint"
                : "Show catalog location"}
          </button>
        )}
        <button
          type="button"
          onClick={() => {
            const state = useLandContext.getState();
            state.setResearchQuestion(
              `${state.researchQuestion ? state.researchQuestion + "\n\n" : ""}Investigate this archive source: ${media.title} (evidence ${id}). What can its catalog metadata establish about this land, and what needs verification?`,
            );
            onAsk?.();
          }}
        >
          Ask about this source
        </button>
      </div>
      <details>
        <summary>Dates, attribution and interpretation</summary>
        <p>{media.dateMeaning}</p>
        {media.description && <p>{media.description}</p>}
        <p className="land-footnote">
          Metadata saved {new Date(evidence.retrievedAt).toLocaleString()}.{" "}
          {archived.url
            ? "You are viewing the saved image snapshot."
            : "This preview is served by the public archive and may change."}
        </p>
        {media.sourceVersion && <p className="land-footnote">{media.sourceVersion}</p>}
        <p className="land-footnote">{evidence.attribution}</p>
      </details>
    </article>
  );
}

export function ArchiveGallery({
  ids,
  evidence = [],
  onAsk,
}: {
  ids: string[];
  evidence?: LandEvidence[];
  onAsk?: () => void;
}) {
  const [index, setIndex] = useState(0);
  const id = ids[index] ?? ids[0];
  if (!id) return null;
  return (
    <section className="land-archive-gallery" aria-label="Archive gallery">
      <nav className="land-archive-navigation" aria-label="Archive sources">
        <button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>
          Previous source
        </button>
        <span aria-live="polite">
          {index + 1} of {ids.length}
        </span>
        <button
          type="button"
          disabled={index >= ids.length - 1}
          onClick={() => setIndex(index + 1)}
        >
          Next source
        </button>
      </nav>
      <ArchiveCard
        key={id}
        id={id}
        cached={evidence.find((item) => item.id === id)}
        onAsk={onAsk}
      />
    </section>
  );
}
