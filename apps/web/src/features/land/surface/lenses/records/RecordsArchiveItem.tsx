import { useRef, useState, type ReactNode } from "react";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  MapPin,
  MessageCircleQuestion,
} from "lucide-react";

import type { LandEvidence } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import type { LandPoint } from "@/state/land";
import { useLandScope } from "@/state/landIdentity";

import { askAgent } from "../../agentBus";
import { anchorOf } from "../../geometryAnchor";
import { useFlyTo, useLeader } from "../../mapHooks";
import { seedAlignmentDraft } from "./alignmentDraft";
import { formatDate, toArchiveItem } from "./recordsModel";
import { RecordsSavedAlignments } from "./RecordsSavedAlignments";
import { RecordsThenNow } from "./RecordsThenNow";
import { useArchiveImage } from "./useArchiveImage";

export interface ArchiveNavigation {
  index: number;
  total: number;
  onPrevious: () => void;
  onNext: () => void;
}

/**
 * One old photo or map sheet: the picture, who made it and its licence, where it is on the
 * map, and for a map sheet, lining it up with the land.
 */
export function RecordsArchiveItem({
  evidence,
  navigation,
  onAlign,
  onAsk,
  onMap = true,
  back,
  anchor,
}: {
  evidence: LandEvidence;
  /** A way back, shown in the bar with the previous and next buttons. */
  back?: ReactNode;
  /** Where the line from the card ends: the item's pin, else a point inside its shape. */
  anchor?: LandPoint;
  navigation?: ArchiveNavigation;
  /** Opens the alignment tool for a map sheet; absent where the tool cannot run. */
  onAlign?: () => void;
  /** Called after a question is handed to the agent. */
  onAsk?: () => void;
  /**
   * In the Records lens: ties the card to the place with a line and shows the saved
   * alignments. Elsewhere (a research gallery) only the picture and its sources show.
   */
  onMap?: boolean;
}) {
  const scope = useLandScope();
  const item = toArchiveItem(evidence);
  const archived = useArchiveImage(evidence.id, Boolean(item));
  const [failedImage, setFailedImage] = useState<string | null>(null);
  const [large, setLarge] = useState(false);
  const fly = useFlyTo();
  const head = useRef<HTMLDivElement>(null);
  const end = anchor ?? (item?.geometry ? anchorOf(item.geometry) : null);
  useLeader(head, onMap ? end : null, "warn", true);
  if (!item) return <p className="land-muted">This source has no picture to show.</p>;
  const { media } = item;
  const isMap = media.kind === "historical-map";
  const imageUrl = archived.url ?? media.previewUrl;
  const imageFailed = failedImage === imageUrl;
  const label = `${item.year ? `${item.year} · ` : ""}${item.title}`;
  const ask = (question: string) => {
    askAgent(question);
    onAsk?.();
  };
  const align = async () => {
    if (!onAlign) return;
    if (!archived.metadata) {
      if (!archived.canEdit) return;
      if (!(await archived.capture())) return;
    }
    onAlign();
  };

  return (
    <article className="records-item" aria-label={item.title}>
      {(back ?? (navigation && navigation.total > 1)) && (
        <div className="records-item__bar">
          {back}
          {navigation && navigation.total > 1 && (
            <nav className="records-item__nav" aria-label="Old maps and photos">
              <GlassButton
                iconOnly
                size="sm"
                variant="ghost"
                aria-label="Previous source"
                disabled={navigation.index === 0}
                onClick={navigation.onPrevious}
              >
                <ChevronLeft size={16} />
              </GlassButton>
              <span aria-live="polite">
                {navigation.index + 1} of {navigation.total}
              </span>
              <GlassButton
                iconOnly
                size="sm"
                variant="ghost"
                aria-label="Next source"
                disabled={navigation.index >= navigation.total - 1}
                onClick={navigation.onNext}
              >
                <ChevronRight size={16} />
              </GlassButton>
            </nav>
          )}
        </div>
      )}
      <figure className={`records-item__figure${large ? " is-large" : ""}`}>
        {imageFailed ? (
          <p className="land-note">
            {archived.url
              ? "The saved copy could not be shown."
              : "The archive preview is not available."}{" "}
            The original is still linked below.
          </p>
        ) : (
          <button
            type="button"
            className="records-item__picture"
            aria-label={large ? "Reduce preview" : "Enlarge preview"}
            aria-expanded={large}
            onClick={() => setLarge(!large)}
          >
            <img
              src={imageUrl}
              alt={item.title}
              loading="lazy"
              decoding="async"
              referrerPolicy="no-referrer"
              onError={() => setFailedImage(imageUrl)}
            />
          </button>
        )}
        <figcaption>
          {media.creator} ·{" "}
          <a href={media.licenseUrl} target="_blank" rel="noopener noreferrer">
            {media.license}
          </a>
        </figcaption>
      </figure>
      <div ref={head} className="records-item__head">
        <h4 className="records-item__title">{item.title}</h4>
        <p className="land-muted">
          {isMap ? "Map sheet" : "Photo"} · {media.sourceDate ?? "Date not given"} ·{" "}
          {isMap ? "covers the land" : item.onLand ? "taken on the land" : "taken nearby"}
        </p>
      </div>
      <div className="land-row">
        {item.geometry && (
          <GlassButton
            size="sm"
            variant="ghost"
            leadingIcon={<MapPin size={14} />}
            onClick={() => item.geometry && fly(item.geometry)}
          >
            {isMap ? "Show sheet" : "Show where"}
          </GlassButton>
        )}
        <GlassButton
          size="sm"
          variant="ghost"
          leadingIcon={<MessageCircleQuestion size={14} />}
          onClick={() =>
            ask(
              `What can "${media.title}" (archive source ${evidence.id}) tell us about this land? Separate what the catalog says from what still needs checking.`,
            )
          }
        >
          Ask about this
        </GlassButton>
        <a className="records-out" href={media.sourceUrl} target="_blank" rel="noopener noreferrer">
          Open original source <ArrowUpRight size={13} aria-hidden="true" />
        </a>
        {media.downloadUrl && (
          <a
            className="records-out"
            href={media.downloadUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            Open map file <ArrowUpRight size={13} aria-hidden="true" />
          </a>
        )}
      </div>
      {isMap && onAlign && (
        <div className="records-item__align">
          <GlassButton
            variant="primary"
            size="sm"
            loading={archived.busy && !archived.metadata}
            disabled={
              archived.busy || archived.statusPending || (!archived.metadata && !archived.canEdit)
            }
            onClick={() => void align()}
          >
            Line it up with the land
          </GlassButton>
          <span className="land-muted">
            {archived.metadata || archived.canEdit
              ? "Match a few spots to lay it over today's map."
              : "An editor can save a copy to line it up."}
          </span>
        </div>
      )}
      {isMap && onAlign && onMap && <RecordsThenNow />}
      {isMap && onAlign && onMap && (
        <RecordsSavedAlignments
          evidenceId={evidence.id}
          label={label}
          attribution={evidence.attribution}
          hideWhenEmpty
          onRefine={(row) => {
            seedAlignmentDraft(scope, evidence.id, row.request);
            onAlign();
          }}
        />
      )}
      {(archived.error !== null || archived.statusError) && (
        <p className="land-alert" role="alert">
          {archived.error ?? "Whether a copy is saved could not be checked."}{" "}
          <button type="button" className="land-link" onClick={archived.retry}>
            Try again
          </button>
        </p>
      )}
      <details className="records-more">
        <summary>Dates, licence and saved copy</summary>
        {item.title !== media.title && <p className="land-muted">Catalog title: {media.title}</p>}
        <p>{media.dateMeaning}</p>
        {media.description && <p>{media.description}</p>}
        <p>{media.relevance}</p>
        <p className="land-muted">
          Found {formatDate(evidence.retrievedAt)}.{" "}
          {archived.url
            ? "You are viewing the copy saved in this workspace."
            : "This preview comes from the public archive and may change."}
        </p>
        {media.sourceVersion && <p className="land-muted">{media.sourceVersion}</p>}
        <p className="land-muted">{evidence.attribution}</p>
        {archived.metadata ? (
          <>
            <p className="land-muted">
              Copy saved {formatDate(archived.metadata.createdAt)} · {archived.metadata.width} ×{" "}
              {archived.metadata.height} pixels.
            </p>
            <div className="land-row">
              <button
                type="button"
                className="land-link"
                disabled={archived.busy}
                onClick={() => void archived.download()}
              >
                {archived.busy ? "Preparing image…" : "Download saved copy"}
              </button>
            </div>
            <details className="records-more">
              <summary>Saved copy fingerprints</summary>
              <p className="land-muted">
                The archive preview is kept byte for byte. The display copy is turned upright and
                has its embedded metadata removed. Neither is the full-resolution master.
              </p>
              <p className="land-muted records-hash">
                Source SHA-256 {archived.metadata.sourceSha256}
              </p>
              <p className="land-muted records-hash">Display SHA-256 {archived.metadata.sha256}</p>
            </details>
          </>
        ) : archived.canEdit ? (
          <div className="land-row">
            <button
              type="button"
              className="land-link"
              disabled={archived.busy || archived.statusPending}
              onClick={() => void archived.capture()}
            >
              {archived.busy ? "Saving a copy…" : "Save a copy to this workspace"}
            </button>
          </div>
        ) : null}
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            onClick={() =>
              ask(
                `Look closely at the archive image "${media.title}" (archive source ${evidence.id}). Use the image pixels, distinguish visible details from catalog claims, and say what its resolution cannot show about this land.`,
              )
            }
          >
            Look closely at the image
          </button>
        </div>
      </details>
    </article>
  );
}
