import { useEffect, useRef, useState } from "react";

import type { components } from "@twin/contracts";

import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";

import { ALIGNMENT_PAGE, useAlignments } from "./recordsApi";
import { formatDate, saveBlob } from "./recordsModel";
import { useRecords } from "./recordsStore";

type Fit = components["schemas"]["ImageRegistrationResult"];
type Registration = components["schemas"]["ImageRegistrationRead"];

const metres = (value: number | null) =>
  value == null
    ? "not known"
    : `${value.toLocaleString(undefined, { maximumFractionDigits: value < 10 ? 1 : 0 })} m`;

/** How well the matched points agree, in metres on the ground. */
export function AlignmentFit({ result }: { result: Fit }) {
  return (
    <div className="records-fit">
      <dl className="records-fit__numbers">
        <div>
          <dt>Typical error</dt>
          <dd>{metres(result.rmsErrorM)}</dd>
        </div>
        <div>
          <dt>Worst point</dt>
          <dd>{metres(result.maximumErrorM)}</dd>
        </div>
        <div>
          <dt>Each point left out</dt>
          <dd>{metres(result.leaveOneOutRmsM)}</dd>
        </div>
        <div>
          <dt>Map covered by matches</dt>
          <dd>{Math.round(result.controlPointCoverage * 100)}%</dd>
        </div>
      </dl>
      {result.warnings.map((warning) => (
        <p key={warning} className="land-muted">
          {warning}
        </p>
      ))}
      <details className="records-more">
        <summary>Each point</summary>
        <ul className="records-plain-list">
          {result.points.map((point) => (
            <li key={point.label}>
              {point.label}: {metres(point.errorM)} off, {metres(point.leaveOneOutErrorM)} when left
              out
            </li>
          ))}
        </ul>
        <p className="land-muted">
          Leaving each point out tests it against a fit made from the others. Small errors mean the
          matches agree with each other, not that the map is surveyed accurately.
        </p>
      </details>
    </div>
  );
}

function SavedAlignment({
  row,
  label,
  attribution,
  onRefine,
  refiningDisabled,
}: {
  row: Registration;
  label: string;
  attribution: string;
  onRefine: (row: Registration) => void;
  refiningDisabled?: boolean;
}) {
  const scene = useScene();
  const shown = useLandContext((s) => Boolean(s.rasters[row.id]));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const download = async () => {
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    try {
      const blob = await unwrap(
        api.GET("/api/v1/research/image-registrations/{registration_id}/download", {
          params: { path: { registration_id: row.id } },
          parseAs: "blob",
          signal: controller.signal,
        }),
      );
      if (!controller.signal.aborted) saveBlob(blob, `aligned-map-${row.id}.tif`);
    } catch (cause) {
      if (!controller.signal.aborted)
        setError(`The aligned map could not be downloaded. ${describeError(cause)}`);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  };
  return (
    <article className="records-alignment" aria-label={`Saved alignment: ${row.request.name}`}>
      <div className="records-alignment__head">
        <strong>{row.request.name}</strong>
        <span className="land-muted">
          {formatDate(row.createdAt)} · {row.request.points.length} matches · typical error{" "}
          {metres(row.result.rmsErrorM)}
        </span>
      </div>
      <div className="land-row">
        <button
          type="button"
          className="land-link"
          aria-pressed={shown}
          onClick={() =>
            shown
              ? useRecords.getState().hideOverlay()
              : useRecords
                  .getState()
                  .showOverlay({ id: row.id, label, bounds: row.result.bounds, attribution })
          }
        >
          {shown ? "Hide aligned map" : "Show on the land"}
        </button>
        <button
          type="button"
          className="land-link"
          disabled={!scene}
          onClick={() => {
            const [west = 0, south = 0, east = 0, north = 0] = row.result.bounds;
            scene?.camera.flyToRectangle(west, south, east, north);
          }}
        >
          Frame it
        </button>
        <button
          type="button"
          className="land-link"
          disabled={refiningDisabled}
          onClick={() => onRefine(row)}
        >
          Refine a copy
        </button>
        <button type="button" className="land-link" disabled={busy} onClick={() => void download()}>
          {busy ? "Preparing GeoTIFF…" : "Download GeoTIFF"}
        </button>
      </div>
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      <details className="records-more">
        <summary>Accuracy and provenance</summary>
        <AlignmentFit result={row.result} />
        {row.request.notes && <p>{row.request.notes}</p>}
        <p className="land-muted">
          Display copy {row.displayWidth} × {row.displayHeight} pixels, method{" "}
          {row.result.algorithm}. The original image and point matches are kept separately.
        </p>
        <p className="land-muted records-hash">Source image SHA-256 {row.request.imageSha256}</p>
        <p className="land-muted records-hash">GeoTIFF SHA-256 {row.sha256}</p>
      </details>
    </article>
  );
}

/** The alignments saved for one historical map, newest first, twenty at a time. */
export function RecordsSavedAlignments({
  evidenceId,
  label,
  attribution,
  onRefine,
  refiningDisabled,
  hideWhenEmpty,
}: {
  evidenceId: string;
  label: string;
  attribution: string;
  onRefine: (row: Registration) => void;
  refiningDisabled?: boolean;
  hideWhenEmpty?: boolean;
}) {
  const [offset, setOffset] = useState(0);
  const query = useAlignments(evidenceId, offset);
  if (hideWhenEmpty && !offset && !query.data?.length && !query.isError) return null;
  return (
    <section className="records-section" aria-label="Saved alignments">
      <h5 className="records-subhead">Lined up before</h5>
      {query.isPending && (
        <p className="land-muted" role="status">
          Loading saved alignments…
        </p>
      )}
      {query.isError && (
        <p className="land-alert" role="alert">
          Saved alignments did not load.{" "}
          <button type="button" className="land-link" onClick={() => void query.refetch()}>
            Try again
          </button>
        </p>
      )}
      {query.data?.length === 0 && <p className="land-muted">None saved yet.</p>}
      {query.data?.map((row) => (
        <SavedAlignment
          key={row.id}
          row={row}
          label={label}
          attribution={attribution}
          onRefine={onRefine}
          refiningDisabled={refiningDisabled}
        />
      ))}
      {(offset > 0 || query.data?.length === ALIGNMENT_PAGE) && (
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            disabled={!offset || query.isFetching}
            onClick={() => setOffset(Math.max(0, offset - ALIGNMENT_PAGE))}
          >
            Newer alignments
          </button>
          <button
            type="button"
            className="land-link"
            disabled={query.data?.length !== ALIGNMENT_PAGE || query.isFetching}
            onClick={() => setOffset(offset + ALIGNMENT_PAGE)}
          >
            Older alignments
          </button>
        </div>
      )}
    </section>
  );
}
