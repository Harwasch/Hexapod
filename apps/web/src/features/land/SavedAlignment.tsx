import { useEffect, useRef, useState } from "react";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { useLandContext } from "@/state/landContext";

type Fit = components["schemas"]["ImageRegistrationResult"];
type Registration = components["schemas"]["ImageRegistrationRead"];
const distance = (n: number | null) =>
  n == null ? "Not determined" : `${n.toLocaleString(undefined, { maximumFractionDigits: 2 })} m`;
export function AlignmentFit({ result }: { result: Fit }) {
  return (
    <div className="land-alignment-fit">
      <dl>
        <dt>Fit RMS</dt>
        <dd>{distance(result.rmsErrorM)}</dd>
        <dt>Largest residual</dt>
        <dd>{distance(result.maximumErrorM)}</dd>
        <dt>Leave-one-out RMS</dt>
        <dd>{distance(result.leaveOneOutRmsM)}</dd>
        <dt>Image covered by point matches</dt>
        <dd>{(result.controlPointCoverage * 100).toFixed(1)}%</dd>
      </dl>
      <p className="land-footnote">
        Leave-one-out error tests each point using a fit made from the others. Small errors show
        consistent matches, not independent geographic accuracy.
      </p>
      <ul>
        {result.warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
      <details>
        <summary>Individual point errors</summary>
        <dl>
          {result.points.map((p) => (
            <div key={p.label}>
              <dt>{p.label}</dt>
              <dd>
                {distance(p.errorM)} fit · {distance(p.leaveOneOutErrorM)} left out
              </dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
export function SavedAlignment({
  row,
  attribution,
  onRefine,
  refiningDisabled,
}: {
  row: Registration;
  attribution: string;
  onRefine: () => void;
  refiningDisabled: boolean;
}) {
  const scene = useScene();
  const shown = useLandContext((s) => s.rasters[row.id]);
  const tileError = useLandContext((s) => s.rasterErrors[row.id]);
  const [opacity, setOpacity] = useState(0.6),
    [error, setError] = useState<string | null>(null),
    [busy, setBusy] = useState(false);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const show = (alpha = opacity) =>
    useLandContext.getState().setRaster({
      id: row.id,
      kind: "archive-alignment",
      band: 1,
      bounds: row.result.bounds,
      attribution,
      opacity: alpha,
    });
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
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob),
        anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `aligned-map-${row.id}.tif`;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      if (!controller.signal.aborted) setError(describeError(cause));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  };
  return (
    <article className="land-saved-alignment" aria-label={`Saved alignment: ${row.request.name}`}>
      <h6>{row.request.name}</h6>
      <p className="land-footnote">
        {new Date(row.createdAt).toLocaleString()} · {row.request.points.length} matches · fit RMS{" "}
        {distance(row.result.rmsErrorM)}
      </p>
      <div className="land-actions">
        <button
          type="button"
          onClick={() => (shown ? useLandContext.getState().removeRaster(row.id) : show())}
        >
          {shown ? "Hide aligned map" : "Show aligned map"}
        </button>
        <button
          type="button"
          onClick={() => {
            const [w = 0, s = 0, e = 0, n = 0] = row.result.bounds;
            scene?.camera.flyToRectangle(w, s, e, n);
          }}
        >
          Frame aligned map
        </button>
        <button type="button" disabled={refiningDisabled} onClick={onRefine}>
          Refine a copy
        </button>
        <button type="button" disabled={busy} onClick={() => void download()}>
          {busy ? "Preparing GeoTIFF…" : "Download aligned GeoTIFF"}
        </button>
      </div>
      <label>
        Aligned map opacity {Math.round((shown?.opacity ?? opacity) * 100)}%
        <input
          type="range"
          aria-label="Aligned map opacity"
          min={0}
          max={1}
          step={0.05}
          value={shown?.opacity ?? opacity}
          onChange={(e) => {
            const alpha = +e.target.value;
            setOpacity(alpha);
            if (shown) show(alpha);
          }}
        />
      </label>
      {tileError && <p role="alert">{tileError}</p>}
      {error && <p role="alert">{error}</p>}
      <details>
        <summary>Alignment accuracy and provenance</summary>
        <AlignmentFit result={row.result} />
        {row.request.notes && <p>{row.request.notes}</p>}
        <p className="land-footnote">
          Aligned display raster: {row.displayWidth} × {row.displayHeight} pixels. Original source
          pixels and point matches are preserved separately. Algorithm: {row.result.algorithm}.
        </p>
        <p className="land-footnote">
          Source image SHA-256: <code>{row.request.imageSha256}</code>
        </p>
        <p className="land-footnote">
          GeoTIFF SHA-256: <code>{row.sha256}</code>
        </p>
      </details>
    </article>
  );
}
