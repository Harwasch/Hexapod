import { useEffect, useRef, useState, type RefObject } from "react";
import type { components, Footprint } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLand } from "@/state/land";

type ImportResult = components["schemas"]["BoundaryImportRead"];

export function LandBoundaryImport({
  inputRef,
  frame,
}: {
  inputRef: RefObject<HTMLInputElement | null>;
  frame: (boundary: Footprint) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [crs, setCrs] = useState("");
  const [layer, setLayer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  const preview = async (source: File, repair = false, reset = false) => {
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    const session = useLand.getState().session;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", source);
      if (!reset && crs.trim())
        form.append("source_crs", `EPSG:${crs.replace(/^EPSG:/i, "").trim()}`);
      if (!reset && layer) form.append("layer", layer);
      form.append("repair", String(repair));
      const next = await unwrap(
        api.POST("/api/v1/land/import", {
          body: { file: source.name },
          bodySerializer: () => form,
          signal: request.signal,
        }),
      );
      if (request.signal.aborted || useLand.getState().session !== session) return;
      setResult(next);
      if (next.boundary) {
        useLand.getState().propose({
          name: source.name.replace(/\.[^.]+$/, ""),
          description: "",
          boundary: next.boundary,
          source: { method: "imported", label: source.name, meaning: "study-area" },
        });
        frame(next.boundary);
      }
    } catch (failure) {
      if (!request.signal.aborted) setError(describeError(failure));
    } finally {
      if (!request.signal.aborted) setBusy(false);
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        className="land-file-input"
        accept=".geojson,.json,.kml,.kmz,.zip,.gpkg"
        aria-label="Import land boundary"
        onChange={(event) => {
          const source = event.target.files?.[0];
          event.target.value = "";
          if (!source) return;
          if (source.size > 20 * 1024 * 1024) {
            useLand.getState().setError("Choose a boundary file smaller than 20 MB.");
            return;
          }
          setFile(source);
          setResult(null);
          setCrs("");
          setLayer("");
          void preview(source, false, true);
        }}
      />
      {file && (
        <section className="land-import-review" aria-label="Boundary import review">
          <strong>{file.name}</strong>
          {busy && <p role="status">Reading and checking the boundary…</p>}
          {error && <p role="alert">{error}</p>}
          {result?.warnings?.map((warning) => (
            <p className="land-notice" key={warning}>
              {warning}
            </p>
          ))}
          {result?.status === "ready" && (
            <p>
              Boundary ready to review on the map. Save it when the outline and exclusions look
              right.
            </p>
          )}
          {result?.status === "choose-layer" && (
            <label className="land-name">
              Boundary layer
              <select value={layer} onChange={(event) => setLayer(event.target.value)}>
                <option value="">Choose a layer</option>
                {result.layers?.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
          )}
          {(result?.status === "needs-crs" || error) && (
            <label className="land-name">
              Source coordinate system (EPSG code)
              <input
                value={crs}
                onChange={(event) => setCrs(event.target.value)}
                placeholder="For example, 32610"
                inputMode="numeric"
              />
            </label>
          )}
          <div className="land-actions">
            {result?.status !== "ready" && (
              <button
                type="button"
                disabled={
                  busy ||
                  (result?.status === "choose-layer" && !layer) ||
                  (result?.status === "needs-crs" && !crs.trim())
                }
                onClick={() => void preview(file, result?.status === "needs-repair")}
              >
                {result?.status === "needs-repair"
                  ? "Preview repaired boundary"
                  : "Preview boundary"}
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                controller.current?.abort();
                setFile(null);
                setBusy(false);
              }}
            >
              Dismiss import details
            </button>
          </div>
        </section>
      )}
    </>
  );
}
