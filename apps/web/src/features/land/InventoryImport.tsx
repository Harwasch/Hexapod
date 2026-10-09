import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components, LandArea } from "@twin/contracts";
import { api, ApiError, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useLandContext } from "@/state/landContext";
import { useUi } from "@/state/ui";
import {
  defaultImportMapping,
  inventoryCategories,
  parseInventoryImport,
  type ImportCategory,
  type ImportMapping,
} from "./inventoryImport";
import "./inventoryImport.css";
type Batch = components["schemas"]["FeatureBatchRequest"];
type Preview = components["schemas"]["FeatureBatchPreview"];
type Receipt = components["schemas"]["FeatureBatchRead"];
interface Draft {
  version: 1;
  text: string;
  format: "csv" | "geojson";
  fileName: string;
  sha256: string;
  namespace: string;
  mapping: ImportMapping;
  excluded: string[];
  names: Record<string, string>;
  categories: Record<string, ImportCategory>;
  requestKey: string;
  boundaryRevision: number;
}
export function InventoryImport({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <Import key={`${scope}:${land.id}`} land={land} scope={scope} />;
}
function Import({ land, scope }: { land: LandArea; scope: string }) {
  const canEdit = useLandCanEdit(),
    ready = useLandAccessReady(),
    cache = useQueryClient();
  const storage = `living-world-land-draft:${encodeURIComponent(scope)}:inventory-import:${land.id}`;
  const [draft, setDraft] = useState<Draft | null>(null),
    [error, setError] = useState<string | null>(null),
    [busy, setBusy] = useState(false);
  const [recovery, setRecovery] = useState<string | null>(() => {
    try {
      return localStorage.getItem(storage);
    } catch {
      return null;
    }
  });
  const [storageError, setStorageError] = useState(false),
    [offset, setOffset] = useState(0),
    [historyOffset, setHistoryOffset] = useState(0);
  const [review, setReview] = useState<{ draft: Draft; value: Preview } | null>(null),
    [receipt, setReceipt] = useState<Receipt | null>(null);
  const ticket = useRef(0);
  const section = useLandContext((s) => s.section),
    panel = useUi((s) => s.activePanel);
  const importText = draft?.text,
    importFormat = draft?.format,
    importMapping = draft?.mapping;
  const parsed = useMemo(() => {
    if (importText === undefined || !importFormat || !importMapping)
      return { rows: [], error: null };
    try {
      return { rows: parseInventoryImport(importText, importFormat, importMapping), error: null };
    } catch (cause) {
      return { rows: [], error: describeError(cause) };
    }
  }, [importText, importFormat, importMapping]);
  const selected = parsed.rows.filter(
    (row) => !row.error && row.geometry && !draft?.excluded.includes(row.rowId),
  );
  const preview =
    draft && review?.draft === draft && draft.boundaryRevision === land.revision
      ? review.value
      : null;
  const history = useQuery({
    queryKey: ["land-inventory-imports", scope, land.id, historyOffset],
    enabled: ready,
    retry: false,
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/land/{land_id}/features/imports", {
          params: { path: { land_id: land.id }, query: { limit: 20, offset: historyOffset } },
        }),
      ),
  });
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );
  useEffect(() => {
    if (!preview || section !== "inventory" || panel !== "land") return;
    useLandContext.getState().setLayer({
      id: "inventory-import",
      title: "Reviewed import candidates",
      features: preview.rows.flatMap((row) =>
        row.geometryPreview
          ? [{ id: row.rowId, label: row.name, geometry: row.geometryPreview.geometry }]
          : [],
      ),
    });
    return () => useLandContext.getState().removeLayer("inventory-import");
  }, [preview, section, panel]);
  function persist(next: Draft | null) {
    setDraft(next);
    setReview(null);
    setError(null);
    setStorageError(false);
    try {
      if (next) localStorage.setItem(storage, JSON.stringify(next));
      else localStorage.removeItem(storage);
    } catch {
      setStorageError(true);
    }
  }
  function update(change: Partial<Draft>) {
    if (!draft) return;
    ticket.current++;
    setBusy(false);
    persist({ ...draft, ...change, requestKey: crypto.randomUUID() });
  }
  async function choose(file: File | undefined) {
    if (!file) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      if (file.size > 5 * 1024 * 1024)
        throw new Error("Choose a GeoJSON or CSV file no larger than 5 MiB.");
      const buffer = await file.arrayBuffer(),
        text = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
      const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      if (current !== ticket.current) return;
      persist({
        version: 1,
        text,
        sha256,
        fileName: file.name.slice(0, 200),
        format: file.name.toLowerCase().endsWith(".csv") ? "csv" : "geojson",
        mapping: { ...defaultImportMapping },
        namespace: `file-sha256:${sha256}`,
        excluded: [],
        names: {},
        categories: {},
        requestKey: crypto.randomUUID(),
        boundaryRevision: land.revision,
      });
      setOffset(0);
      setReceipt(null);
      setRecovery(null);
    } catch (cause) {
      if (current === ticket.current) setError(describeError(cause));
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  }
  function restore() {
    try {
      const value = JSON.parse(recovery ?? "null") as Draft | null;
      if (
        value?.version !== 1 ||
        typeof value.text !== "string" ||
        value.text.length > 5 * 1024 * 1024 ||
        !["csv", "geojson"].includes(value.format) ||
        typeof value.fileName !== "string" ||
        !/^[a-f0-9]{64}$/.test(value.sha256) ||
        !/^[a-f0-9-]{36}$/.test(value.requestKey) ||
        typeof value.namespace !== "string" ||
        !Number.isInteger(value.boundaryRevision) ||
        !value.mapping ||
        Object.values(defaultImportMapping).some(
          (key) => typeof value.mapping[key as keyof ImportMapping] !== "string",
        ) ||
        !Array.isArray(value.excluded) ||
        value.excluded.some((id) => typeof id !== "string") ||
        !value.names ||
        Object.values(value.names).some((name) => typeof name !== "string") ||
        !value.categories ||
        Object.values(value.categories).some((category) => !inventoryCategories.includes(category))
      )
        throw new Error(
          "This saved import draft is not readable. Discard it and choose the original file again.",
        );
      persist(value);
      setRecovery(null);
    } catch (cause) {
      setError(describeError(cause));
    }
  }
  function request(): Batch {
    if (!draft || !selected.length) throw new Error("Select at least one valid feature.");
    if (
      draft.namespace !== `file-sha256:${draft.sha256}` &&
      selected.some((row) => row.generatedId)
    )
      throw new Error(
        "Choose a record-ID column for every selected row before using a shared dataset namespace.",
      );
    return {
      requestKey: draft.requestKey,
      sourceLabel: draft.fileName,
      sourceFileSha256: draft.sha256,
      boundaryRevision: draft.boundaryRevision,
      skipDuplicates: true,
      rows: selected.flatMap((row) =>
        row.geometry
          ? [
              {
                rowId: row.rowId,
                feature: {
                  requestKey: draft.requestKey,
                  name: draft.names[row.rowId] ?? row.name,
                  category: draft.categories[row.rowId] ?? row.category,
                  geometry: row.geometry,
                  status: "candidate" as const,
                  source: {
                    method: "imported" as const,
                    label: draft.fileName,
                    recordId: row.recordId,
                    meaning: "physical-feature" as const,
                  },
                  externalRef: { namespace: draft.namespace, recordId: row.recordId },
                  attributes: row.attributes,
                },
              },
            ]
          : [],
      ),
    };
  }
  async function run(save: boolean) {
    if (!draft || (save && (!preview || !canEdit))) return;
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const body = request();
      if (save) {
        const value = await unwrap(
          api.POST("/api/v1/land/{land_id}/features/imports", {
            params: { path: { land_id: land.id } },
            body,
          }),
        );
        if (current === ticket.current) {
          setReceipt(value);
          persist(null);
        }
        await cache.invalidateQueries({ queryKey: ["land-inventory", scope, land.id] });
        await cache.invalidateQueries({ queryKey: ["land-inventory-imports", scope, land.id] });
      } else {
        const value = await unwrap(
          api.POST("/api/v1/land/{land_id}/features/imports/preview", {
            params: { path: { land_id: land.id } },
            body,
          }),
        );
        if (current === ticket.current) setReview({ draft, value });
      }
    } catch (cause) {
      if (current === ticket.current)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      if (current === ticket.current) setBusy(false);
    }
  }
  return (
    <section className="inventory-import" aria-label="Import inventory">
      <details open={!!draft || !!recovery || !!receipt}>
        <summary>Import an asset inventory</summary>
        <p>
          Review GeoJSON features or CSV points, then add them as candidates. Existing records are
          kept; imports never overwrite their history.
        </p>
        {error && (
          <p role="alert" className="land-error">
            {error}
          </p>
        )}
        {storageError && (
          <p role="alert">
            This browser could not store the import draft for recovery. Keep this page open.
          </p>
        )}
        {recovery && canEdit && (
          <div className="land-actions">
            <p>An unfinished import is available in this browser.</p>
            <button type="button" onClick={restore}>
              Recover inventory import
            </button>
            <button
              type="button"
              onClick={() => {
                persist(null);
                setRecovery(null);
              }}
            >
              Discard saved import
            </button>
          </div>
        )}
        {canEdit && !draft && !recovery && (
          <label className="land-name">
            Inventory file
            <input
              type="file"
              accept=".geojson,.json,.csv,application/geo+json,text/csv"
              disabled={busy}
              onChange={(event) => {
                void choose(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </label>
        )}
        {draft && canEdit && (
          <fieldset disabled={busy} className="inventory-import-controls">
            <p>
              {draft.fileName} · {parsed.rows.length} rows · {selected.length} selected
            </p>
            <p className="land-footnote">
              WGS 84 longitude/latitude only. Up to 200 assets, 100,000 vertices and 5 MiB per file.
              The receipt retains the original file's fingerprint; the original file itself is not
              uploaded.
            </p>
            <details>
              <summary>Column mapping and dataset identity</summary>
              {(
                [
                  "name",
                  "id",
                  "category",
                  ...(draft.format === "csv" ? (["longitude", "latitude"] as const) : []),
                ] as (keyof ImportMapping)[]
              ).map((key) => (
                <label key={key} className="land-name">
                  {key} column
                  <input
                    aria-label={`Import ${key} column`}
                    maxLength={100}
                    value={draft.mapping[key]}
                    onChange={(event) => {
                      update({
                        mapping: { ...draft.mapping, [key]: event.target.value },
                        names: {},
                        categories: {},
                        excluded: [],
                      });
                      setOffset(0);
                    }}
                  />
                </label>
              ))}
              <label className="land-name">
                Dataset namespace
                <input
                  aria-label="Dataset namespace"
                  maxLength={150}
                  value={draft.namespace}
                  onChange={(event) => update({ namespace: event.target.value })}
                />
              </label>
              <p className="land-footnote">
                The file fingerprint identifies repeated uploads by default. Use a stable dataset
                name and record-ID column to recognize assets across different files. Rows without
                record IDs use their file row numbers.
              </p>
            </details>
            {draft.boundaryRevision !== land.revision && (
              <p role="alert">
                The boundary changed since this draft.{" "}
                <button type="button" onClick={() => update({ boundaryRevision: land.revision })}>
                  Use current boundary and review again
                </button>
              </p>
            )}
            {parsed.error && <p role="alert">{parsed.error}</p>}
            <div className="land-actions">
              <button type="button" onClick={() => update({ excluded: [] })}>
                Select all valid rows
              </button>
              <button
                type="button"
                onClick={() => update({ excluded: parsed.rows.map((row) => row.rowId) })}
              >
                Clear selection
              </button>
            </div>
            <div className="inventory-import-rows">
              {parsed.rows.slice(offset, offset + 20).map((row) => {
                const checked = !row.error && !draft.excluded.includes(row.rowId),
                  result = preview?.rows.find((r) => r.rowId === row.rowId);
                return (
                  <article key={row.rowId}>
                    <label>
                      <input
                        type="checkbox"
                        aria-label={`Import row ${row.rowId}`}
                        checked={checked}
                        disabled={!!row.error}
                        onChange={(event) =>
                          update({
                            excluded: event.target.checked
                              ? draft.excluded.filter((id) => id !== row.rowId)
                              : [...draft.excluded, row.rowId],
                          })
                        }
                      />{" "}
                      Row {row.rowId} · {row.geometry?.type ?? "Invalid geometry"}
                    </label>
                    <label className="land-name">
                      Asset name
                      <input
                        aria-label={`Row ${row.rowId} name`}
                        maxLength={200}
                        value={draft.names[row.rowId] ?? row.name}
                        onChange={(event) =>
                          update({ names: { ...draft.names, [row.rowId]: event.target.value } })
                        }
                      />
                    </label>
                    <label className="land-name">
                      Category
                      <select
                        aria-label={`Row ${row.rowId} category`}
                        value={draft.categories[row.rowId] ?? row.category}
                        onChange={(event) =>
                          update({
                            categories: {
                              ...draft.categories,
                              [row.rowId]: event.target.value as ImportCategory,
                            },
                          })
                        }
                      >
                        {inventoryCategories.map((category) => (
                          <option key={category}>{category}</option>
                        ))}
                      </select>
                    </label>
                    <p className="land-footnote">Record ID: {row.recordId}</p>
                    {row.error && <p className="land-error">{row.error}</p>}
                    {row.warnings.map((warning) => (
                      <p className="land-footnote" key={warning}>
                        {warning}
                      </p>
                    ))}
                    {result && (
                      <p>
                        {result.disposition === "created"
                          ? "Ready to create"
                          : result.disposition === "existing"
                            ? "Already in inventory · will be skipped"
                            : "Duplicate in this file · will be skipped"}
                        {result.geometryPreview && !result.geometryPreview.intersectsLand
                          ? ` · ${Math.round(result.geometryPreview.distanceM).toLocaleString()} m outside the current boundary`
                          : ""}
                      </p>
                    )}
                  </article>
                );
              })}
            </div>
            {parsed.rows.length > 20 && (
              <div className="land-actions">
                <button
                  type="button"
                  disabled={offset === 0}
                  onClick={() => setOffset(offset - 20)}
                >
                  Previous import rows
                </button>
                <span>
                  {offset + 1}–{Math.min(offset + 20, parsed.rows.length)}
                </span>
                <button
                  type="button"
                  disabled={offset + 20 >= parsed.rows.length}
                  onClick={() => setOffset(offset + 20)}
                >
                  More import rows
                </button>
              </div>
            )}
            {preview && (
              <p role="status">
                {preview.rows.filter((r) => r.disposition === "created").length} new candidates;{" "}
                {preview.rows.filter((r) => r.disposition !== "created").length} duplicates will be
                skipped. Review the candidate locations on the map.
              </p>
            )}
            <div className="land-actions">
              <button
                type="button"
                disabled={!selected.length || draft.boundaryRevision !== land.revision}
                onClick={() => void run(false)}
              >
                Preview selected assets
              </button>
              <button type="button" disabled={!preview} onClick={() => void run(true)}>
                Save reviewed import
              </button>
              <button
                type="button"
                onClick={() => {
                  ticket.current++;
                  persist(null);
                }}
              >
                Discard import draft
              </button>
            </div>
          </fieldset>
        )}
        {busy && (
          <p role="status">
            {preview ? "Saving inventory import…" : "Preparing inventory import…"}
          </p>
        )}
        {receipt && (
          <article aria-label="Import receipt">
            <h4>Import saved</h4>
            <p>
              {receipt.rows.filter((row) => row.disposition === "created").length} assets created;{" "}
              {receipt.rows.filter((row) => row.disposition !== "created").length} duplicates
              skipped.
            </p>
            <p className="land-footnote">
              {receipt.sourceLabel} · boundary revision {receipt.boundaryRevision}
            </p>
            <details>
              <summary>Imported asset links and file fingerprint</summary>
              <code>{receipt.sourceFileSha256}</code>
              {receipt.rows.map((row) => (
                <button
                  type="button"
                  key={row.rowId}
                  onClick={() => useLandContext.getState().selectInventory(row.featureId)}
                >
                  {row.name} · {row.disposition}
                </button>
              ))}
            </details>
          </article>
        )}
        <details>
          <summary>Previous inventory imports</summary>
          {history.isError && (
            <p role="alert">
              Import history could not be loaded.{" "}
              <button type="button" onClick={() => void history.refetch()}>
                Retry import history
              </button>
            </p>
          )}
          {history.data?.map((item) => (
            <button type="button" key={item.id} onClick={() => setReceipt(item)}>
              {item.sourceLabel} · {item.rows.length} rows ·{" "}
              {new Date(item.createdAt).toLocaleString()}
            </button>
          ))}
          {(historyOffset > 0 || history.data?.length === 20) && (
            <div className="land-actions">
              <button
                type="button"
                disabled={historyOffset === 0}
                onClick={() => setHistoryOffset(historyOffset - 20)}
              >
                Previous imports
              </button>
              <button
                type="button"
                disabled={history.data?.length !== 20}
                onClick={() => setHistoryOffset(historyOffset + 20)}
              >
                More imports
              </button>
            </div>
          )}
        </details>
      </details>
    </section>
  );
}
