import { useEffect, useRef, useState } from "react";
import { Download, RefreshCw, Upload } from "lucide-react";
import { createWorldApi, type WorkerHandle } from "./core/api";
import { downloadBlob } from "./core/storage";

interface BillingReport {
  records: {
    id: string;
    provider: string;
    reference: string;
    source: string;
    amount: number;
    currency: string;
    category: string;
    matchStatus: string;
    workerId: string | null;
    unmatchedReason: string | null;
  }[];
  totalRecords: number;
  unmatchedCount: number;
  workerTotals: {
    workerId: string;
    currency: string;
    importedActual: number;
    importedCompute: number;
    computeDeltaUSD: number | null;
    estimatedComputeCostUSD: number | null;
  }[];
  totals: {
    currency: string;
    importedActual: number;
    matchedActual: number;
    unmatchedActual: number;
  }[];
  message: string;
}
interface ProviderBill {
  workerId: string;
  currency: string;
  reportedAmountUSD: number | null;
  observedAt: string;
  message: string;
  records: { time: string; amount: number; timeBilledMs?: number; diskSpaceBilledGb?: number }[];
}
const dollars = (value: number | null) =>
  value === null ? "Not comparable" : `$${value.toFixed(4)}`;

export function BillingPanel({ serverUrl }: { serverUrl: string }) {
  const [report, setReport] = useState<BillingReport>();
  const [workers, setWorkers] = useState<WorkerHandle[]>([]);
  const [workerId, setWorkerId] = useState("");
  const [start, setStart] = useState(() =>
    new Date(Date.now() - 7 * 86400000).toISOString().slice(0, 10),
  );
  const [end, setEnd] = useState(() => new Date().toISOString().slice(0, 10));
  const [providerBill, setProviderBill] = useState<ProviderBill>();
  const [file, setFile] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), [serverUrl]);
  async function run(kind: "refresh" | "import" | "provider") {
    if (busy) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setError("");
    try {
      const api = createWorldApi(serverUrl);
      if (kind === "provider") {
        if (!workerId || !start || !end || start >= end)
          throw new Error("Choose an owned RunPod worker and a valid billing period.");
        setProviderBill(undefined);
        const query = new URLSearchParams({
          startTime: `${start}T00:00:00Z`,
          endTime: `${end}T00:00:00Z`,
        });
        const bill = await api.request<ProviderBill>(
          `/billing/runpod/${encodeURIComponent(workerId)}?${query}`,
          { signal: controller.signal },
        );
        controller.signal.throwIfAborted();
        setProviderBill(bill);
      } else {
        if (kind === "import") {
          if (!file) throw new Error("Select an invoice export JSON file.");
          if (file.size > 1024 * 1024)
            throw new Error("Invoice imports must be smaller than 1 MB.");
          const body: unknown = JSON.parse(await file.text());
          controller.signal.throwIfAborted();
          const result = await api.request<BillingReport>("/billing/import", {
            method: "POST",
            body: JSON.stringify(Array.isArray(body) ? { rows: body } : body),
            signal: controller.signal,
          });
          controller.signal.throwIfAborted();
          setReport(result);
          setFile(undefined);
        } else {
          const result = await api.request<BillingReport>("/billing?limit=100", {
            signal: controller.signal,
          });
          controller.signal.throwIfAborted();
          setReport(result);
        }
        const result = await api.workers();
        controller.signal.throwIfAborted();
        setWorkers(
          result.workers.filter((worker) => worker.provider === "runpod" && worker.managed),
        );
      }
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (active.current === controller) {
        active.current = null;
        setBusy(false);
      }
    }
  }
  return (
    <section className="w-panel">
      <h2>Billing reconciliation</h2>
      <p className="w-muted">
        Compare compute estimates with invoice exports or a read-only RunPod billing lookup.
        Imported amounts retain their source and currency; estimates are never presented as actual
        charges.
      </p>
      <button className="w-btn w-btn-quiet" disabled={busy} onClick={() => void run("refresh")}>
        <RefreshCw size={14} />
        Load billing ledger
      </button>
      <details>
        <summary>Invoice JSON format</summary>
        <p className="w-muted">
          Use a unique line reference from the provider's export. Identify a worker with either its
          local workerId or its providerId. Credits must use kind "credit" and a negative amount.
        </p>
        <pre className="w-code-block">
          {JSON.stringify(
            {
              rows: [
                {
                  provider: "runpod",
                  reference: "invoice-line-123",
                  source: "provider-invoice.json",
                  workerId: "owned-worker-id",
                  amount: "0.42",
                  currency: "USD",
                  periodStart: "2026-10-09T00:00:00Z",
                  periodEnd: "2026-10-09T01:00:00Z",
                  kind: "charge",
                  category: "compute",
                },
              ],
            },
            null,
            2,
          )}
        </pre>
      </details>
      <label className="w-form-field">
        <span>INVOICE EXPORT</span>
        <input
          type="file"
          accept="application/json,.json"
          disabled={busy}
          onChange={(event) => setFile(event.target.files?.[0])}
        />
      </label>
      <button
        className="w-btn w-btn-quiet"
        disabled={busy || !file}
        onClick={() => void run("import")}
      >
        <Upload size={14} />
        Import invoice lines
      </button>
      {report && (
        <>
          <p className="w-muted">
            {report.message} {report.totalRecords} lines · {report.unmatchedCount} unmatched.
          </p>
          {report.totals.map((total) => (
            <p key={total.currency}>
              <strong>
                {total.importedActual.toFixed(4)} {total.currency}
              </strong>{" "}
              imported · {total.unmatchedActual.toFixed(4)} unmatched
            </p>
          ))}
          <div className="w-table-wrap">
            <table className="w-table">
              <thead>
                <tr>
                  <th>Worker / currency</th>
                  <th>Imported charges</th>
                  <th>Estimated compute for invoiced periods</th>
                  <th>Compute difference</th>
                </tr>
              </thead>
              <tbody>
                {report.workerTotals.map((worker) => (
                  <tr key={`${worker.workerId}:${worker.currency}`}>
                    <td>
                      {worker.workerId}
                      <br />
                      {worker.currency}
                    </td>
                    <td>{worker.importedActual.toFixed(4)}</td>
                    <td>{dollars(worker.estimatedComputeCostUSD)}</td>
                    <td>{dollars(worker.computeDeltaUSD)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {report.records
            .filter((record) => record.matchStatus === "unmatched")
            .slice(0, 10)
            .map((record) => (
              <p className="w-muted" key={record.id}>
                {record.reference} · {record.unmatchedReason}
              </p>
            ))}
          <button
            className="w-btn w-btn-quiet"
            onClick={() =>
              downloadBlob(
                new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
                "worlds-billing-reconciliation.json",
              )
            }
          >
            <Download size={14} />
            Export reconciliation
          </button>
        </>
      )}
      <h3>RunPod provider-reported billing</h3>
      <label className="w-form-field">
        <span>OWNED RUNPOD WORKER</span>
        <select
          value={workerId}
          disabled={busy}
          onChange={(event) => {
            setWorkerId(event.target.value);
            setProviderBill(undefined);
          }}
        >
          <option value="">Load the ledger, then choose a worker</option>
          {workers.map((worker) => (
            <option key={worker.id} value={worker.id}>
              {worker.modelId ?? "World model"} · {worker.id} · {worker.status}
            </option>
          ))}
        </select>
      </label>
      <label className="w-form-field">
        <span>START DATE (UTC)</span>
        <input
          type="date"
          value={start}
          disabled={busy}
          onChange={(event) => {
            setStart(event.target.value);
            setProviderBill(undefined);
          }}
        />
      </label>
      <label className="w-form-field">
        <span>END DATE (UTC, EXCLUSIVE)</span>
        <input
          type="date"
          value={end}
          disabled={busy}
          onChange={(event) => {
            setEnd(event.target.value);
            setProviderBill(undefined);
          }}
        />
      </label>
      <button
        className="w-btn w-btn-quiet"
        disabled={busy || !workerId}
        onClick={() => void run("provider")}
      >
        Read provider billing
      </button>
      {providerBill && (
        <div className="w-info-strip">
          <span>
            <strong>{dollars(providerBill.reportedAmountUSD)} reported by RunPod</strong>
            <br />
            {providerBill.message}
            <br />
            Observed {new Date(providerBill.observedAt).toLocaleString()}
            <div className="w-table-wrap">
              <table className="w-table">
                <thead>
                  <tr>
                    <th>Billing date</th>
                    <th>Amount (USD)</th>
                    <th>Billed GPU minutes</th>
                  </tr>
                </thead>
                <tbody>
                  {providerBill.records.map((record, index) => (
                    <tr key={`${record.time}:${index}`}>
                      <td>{new Date(record.time).toLocaleDateString()}</td>
                      <td>{dollars(record.amount)}</td>
                      <td>
                        {record.timeBilledMs === undefined
                          ? "Not reported"
                          : (record.timeBilledMs / 60000).toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </span>
        </div>
      )}
      {error && (
        <p className="w-inline-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
