import { useEffect, useRef, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { createWorldApi } from "./core/api";
import { MODELS, PROVIDERS } from "./core/catalog";
import type { ProviderId } from "./core/types";

interface Hardware {
  id: string;
  name: string;
  memoryGB: number | null;
  regions: string[];
  available: boolean | null;
  hourlyCost: number | null;
  pricingSource: string;
}
interface Quote {
  gpuTypeId: string | null;
  estimatedCostUSD: number | null;
  hourlyCost: number | null;
  pricingSource: string;
  message: string;
}
interface Usage {
  estimatedCostUSD: number | null;
  message: string;
  workers: {
    workerId: string;
    modelId: string;
    provider: string;
    status: string;
    durationSeconds: number;
    estimatedCostUSD: number | null;
  }[];
}
const cost = (value: number | null) => (value === null ? "Unavailable" : `$${value.toFixed(3)}`);

/** Provider reads are explicit; this panel never allocates compute. */
export function ComputeInventory({
  serverUrl,
  initialProvider,
}: {
  serverUrl: string;
  initialProvider: ProviderId;
}) {
  const [provider, setProvider] = useState(initialProvider);
  const [modelId, setModelId] = useState("astronex-world");
  const [minutes, setMinutes] = useState(10);
  const [hardware, setHardware] = useState<Hardware[]>([]);
  const [quote, setQuote] = useState<Quote>();
  const [usage, setUsage] = useState<Usage>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), [serverUrl]);
  function reset() {
    active.current?.abort();
    setHardware([]);
    setQuote(undefined);
    setError("");
    setBusy(false);
  }
  async function inspect() {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setError("");
    setQuote(undefined);
    setHardware([]);
    const api = createWorldApi(serverUrl);
    const results = await Promise.allSettled([
      api.request<{ hardware: Hardware[] }>(`/providers/${provider}/hardware`, {
        signal: controller.signal,
      }),
      api.request<Quote>(`/providers/${provider}/quote`, {
        method: "POST",
        body: JSON.stringify({ modelId, durationMinutes: minutes }),
        signal: controller.signal,
      }),
      api.request<Usage>("/usage", { signal: controller.signal }),
    ]);
    if (controller.signal.aborted) return;
    const [devices, estimate, ledger] = results;
    if (devices.status === "fulfilled") setHardware(devices.value.hardware);
    if (estimate.status === "fulfilled") setQuote(estimate.value);
    if (ledger.status === "fulfilled") setUsage(ledger.value);
    setError(
      results
        .filter((result) => result.status === "rejected")
        .map((result) =>
          String(result.reason instanceof Error ? result.reason.message : result.reason),
        )
        .join(" "),
    );
    setBusy(false);
  }
  return (
    <section className="w-panel">
      <h2>Hardware and usage</h2>
      <p className="w-muted">
        Inspect provider inventory and the approved runtime's compute estimate. Launching uses the
        runtime configured by your operator.
      </p>
      <label className="w-form-field">
        <span>PROVIDER</span>
        <select
          value={provider}
          onChange={(event) => {
            reset();
            setProvider(event.target.value as ProviderId);
          }}
        >
          {PROVIDERS.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      <label className="w-form-field">
        <span>MODEL RUNTIME</span>
        <select
          value={modelId}
          onChange={(event) => {
            reset();
            setModelId(event.target.value);
          }}
        >
          {MODELS.filter((model) => model.status === "adapter-ready").map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </label>
      <label className="w-form-field">
        <span>ESTIMATE DURATION (MINUTES)</span>
        <input
          type="number"
          min={1}
          max={60}
          value={minutes}
          onChange={(event) => {
            reset();
            setMinutes(Math.max(1, Math.min(60, Number(event.target.value) || 1)));
          }}
        />
      </label>
      <button className="w-btn w-btn-quiet" disabled={busy} onClick={() => void inspect()}>
        {busy ? <Loader2 className="w-spin" size={15} /> : <RefreshCw size={15} />}Inspect hardware
        and costs
      </button>
      {quote && (
        <div className="w-info-strip">
          <span>
            <strong>
              {cost(quote.estimatedCostUSD)} for {minutes} min
            </strong>{" "}
            · {cost(quote.hourlyCost)}/hr · {quote.pricingSource}
            <br />
            {quote.gpuTypeId}
            <br />
            {quote.message}
          </span>
        </div>
      )}
      {hardware.length > 0 && (
        <div className="w-table-wrap">
          <table className="w-table">
            <thead>
              <tr>
                <th>GPU</th>
                <th>VRAM</th>
                <th>Availability</th>
                <th>Rate/hr</th>
              </tr>
            </thead>
            <tbody>
              {hardware.map((gpu) => (
                <tr key={gpu.id}>
                  <td>
                    {gpu.name}
                    <small>{gpu.regions.join(", ")}</small>
                  </td>
                  <td>{gpu.memoryGB === null ? "Unknown" : `${gpu.memoryGB} GB`}</td>
                  <td>
                    {gpu.available === null
                      ? "Unknown"
                      : gpu.available
                        ? "Available"
                        : "Unavailable"}
                  </td>
                  <td>{cost(gpu.hourlyCost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {usage && (
        <>
          <h3>Session ledger · {cost(usage.estimatedCostUSD)} estimated</h3>
          <p className="w-muted">{usage.message}</p>
          {usage.workers
            .slice(-10)
            .reverse()
            .map((worker) => (
              <p className="w-muted" key={worker.workerId}>
                {worker.modelId} · {worker.provider} · {worker.status} ·{" "}
                {Math.round(worker.durationSeconds / 60)} min · {cost(worker.estimatedCostUSD)}
              </p>
            ))}
        </>
      )}
      {error && (
        <p className="w-inline-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
