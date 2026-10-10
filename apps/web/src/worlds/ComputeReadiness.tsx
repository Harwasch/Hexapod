import { CheckCircle2, CircleAlert, Clock3, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import type { WorldsReadiness } from "./core/api";
import type { ProviderId } from "./core/types";
import { assessCompute } from "./core/readiness";

export function ReadinessDetails({
  report,
  providerId,
  modelId,
  compact = false,
}: {
  report: WorldsReadiness;
  providerId: ProviderId;
  modelId?: string;
  compact?: boolean;
}) {
  const status = assessCompute(report, providerId, modelId);
  const provider = report.providers.find((p) => p.id === providerId);
  return (
    <div className={`w-readiness ${compact ? "w-readiness-compact" : ""}`}>
      <div className={`w-readiness-status ${status.kind}`}>
        {status.kind === "blocked" ? <CircleAlert size={17} /> : <CheckCircle2 size={17} />}
        <div>
          <strong>{status.title}</strong>
          <p>{status.detail}</p>
        </div>
      </div>
      {provider?.gpuTypeId && (
        <p className="w-readiness-fact">
          <span>Configured GPU</span>
          <strong>{provider.gpuTypeId}</strong>
        </p>
      )}
      {!compact && (
        <>
          <div className="w-readiness-limits">
            <div>
              <ShieldCheck size={15} />
              <span>Abandoned session cleanup</span>
              <strong>
                {report.lifecycle.enabled && report.lifecycle.running
                  ? `After ${report.lifecycle.sessionLeaseSeconds}s without a heartbeat`
                  : "Not running"}
              </strong>
            </div>
            <div>
              <Clock3 size={15} />
              <span>Maximum managed worker lifetime</span>
              <strong>
                {report.lifecycle.workerMaxLifetimeSeconds > 0
                  ? `${Math.round(report.lifecycle.workerMaxLifetimeSeconds / 60)} min`
                  : "Not set"}
              </strong>
            </div>
            <div>
              <Clock3 size={15} />
              <span>Idle managed worker timeout</span>
              <strong>{Math.round(report.lifecycle.workerIdleSeconds / 60)} min</strong>
            </div>
          </div>
          <p className="w-readiness-note">
            Managed worker limit: {report.lifecycle.maxManagedWorkers}.{" "}
            {report.lifecycle.maxWorkerHourlyCost === null
              ? "No hourly GPU price cap configured."
              : `Hourly GPU price cap: $${report.lifecycle.maxWorkerHourlyCost.toFixed(2)}.`}{" "}
            Provider storage and network charges are separate. Existing external workers must be
            stopped by their owner.
          </p>
        </>
      )}
    </div>
  );
}
export function ReadinessRefresh({
  checking,
  onRefresh,
}: {
  checking: boolean;
  onRefresh: () => void;
}) {
  return (
    <button className="w-btn w-btn-quiet" disabled={checking} onClick={onRefresh}>
      {checking ? <Loader2 size={14} className="w-spin" /> : <RefreshCw size={14} />}{" "}
      {checking ? "Checking readiness…" : "Refresh readiness"}
    </button>
  );
}
