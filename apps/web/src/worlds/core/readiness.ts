import { useCallback, useEffect, useState } from "react";
import { createWorldApi, type WorldsReadiness } from "./api";
import type { ProviderId } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function isNonnegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
function optionalString(value: unknown): boolean {
  return value === undefined || typeof value === "string";
}
function validProvider(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    ["runpod", "local", "modal", "lambda", "coreweave", "aws", "gcp", "azure"].includes(value.id) &&
    typeof value.name === "string" &&
    typeof value.configured === "boolean" &&
    typeof value.canProvision === "boolean" &&
    typeof value.message === "string" &&
    (value.gpuTypeId === null || optionalString(value.gpuTypeId))
  );
}
function validModel(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    value.id.length > 0 &&
    optionalString(value.status) &&
    optionalString(value.reason) &&
    optionalString(value.name)
  );
}
function validGateway(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.provider === "string" &&
    optionalString(value.modelId) &&
    typeof value.status === "string" &&
    optionalString(value.message) &&
    optionalString(value.error) &&
    Array.isArray(value.models) &&
    value.models.every(validModel)
  );
}
function validReadiness(value: unknown): value is WorldsReadiness {
  if (
    !isRecord(value) ||
    typeof value.provisioningEnabled !== "boolean" ||
    !Array.isArray(value.providers) ||
    !value.providers.every(validProvider) ||
    !Array.isArray(value.gateways) ||
    !value.gateways.every(validGateway) ||
    (value.modelProfiles !== undefined &&
      (!Array.isArray(value.modelProfiles) ||
        !value.modelProfiles.every(
          (profile: unknown) =>
            isRecord(profile) &&
            typeof profile.modelId === "string" &&
            Array.isArray(profile.providers) &&
            profile.providers.every((item: unknown) => typeof item === "string") &&
            [profile.gatewayProviders, profile.provisioningProviders].every(
              (list) =>
                list === undefined ||
                (Array.isArray(list) && list.every((item: unknown) => typeof item === "string")),
            ),
        ))) ||
    (value.configurationIssue !== null && !optionalString(value.configurationIssue)) ||
    !isRecord(value.lifecycle)
  )
    return false;
  const life = value.lifecycle;
  return (
    typeof life.enabled === "boolean" &&
    typeof life.running === "boolean" &&
    [
      "sessionLeaseSeconds",
      "heartbeatIntervalSeconds",
      "workerIdleSeconds",
      "workerMaxLifetimeSeconds",
      "workerStartupSeconds",
      "maxManagedWorkers",
    ].every((key) => isNonnegativeNumber(life[key])) &&
    (life.maxWorkerHourlyCost === null || isNonnegativeNumber(life.maxWorkerHourlyCost))
  );
}
/** Reject partial/stale API responses before they can affect paid-compute readiness. */
export function parseReadinessReport(value: unknown): WorldsReadiness {
  if (!validReadiness(value))
    throw new Error(
      "The session manager returned an invalid readiness report. Update the Worlds API and check the server URL, then refresh readiness.",
    );
  return value;
}

export interface ComputeStatus {
  kind: "ready" | "provision" | "blocked";
  title: string;
  detail: string;
  canLaunch: boolean;
}

/** Configuration is not evidence of a reachable, installed model. */
export function assessCompute(
  report: WorldsReadiness,
  providerId: ProviderId,
  modelId?: string,
): ComputeStatus {
  const provider = report.providers.find((item) => item.id === providerId);
  if (!provider?.configured) {
    return {
      kind: "blocked",
      title: "Compute needs configuration",
      detail:
        (providerId === "runpod" ? report.configurationIssue : undefined) ??
        provider?.message ??
        "Configure this provider in the session manager before starting inference.",
      canLaunch: false,
    };
  }
  const profile = report.modelProfiles?.find((item) => item.modelId === modelId);
  if (modelId && report.modelProfiles && !profile?.providers.includes(providerId))
    return {
      kind: "blocked",
      title: "Model runtime needs configuration",
      detail: `Configure an approved ${modelId} runtime for ${provider.name} before starting compute.`,
      canLaunch: false,
    };
  const gateways = report.gateways.filter((item) => item.provider === providerId);
  const gateway =
    gateways.find((item) => item.modelId === modelId) ??
    gateways.find((item) => item.models.some((model) => model.id === modelId)) ??
    gateways.find((item) => item.modelId === undefined);
  if (
    profile?.gatewayProviders?.includes(providerId) &&
    (!gateway || gateway.status === "unconfigured")
  )
    return {
      kind: "blocked",
      title: "Gateway is not ready",
      detail:
        "The configured model gateway has not passed its health check. Refresh readiness after checking the worker.",
      canLaunch: false,
    };
  if (gateway && gateway.status !== "unconfigured") {
    if (!["ready", "healthy", "ok"].includes(gateway.status)) {
      return {
        kind: "blocked",
        title: "Gateway is not ready",
        detail:
          gateway.message ??
          gateway.error ??
          "The configured worker did not pass its health check. Check the worker installation and connection, then refresh.",
        canLaunch: false,
      };
    }
    const model = modelId ? gateway.models.find((item) => item.id === modelId) : gateway.models[0];
    if (!model)
      return {
        kind: "blocked",
        title: "Model is not installed",
        detail: modelId
          ? `The worker does not report ${modelId}. Select an installed model or configure its adapter on this worker.`
          : "The gateway is reachable but does not report an installed model.",
        canLaunch: false,
      };
    if (model.status && !["ready", "healthy", "ok", "available"].includes(model.status))
      return {
        kind: "blocked",
        title: "Model is not ready",
        detail:
          model.reason ??
          `The worker reports ${model.status}. Finish model setup before starting inference.`,
        canLaunch: false,
      };
    return {
      kind: "ready",
      title: "Worker and model are ready",
      detail:
        "A live health check confirmed the configured model. This uses an existing worker; generation performance has not been measured.",
      canLaunch: true,
    };
  }
  if (
    provider.canProvision &&
    report.provisioningEnabled &&
    (!profile?.provisioningProviders || profile.provisioningProviders.includes(providerId))
  ) {
    if (!report.lifecycle.enabled || !report.lifecycle.running)
      return {
        kind: "blocked",
        title: "Automatic cleanup is not running",
        detail:
          "Start the session manager lifecycle supervisor before allocating a paid GPU. Allocation stays blocked until abandoned-worker cleanup is active.",
        canLaunch: false,
      };
    return {
      kind: "provision",
      title: "Ready to start a GPU",
      detail: `An approved ${provider.name} runtime is configured. Starting allocates paid compute; model availability is verified after the worker boots.`,
      canLaunch: true,
    };
  }
  return {
    kind: "blocked",
    title: "No worker is connected",
    detail:
      "Configure a compatible gateway or enable an approved provider runtime in the session manager.",
    canLaunch: false,
  };
}

export function useComputeReadiness(serverUrl: string, credentialsRevision = 0) {
  const [revision, setRevision] = useState(0);
  const key = `${serverUrl}:${credentialsRevision}:${revision}`;
  const [snapshot, setSnapshot] = useState<{
    key: string;
    report?: WorldsReadiness;
    error?: string;
  }>();
  useEffect(() => {
    const controller = new AbortController();
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]);
    void (async () => {
      try {
        const report = parseReadinessReport(await createWorldApi(serverUrl).readiness(signal));
        if (!controller.signal.aborted) setSnapshot({ key, report });
      } catch (failure) {
        if (!controller.signal.aborted)
          setSnapshot({
            key,
            error:
              failure instanceof Error ? failure.message : "Could not check compute readiness.",
          });
      }
    })();
    return () => controller.abort();
  }, [serverUrl, key]);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  return {
    report: snapshot?.key === key ? snapshot.report : undefined,
    error: snapshot?.key === key ? snapshot.error : undefined,
    checking: snapshot?.key !== key,
    refresh,
  };
}
