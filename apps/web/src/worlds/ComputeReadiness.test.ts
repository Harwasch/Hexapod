import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import * as apiModule from "./core/api";
import { assessCompute, parseReadinessReport, useComputeReadiness } from "./core/readiness";
import type { WorldsReadiness } from "./core/api";

function report(): WorldsReadiness {
  return {
    provisioningEnabled: true,
    providers: [
      {
        id: "runpod",
        name: "RunPod",
        configured: true,
        canProvision: true,
        message: "Template configured",
      },
    ],
    gateways: [{ provider: "runpod", status: "unconfigured", models: [] }],
    lifecycle: {
      enabled: true,
      running: true,
      sessionLeaseSeconds: 90,
      heartbeatIntervalSeconds: 20,
      workerIdleSeconds: 600,
      workerMaxLifetimeSeconds: 3600,
      workerStartupSeconds: 1200,
      maxManagedWorkers: 1,
      maxWorkerHourlyCost: 2,
    },
  };
}
describe("compute launch readiness", () => {
  it("matches a model-specific gateway and never replaces its failed health check with provisioning", () => {
    const input = report();
    input.modelProfiles = [
      {
        modelId: "forge-wm",
        providers: ["runpod"],
        gatewayProviders: ["runpod"],
        provisioningProviders: ["runpod"],
      },
    ];
    input.gateways = [
      {
        provider: "runpod",
        modelId: "astronex-world",
        status: "ready",
        models: [{ id: "astronex-world", status: "ready" }],
      },
      { provider: "runpod", modelId: "forge-wm", status: "unavailable", models: [] },
    ];
    expect(assessCompute(input, "runpod", "forge-wm").canLaunch).toBe(false);
    input.gateways[1] = {
      provider: "runpod",
      modelId: "forge-wm",
      status: "ready",
      models: [{ id: "forge-wm", status: "ready" }],
    };
    expect(assessCompute(input, "runpod", "forge-wm")).toMatchObject({
      canLaunch: true,
      kind: "ready",
    });
    expect(assessCompute(input, "runpod", "sana-wm").canLaunch).toBe(false);
  });
  it("does not confuse provider configuration with a healthy installed model", () => {
    const input = report();
    input.gateways = [
      { provider: "runpod", status: "unavailable", message: "Weights are missing", models: [] },
    ];
    expect(assessCompute(input, "runpod", "astronex-world")).toMatchObject({
      canLaunch: false,
      detail: "Weights are missing",
    });
    input.gateways = [
      { provider: "runpod", status: "ready", models: [{ id: "another-model", status: "ready" }] },
    ];
    expect(assessCompute(input, "runpod", "astronex-world")).toMatchObject({
      canLaunch: false,
      title: "Model is not installed",
    });
  });
  it("blocks paid allocation if abandoned-worker cleanup has stopped", () => {
    const input = report();
    input.lifecycle.running = false;
    expect(assessCompute(input, "runpod", "astronex-world")).toMatchObject({
      canLaunch: false,
      title: "Automatic cleanup is not running",
    });
    input.lifecycle.running = true;
    expect(assessCompute(input, "runpod", "astronex-world")).toMatchObject({
      canLaunch: true,
      kind: "provision",
    });
  });
  it("does not silently provision a replacement for an unhealthy configured gateway", () => {
    const input = report();
    input.gateways = [{ provider: "runpod", status: "unavailable", models: [] }];
    expect(assessCompute(input, "runpod", "astronex-world").canLaunch).toBe(false);
  });
  it("requires actual model readiness even when its gateway is healthy", () => {
    const input = report();
    input.gateways = [
      {
        provider: "runpod",
        status: "ready",
        models: [{ id: "astronex-world", status: "unavailable", reason: "Checkpoint missing" }],
      },
    ];
    expect(assessCompute(input, "runpod", "astronex-world")).toMatchObject({
      canLaunch: false,
      detail: "Checkpoint missing",
    });
    input.gateways[0]!.models[0]!.status = "ready";
    expect(assessCompute(input, "runpod", "astronex-world")).toMatchObject({
      canLaunch: true,
      kind: "ready",
    });
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
describe("readiness response validation", () => {
  it.each([
    null,
    [],
    {},
    { providers: [] },
    { ...report(), providers: {} },
    { ...report(), providers: [null] },
    { ...report(), gateways: ["ready"] },
    { ...report(), gateways: [{ provider: "runpod", status: "ready", models: {} }] },
    { ...report(), gateways: [{ provider: "runpod", status: "ready", models: [null] }] },
    { ...report(), lifecycle: [] },
    { ...report(), lifecycle: { ...report().lifecycle, maxWorkerHourlyCost: "2.00" } },
    { ...report(), lifecycle: { ...report().lifecycle, maxManagedWorkers: -1 } },
  ])("rejects malformed JSON without exposing it to readiness rendering (%#)", (input: unknown) => {
    expect(() => parseReadinessReport(input)).toThrow("invalid readiness report");
  });
  it("accepts the complete response and an explicitly unset hourly cap", () => {
    const input = report();
    input.lifecycle.maxWorkerHourlyCost = null;
    expect(parseReadinessReport(input)).toEqual(input);
  });
  it("turns a malformed HTTP 200 response into a recoverable hook error", async () => {
    const api = apiModule.createWorldApi("https://manager.example");
    vi.spyOn(apiModule, "createWorldApi").mockReturnValue({
      ...api,
      readiness: vi.fn().mockResolvedValue({ providers: [] }),
    });
    const { result } = renderHook(() => useComputeReadiness("https://manager.example"));
    await waitFor(() => expect(result.current.checking).toBe(false));
    expect(result.current.report).toBeUndefined();
    expect(result.current.error).toContain("Update the Worlds API");
  });
});
