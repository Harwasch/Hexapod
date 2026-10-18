import { fireEvent, render, screen, waitFor, cleanup } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IntelligencePanel, type IntelligencePanelProps } from "./IntelligencePanel";
import { getIntelligenceStatus, interpretCommand } from "./client";
import type * as Client from "./client";
import { emptyCapabilities } from "../core/catalog";
vi.mock("./client", async (importOriginal) => ({
  ...(await importOriginal<typeof Client>()),
  getIntelligenceStatus: vi.fn(),
  interpretCommand: vi.fn(),
  observeWorld: vi.fn(),
}));
beforeEach(() => {
  vi.mocked(getIntelligenceStatus).mockResolvedValue({
    configured: true,
    visionConfigured: true,
    imageConfigured: false,
    message: "Ready",
  });
  vi.mocked(interpretCommand).mockResolvedValue({
    observation: "A red object on the ground",
    explanation: "Native movement",
    action: { type: "native", action: "forward" },
    source: "vision-llm",
    experimental: true,
  });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
function props(): IntelligencePanelProps {
  return {
    serverUrl: "",
    modelId: "test",
    capabilities: emptyCapabilities(),
    nativeActions: ["forward"],
    worldPrompt: "A forest",
    getFrame: vi.fn().mockResolvedValue(new Blob(["frame"], { type: "image/jpeg" })),
    onCommand: vi.fn().mockResolvedValue(undefined),
  };
}
async function requestProposal() {
  fireEvent.click(screen.getByLabelText(/Send current frames/));
  fireEvent.change(screen.getByLabelText("Describe an action"), {
    target: { value: "Move forward" },
  });
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Interpret from this frame" })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Interpret from this frame" }));
  await screen.findByRole("button", { name: "Apply suggested action" });
}
describe("frame-grounded command consent and stale-action protection", () => {
  it("does not send frames until consent and reviews the action before applying", async () => {
    const p = props();
    render(<IntelligencePanel {...p} />);
    fireEvent.change(screen.getByLabelText("Describe an action"), { target: { value: "Move" } });
    expect(screen.getByRole("button", { name: "Interpret from this frame" })).toBeDisabled();
    expect(p.getFrame).not.toHaveBeenCalled();
    await requestProposal();
    expect(p.onCommand).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Apply suggested action" }));
    await waitFor(() =>
      expect(p.onCommand).toHaveBeenCalledExactlyOnceWith({ type: "native", action: "forward" }),
    );
  });
  it("rejects a reviewed proposal after pause and resume", async () => {
    const p = props();
    const { rerender } = render(<IntelligencePanel {...p} />);
    await requestProposal();
    rerender(<IntelligencePanel {...p} disabled />);
    rerender(<IntelligencePanel {...p} disabled={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Apply suggested action" }));
    await screen.findByText(/stale, paused, or unsupported/);
    expect(p.onCommand).not.toHaveBeenCalled();
  });
  it("never applies a native action absent from the model vocabulary", async () => {
    vi.mocked(interpretCommand).mockResolvedValue({
      observation: "A mountain",
      explanation: "Fly",
      action: { type: "native", action: "fly" },
      source: "vision-llm",
      experimental: true,
    });
    const p = props();
    render(<IntelligencePanel {...p} />);
    await requestProposal();
    fireEvent.click(screen.getByRole("button", { name: "Apply suggested action" }));
    await screen.findByText(/stale, paused, or unsupported/);
    expect(p.onCommand).not.toHaveBeenCalled();
  });
  it("aborts pending observation when the session becomes disabled", async () => {
    vi.mocked(interpretCommand).mockImplementation(() => new Promise(() => undefined));
    const p = props();
    const { rerender } = render(<IntelligencePanel {...p} />);
    fireEvent.click(screen.getByLabelText(/Send current frames/));
    fireEvent.change(screen.getByLabelText("Describe an action"), { target: { value: "Move" } });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Interpret from this frame" })).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Interpret from this frame" }));
    await waitFor(() => expect(interpretCommand).toHaveBeenCalledOnce());
    const signal = vi.mocked(interpretCommand).mock.calls[0]?.[0].signal;
    rerender(<IntelligencePanel {...p} disabled />);
    expect(signal?.aborted).toBe(true);
    expect(p.onCommand).not.toHaveBeenCalled();
  });
});
