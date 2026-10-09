import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ExplorationControls } from "./ExplorationControls";
import { explorationProgress } from "./exploration";

describe("chunked exploration controls", () => {
  it("sends a bounded complete configuration only when applied", async () => {
    const submit = vi.fn().mockResolvedValue(undefined);
    render(<ExplorationControls initialQuality="low-latency" disabled={false} onSubmit={submit} />);
    expect(screen.getByRole("combobox", { name: "Generation cadence" })).toHaveValue("responsive");
    fireEvent.change(screen.getByRole("combobox", { name: "Exploration mode" }), {
      target: { value: "cruise" },
    });
    fireEvent.change(screen.getByRole("slider"), { target: { value: "0.75" } });
    expect(submit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Apply exploration settings" }));
    await waitFor(() =>
      expect(submit).toHaveBeenCalledWith({
        type: "native",
        action: "exploration",
        values: { mode: "cruise", cadence: "responsive", speed: 0.75 },
      }),
    );
    expect(await screen.findByText("Settings received · apply to the next chunk")).toBeVisible();
  });

  it("shows failed commands without claiming they applied", async () => {
    render(
      <ExplorationControls
        initialQuality="quality"
        disabled={false}
        onSubmit={() => Promise.reject(new Error("Worker unavailable"))}
      />,
    );
    expect(screen.getByRole("combobox", { name: "Generation cadence" })).toHaveValue("smooth");
    fireEvent.click(screen.getByRole("button", { name: "Apply exploration settings" }));
    expect(await screen.findByText("Worker unavailable")).toBeVisible();
    expect(screen.queryByText(/Settings received/)).not.toBeInTheDocument();
  });

  it("reports shortened model context only when the worker reports truncation", () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(
      <ExplorationControls initialQuality="balanced" disabled={false} onSubmit={onSubmit} />,
    );
    expect(screen.queryByText(/Scene context shortened/)).not.toBeInTheDocument();
    rerender(
      <ExplorationControls
        initialQuality="balanced"
        disabled={false}
        onSubmit={onSubmit}
        promptTruncated
      />,
    );
    expect(
      screen.getByText("Scene context shortened to fit model; newest directions prioritized."),
    ).toBeVisible();
    rerender(
      <ExplorationControls
        initialQuality="balanced"
        disabled={false}
        onSubmit={onSubmit}
        promptTruncated={false}
      />,
    );
    expect(screen.queryByText(/Scene context shortened/)).not.toBeInTheDocument();
  });

  it("keeps accepted changes queued until generated output reaches their revision", () => {
    expect(explorationProgress(3, 1, 2)).toBe(
      "Change 3 received · queued for the next chunk · generating change 2",
    );
    expect(explorationProgress(3, 3, null)).toBe("Change 3 applied to generated output");
    expect(explorationProgress(1)).toBe("Change 1 received · queued for the next chunk");
    expect(explorationProgress()).toBe("Commands apply at the next chunk boundary");
  });
});
