import { fireEvent, render, screen, waitFor, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkerManager } from "./WorkerManager";
import { createWorldApi } from "./core/api";

vi.mock("./core/api", () => ({ createWorldApi: vi.fn() }));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
function workerApi() {
  return {
    request: vi.fn().mockImplementation((path: string) =>
      Promise.resolve(
        path === "/workers"
          ? {
              workers: [
                {
                  id: "pod-1",
                  provider: "runpod",
                  status: "ready",
                  managed: true,
                  estimatedHourlyCost: 1.2,
                },
              ],
            }
          : { sessions: [] },
      ),
    ),
    destroyWorker: vi.fn().mockResolvedValue({ status: "destroyed" }),
    stopWorker: vi.fn(),
    endSession: vi.fn(),
    worker: vi.fn(),
  };
}
function mockedApi(api: ReturnType<typeof workerApi>) {
  vi.mocked(createWorldApi).mockReturnValue(api as unknown as ReturnType<typeof createWorldApi>);
}
describe("worker recovery cost controls", () => {
  it("only performs reads on entry and requires explicit destructive confirmation", async () => {
    const api = workerApi();
    mockedApi(api);
    render(<WorkerManager serverUrl="https://manager.example" />);
    await screen.findByText("pod-1");
    expect(api.request.mock.calls.map((call) => String(call[0]))).toEqual([
      "/workers",
      "/sessions",
    ]);
    expect(screen.getByText("GPU rate: $1.20/hr · $0.020/min")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Destroy worker" }));
    expect(api.destroyWorker).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Confirm destroy" }));
    await waitFor(() => expect(api.destroyWorker).toHaveBeenCalledExactlyOnceWith("pod-1"));
  });
  it("removes old worker actions immediately when the saved manager changes", async () => {
    const api = workerApi();
    mockedApi(api);
    const { rerender } = render(<WorkerManager serverUrl="https://first.example" />);
    await screen.findByText("pod-1");
    api.request.mockImplementation(() => new Promise(() => undefined));
    rerender(<WorkerManager serverUrl="https://second.example" />);
    expect(screen.queryByRole("button", { name: "Destroy worker" })).not.toBeInTheDocument();
    expect(screen.getByText("Checking retained workers…")).toBeVisible();
    expect(api.destroyWorker).not.toHaveBeenCalled();
  });
  it("does not offer stop or destroy during uncertain provisioning", async () => {
    const api = workerApi();
    api.request.mockImplementation((path: string) =>
      Promise.resolve(
        path === "/workers"
          ? {
              workers: [
                {
                  id: "pod-1",
                  provider: "runpod",
                  status: "unknown",
                  managed: true,
                  estimatedHourlyCost: null,
                },
              ],
            }
          : { sessions: [] },
      ),
    );
    mockedApi(api);
    render(<WorkerManager serverUrl="https://manager.example" />);
    await screen.findByText("pod-1");
    expect(screen.getByRole("button", { name: "Stop worker" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Destroy worker" })).toBeDisabled();
  });
});
