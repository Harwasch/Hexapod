import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { CustomizationPanel } from "./CustomizationPanel";
import { createWorldApi } from "../core/api";
vi.mock("../core/api", () => ({ createWorldApi: vi.fn() }));
const request = vi.fn();
const caps = {
  configured: true,
  trainingEnabled: true,
  maxTrainingSeconds: 3600,
  maxConcurrentJobs: 1,
  profiles: [],
  message: "Ready",
};
const job = {
  id: "safe-job",
  profileId: "approved",
  modelId: "forge-wm",
  stage: "stage3-student",
  status: "prepared",
  createdAt: 1,
  activationCompatible: true,
  gpuValidated: false,
  artifacts: [],
  installed: [],
  message: "Prepared locally",
};
beforeEach(() => {
  vi.mocked(createWorldApi).mockReturnValue({ request } as unknown as ReturnType<
    typeof createWorldApi
  >);
  request.mockImplementation((path: string) =>
    Promise.resolve(
      path.endsWith("capabilities") ? caps : path.endsWith("/jobs") ? { jobs: [job] } : job,
    ),
  );
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
it("requires explicit compute consent before posting a training request", async () => {
  render(<CustomizationPanel serverUrl="" />);
  const run = await screen.findByRole("button", { name: "Run approved training" });
  expect(run).toBeDisabled();
  expect(
    request.mock.calls.every(
      (call) => (call[1] as { method?: string } | undefined)?.method !== "POST",
    ),
  ).toBe(true);
  fireEvent.click(screen.getByLabelText(/I authorize a training job/));
  expect(run).toBeEnabled();
  fireEvent.click(run);
  await waitFor(() =>
    expect(request).toHaveBeenCalledWith(
      "/customization/jobs/safe-job/run",
      expect.objectContaining({ method: "POST", body: '{"confirmTraining":true}' }),
    ),
  );
});
it("shows malformed worker responses without crashing the application", async () => {
  request.mockResolvedValue({ providers: [] });
  render(<CustomizationPanel serverUrl="" />);
  expect(await screen.findByRole("alert")).toHaveTextContent("invalid capabilities");
  expect(screen.queryByRole("button", { name: "Run approved training" })).not.toBeInTheDocument();
});
it("drops old jobs and consent immediately when changing workers", async () => {
  const { rerender } = render(<CustomizationPanel serverUrl="https://first.example" />);
  await screen.findByRole("button", { name: "Run approved training" });
  fireEvent.click(screen.getByLabelText(/I authorize a training job/));
  request.mockImplementation(() => new Promise(() => undefined));
  rerender(<CustomizationPanel serverUrl="https://second.example" />);
  expect(screen.queryByRole("button", { name: "Run approved training" })).not.toBeInTheDocument();
});
it("operator-disabled execution cannot be enabled by browser consent", async () => {
  request.mockImplementation((path: string) =>
    Promise.resolve(
      path.endsWith("capabilities") ? { ...caps, trainingEnabled: false } : { jobs: [job] },
    ),
  );
  render(<CustomizationPanel serverUrl="" />);
  const run = await screen.findByRole("button", { name: "Run approved training" });
  fireEvent.click(screen.getByLabelText(/I authorize a training job/));
  expect(run).toBeDisabled();
});
