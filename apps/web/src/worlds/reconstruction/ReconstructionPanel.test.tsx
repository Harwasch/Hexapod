import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReconstructionPanel } from "./ReconstructionPanel";
import { reconstructionGateway } from "./gateway";

vi.mock("./gateway", () => ({
  reconstructionGateway: {
    capabilities: vi.fn(),
    submit: vi.fn(),
    status: vi.fn(),
    delete: vi.fn(),
  },
}));
vi.mock("./LocalSplatViewer", () => ({ LocalSplatViewer: () => <div>Local geometry viewer</div> }));

beforeEach(() => {
  vi.stubGlobal(
    "URL",
    class extends URL {
      static override createObjectURL = vi.fn(() => "blob:local-test");
      static override revokeObjectURL = vi.fn();
    },
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

function mount(onSave = vi.fn().mockResolvedValue(undefined)) {
  return render(
    <ReconstructionPanel
      projects={[]}
      replays={[]}
      assets={[]}
      reconstructions={[]}
      getAssetBlob={vi.fn()}
      onSave={onSave}
    />,
  );
}

describe("reconstruction consent and offline behavior", () => {
  it("keeps source export available offline without submitting any remote job", async () => {
    vi.mocked(reconstructionGateway.capabilities).mockRejectedValue(new Error("Offline"));
    mount();
    await screen.findByText(/Reconstruction service is offline/);
    fireEvent.change(screen.getByLabelText("Add frames or a recording"), {
      target: { files: [new File(["frame"], "frame.png", { type: "image/png" })] },
    });
    expect(screen.getByRole("button", { name: "Export source bundle .tar" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Reconstruct selected views" })).toBeDisabled();
    expect(reconstructionGateway.submit).not.toHaveBeenCalled();
  });

  it("requires explicit media consent and submits neutral filenames with honest job state", async () => {
    vi.mocked(reconstructionGateway.capabilities).mockResolvedValue({
      configured: true,
      maxUploadBytes: 64 * 1024 * 1024,
      message: "Worker connected",
    });
    vi.mocked(reconstructionGateway.submit).mockResolvedValue({
      id: "job-1",
      status: "completed",
      artifacts: [],
    });
    const onSave = vi.fn().mockResolvedValue(undefined);
    mount(onSave);
    await screen.findByText("Worker connected");
    fireEvent.change(screen.getByLabelText("Add frames or a recording"), {
      target: { files: [new File(["frame"], "alice-private.png", { type: "image/png" })] },
    });
    const submit = screen.getByRole("button", { name: "Reconstruct selected views" });
    expect(submit).toBeDisabled();
    fireEvent.click(screen.getByLabelText(/Send only selected media/));
    fireEvent.click(submit);
    await waitFor(() => expect(onSave).toHaveBeenCalledOnce());
    const form = vi.mocked(reconstructionGateway.submit).mock.calls[0]?.[1];
    expect(form).toBeInstanceOf(FormData);
    expect((form?.get("files") as File).name).toBe("00001.png");
    expect(form?.get("metadata")).not.toContain("alice-private");
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ status: "completed", jobId: "job-1", artifacts: [] }),
    );
  });

  it("opens a local 3D file without uploading it", async () => {
    vi.mocked(reconstructionGateway.capabilities).mockResolvedValue({
      configured: false,
      maxUploadBytes: 0,
      message: "Unavailable",
    });
    mount();
    fireEvent.change(screen.getByLabelText("Open 3D file"), {
      target: { files: [new File(["ply"], "place.ply")] },
    });
    await screen.findByText("Local geometry viewer");
    expect(reconstructionGateway.submit).not.toHaveBeenCalled();
  });

  it("refreshes signed diagnostics and refuses unsafe report links", async () => {
    vi.mocked(reconstructionGateway.capabilities).mockResolvedValue({
      configured: true,
      maxUploadBytes: 64 * 1024 * 1024,
      message: "Worker connected",
    });
    vi.mocked(reconstructionGateway.status).mockResolvedValue({
      id: "job-report",
      status: "completed",
      artifacts: [],
      diagnosticsUrl: "https://worker.example/report?signature=scoped",
      camerasUrl: "javascript:alert(1)",
    });
    render(
      <ReconstructionPanel
        projects={[]}
        replays={[]}
        assets={[]}
        reconstructions={[
          {
            id: "local-report",
            jobId: "job-report",
            projectId: "local-import",
            name: "Place",
            createdAt: 1,
            status: "completed",
            format: "ply",
            sourceAssetIds: [],
          },
        ]}
        getAssetBlob={vi.fn()}
        onSave={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Refresh status" }));
    expect(
      await screen.findByRole("link", { name: "Download reconstruction diagnostics" }),
    ).toHaveAttribute("href", "https://worker.example/report?signature=scoped");
    expect(
      screen.queryByRole("link", { name: "Download estimated cameras" }),
    ).not.toBeInTheDocument();
  });
});
