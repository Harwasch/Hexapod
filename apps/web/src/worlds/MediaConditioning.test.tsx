import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MediaConditioning } from "./MediaConditioning";
import { postIntelligence } from "./intelligence/client";
import type { MediaAsset, WorldProject } from "./core/types";

const store = vi.hoisted(() => ({ getBlob: vi.fn(), saveAsset: vi.fn() }));
vi.mock("./core/storage", () => ({ worldStore: store }));
vi.mock("./intelligence/client", () => ({
  postIntelligence: vi.fn(),
  imageData: () => Promise.resolve("data:image/jpeg;base64,reference"),
  dataImageBlob: () => new Blob(["generated"], { type: "image/jpeg" }),
}));
const project: WorldProject = {
  id: "world",
  name: "Jungle",
  prompt: "A jungle tower",
  modelId: "matrix-game-3",
  providerId: "runpod",
  createdAt: 1,
  updatedAt: 1,
  assetIds: ["original"],
  characterIds: [],
  settings: { performance: "balanced" },
};
const asset: MediaAsset = {
  id: "original",
  name: "Photo",
  kind: "image",
  mimeType: "image/jpeg",
  size: 20,
  createdAt: 1,
};
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
describe("starting-frame preparation", () => {
  it("requires explicit sharing consent and preserves provenance when replacing one reference", async () => {
    store.getBlob.mockResolvedValue(new Blob(["reference"], { type: "image/jpeg" }));
    store.saveAsset.mockResolvedValue({ ...asset, id: "prepared" });
    vi.mocked(postIntelligence).mockResolvedValue({ image: "data:image/jpeg;base64,generated" });
    const onChange = vi.fn();
    render(
      <MediaConditioning project={project} assets={[asset]} serverUrl="" onChange={onChange} />,
    );
    const button = screen.getByRole("button", { name: "Compose references into a starting frame" });
    expect(button).toBeDisabled();
    expect(postIntelligence).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(button);
    await waitFor(() => expect(onChange).toHaveBeenCalledOnce());
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        assetIds: ["prepared"],
        referencePreparation: { method: "image-synthesis", sourceAssetIds: ["original"] },
      }),
    );
  });
  it("does not overwrite an edited world with a late image-service response", async () => {
    store.getBlob.mockResolvedValue(new Blob(["reference"], { type: "image/jpeg" }));
    let finish: ((result: { image: string }) => void) | undefined;
    vi.mocked(postIntelligence).mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const onChange = vi.fn();
    const { rerender } = render(
      <MediaConditioning project={project} assets={[asset]} serverUrl="" onChange={onChange} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(
      screen.getByRole("button", { name: "Compose references into a starting frame" }),
    );
    await waitFor(() => expect(postIntelligence).toHaveBeenCalledOnce());
    rerender(
      <MediaConditioning
        project={{ ...project, prompt: "A different scene" }}
        assets={[asset]}
        serverUrl=""
        onChange={onChange}
      />,
    );
    await act(async () => {
      finish?.({ image: "late" });
      await Promise.resolve();
    });
    expect(onChange).not.toHaveBeenCalled();
    expect(store.saveAsset).not.toHaveBeenCalled();
  });
});
