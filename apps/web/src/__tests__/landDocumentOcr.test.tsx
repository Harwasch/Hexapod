import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "@twin/contracts";
import { api } from "@/api/client";
import { LandDocumentOcr } from "@/features/land/LandDocumentOcr";

const ocr: components["schemas"]["DocumentOcrRead"] = {
  id: "ocr",
  documentId: "deed",
  page: 2,
  language: "eng",
  text: "Mineral rights reserved.",
  sha256: "a".repeat(64),
  textSha256: "b".repeat(64),
  engine: "Tesseract",
  engineVersion: "Tesseract 5",
  renderMaxPixels: 2400,
  meanWordConfidence: 70,
  truncated: false,
  warnings: ["Verify machine readings against the original."],
  createdAt: "2026-10-09T12:00:00Z",
};
function mount(records: (typeof ocr)[] = [], pinnedId?: string) {
  vi.spyOn(api, "GET").mockImplementation(((path: string) =>
    Promise.resolve({
      data: path.endsWith("/ocr-capabilities")
        ? { available: true, languages: ["eng"], reason: "One page at a time." }
        : records,
      response: new Response(),
    })) as typeof api.GET);
  const selected = vi.fn();
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={cache}>
      <LandDocumentOcr
        landId="land"
        documentId="deed"
        page={2}
        pinnedId={pinnedId}
        onSelected={selected}
      />
    </QueryClientProvider>,
  );
  return selected;
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
it("processes only the selected page and exposes the resulting extraction for a question", async () => {
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: ocr, response: new Response() });
  const selected = mount();
  fireEvent.click(await screen.findByRole("button", { name: "Read page 2 with OCR" }));
  await screen.findByText("Mineral rights reserved.");
  expect(screen.getByText("Verify machine readings against the original.")).toBeVisible();
  await waitFor(() => expect(selected).toHaveBeenCalledWith("ocr"));
  const calls = post.mock.calls as unknown as [
    string,
    { body: unknown; params: { path: unknown } },
  ][];
  expect(calls[0]?.[1]).toMatchObject({
    params: { path: { land_id: "land", document_id: "deed", page: 2 } },
    body: { language: "eng" },
  });
});
it("identifies a missing cited extraction instead of silently substituting another reading", async () => {
  const post = vi.spyOn(api, "POST");
  mount([ocr], "missing");
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "cited OCR extraction is not available",
  );
  expect(screen.queryByText(/Cited machine reading/)).not.toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
});
