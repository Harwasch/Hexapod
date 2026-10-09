import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { components, LandArea } from "@twin/contracts";
import { api } from "@/api/client";
import { LandDocuments } from "@/features/land/LandDocuments";
import { LandDocumentViewer } from "@/features/land/LandDocumentViewer";
import { useLandContext } from "@/state/landContext";

const land = { id: "land" } as LandArea;
const record: components["schemas"]["LandDocumentRead"] = {
  id: "deed",
  landId: "land",
  title: "County deed",
  filename: "deed.txt",
  mediaType: "text/plain",
  sizeBytes: 4,
  kind: "deed",
  sourceNote: "County archive",
  status: "ready",
  sha256: "a".repeat(64),
  pageCount: 2,
  extractedCharacters: 30,
  warnings: [],
  createdAt: "2026-10-09T12:00:00Z",
};
function mount(child = <LandDocuments land={land} />, pending = false) {
  const row = pending ? { ...record, status: "awaiting-upload", pageCount: 0 } : record;
  const get = vi.spyOn(api, "GET").mockImplementation(((
    path: string,
    options: { params: { path: { page?: number } } },
  ) => {
    const data = path.endsWith("/documents")
      ? [row]
      : path.endsWith("/links")
        ? []
        : path.endsWith("/pages/{page}")
          ? {
              documentId: "deed",
              page: options.params.path.page,
              text: `Recorded passage on page ${options.params.path.page}`,
              sha256: record.sha256,
              truncated: false,
            }
          : row;
    return Promise.resolve({ data, response: new Response() });
  }) as typeof api.GET);
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(<QueryClientProvider client={cache}>{child}</QueryClientProvider>);
  return get;
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useLandContext.getState().clear();
});

it("retries an interrupted file transfer against the same document without creating another record", async () => {
  const post = vi.spyOn(api, "POST").mockResolvedValue({ data: record, response: new Response() });
  const put = vi
    .spyOn(api, "PUT")
    .mockRejectedValueOnce(new Error("Connection lost"))
    .mockResolvedValue({ data: record, response: new Response() });
  mount();
  fireEvent.click(screen.getByText("Add a land record"));
  fireEvent.change(screen.getByLabelText("Original PDF or text file"), {
    target: { files: [new File(["deed"], "deed.txt", { type: "text/plain" })] },
  });
  fireEvent.change(screen.getByLabelText("Where did this record come from?"), {
    target: { value: "County archive" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save and read record" }));
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Save and read record" }));
  await screen.findByRole("region", { name: "Document page" });
  expect(post).toHaveBeenCalledOnce();
  expect(put).toHaveBeenCalledTimes(2);
  const calls = put.mock.calls as unknown as [
    string,
    { params: { path: unknown }; bodySerializer: () => File },
  ][];
  expect(calls[0]?.[1].params.path).toEqual(calls[1]?.[1].params.path);
  expect(calls[1]?.[1].bodySerializer().name).toBe("deed.txt");
});

it("uses the viewed page for document relationships and a reviewable agent question", async () => {
  mount();
  fireEvent.click(await screen.findByRole("button", { name: /County deed.*Text ready/ }));
  fireEvent.change(await screen.findByLabelText("Document page number"), {
    target: { value: "2" },
  });
  await screen.findByText("Recorded passage on page 2");
  fireEvent.click(screen.getByRole("button", { name: "Relate this record to another" }));
  expect(screen.getByLabelText("Source page")).toHaveValue(2);
  fireEvent.click(screen.getByRole("button", { name: "Ask about this page" }));
  expect(useLandContext.getState().researchQuestion).toContain("document deed, page 2");
  expect(useLandContext.getState().section).toBe("discover");
});

it("resumes a pending upload after reload and checks file identity before transferring", async () => {
  const put = vi.spyOn(api, "PUT").mockResolvedValue({ data: record, response: new Response() });
  mount(<LandDocumentViewer landId="land" documentId="deed" />, true);
  const input = await screen.findByLabelText("Resume original upload");
  fireEvent.change(input, { target: { files: [new File(["wrong"], "other.txt")] } });
  expect(await screen.findByRole("alert")).toHaveTextContent("Choose the original deed.txt");
  expect(put).not.toHaveBeenCalled();
  fireEvent.change(input, { target: { files: [new File(["deed"], "deed.txt")] } });
  await waitFor(() => expect(put).toHaveBeenCalledOnce());
  await screen.findByText("Recorded passage on page 1");
  expect(screen.queryByLabelText("Resume original upload")).not.toBeInTheDocument();
});
