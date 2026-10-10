import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import type { ResearchEvent } from "@twin/contracts";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/api/client";
import {
  mergeResearchEvents,
  readResearchStream,
  ResearchAccessError,
} from "@/features/land/researchStream";
import { useResearchProgress } from "@/features/land/useResearchProgress";
const event = (sequence: number): ResearchEvent => ({
  sequence,
  kind: "progress",
  payload: { message: "Café 🐝" },
  createdAt: "2026-10-09T00:00:00Z",
});
const frame = (sequence: number) =>
  `id: ${sequence}\r\nevent: research\r\ndata: ${JSON.stringify(event(sequence))}\r\n\r\n`;
function stream(text: string, byteChunks = false) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      if (byteChunks) for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      else controller.enqueue(bytes);
      controller.close();
    },
  });
}
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
it("reassembles UTF-8 and CRLF across every byte boundary and ignores heartbeats", async () => {
  const received: ResearchEvent[] = [];
  await readResearchStream(
    stream(": keepalive\r\n\r\n" + frame(1) + frame(2), true),
    new AbortController().signal,
    (item) => received.push(item),
  );
  expect(received).toEqual([event(1), event(2)]);
});
it("does not accept a truncated last frame or a mismatched cursor", async () => {
  const received: ResearchEvent[] = [];
  await readResearchStream(
    stream(frame(1) + frame(2).slice(0, -2)),
    new AbortController().signal,
    (item) => received.push(item),
  );
  expect(received).toEqual([event(1)]);
  await expect(
    readResearchStream(
      stream(frame(2).replace("id: 2", "id: 3")),
      new AbortController().signal,
      vi.fn(),
    ),
  ).rejects.toThrow("cursor mismatch");
});
it("rejects revoked access and oversized updates and cancels a pending read", async () => {
  await expect(
    readResearchStream(
      stream("event: access-revoked\ndata: {}\n\n"),
      new AbortController().signal,
      vi.fn(),
    ),
  ).rejects.toBeInstanceOf(ResearchAccessError);
  await expect(
    readResearchStream(
      stream("data: " + "x".repeat(1_048_576)),
      new AbortController().signal,
      vi.fn(),
    ),
  ).rejects.toThrow("size limit");
  const cancelled = vi.fn(),
    controller = new AbortController();
  const reading = readResearchStream(
    new ReadableStream({ cancel: cancelled }),
    controller.signal,
    vi.fn(),
  );
  controller.abort();
  await reading;
  expect(cancelled).toHaveBeenCalledOnce();
});
it("deduplicates replayed updates and retains a bounded ordered history", () => {
  const merged = mergeResearchEvents(
    Array.from({ length: 2100 }, (_, i) => event(i + 1)),
    [event(2100), event(2101)],
  );
  expect(merged).toHaveLength(2000);
  expect(merged[0]?.sequence).toBe(102);
  expect(merged.at(-1)?.sequence).toBe(2101);
});
function mount(cache: QueryClient) {
  return renderHook(() => useResearchProgress("run", true), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={cache}>{children}</QueryClientProvider>
    ),
  });
}
it("resumes from cached progress and reconnects from the last received sequence", async () => {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  cache.setQueryData(["land-research", "pilot", "events", "run"], [event(1)]);
  const cursors: number[] = [],
    cancelled = vi.fn();
  vi.spyOn(api, "GET").mockImplementation((path, options) => {
    const after = (options as { params: { query: { after: number } } }).params.query.after;
    if (String(path).endsWith("/stream")) {
      cursors.push(after);
      return Promise.resolve({
        data: cursors.length === 1 ? stream(frame(2)) : new ReadableStream({ cancel: cancelled }),
        response: new Response(null, { headers: { "content-type": "text/event-stream" } }),
      });
    }
    return Promise.resolve({ data: [], response: new Response() });
  });
  const mounted = mount(cache);
  await waitFor(() => expect(mounted.result.current.data).toEqual([event(1), event(2)]));
  await waitFor(() => expect(cursors).toEqual([1, 2]), { timeout: 2000 });
  mounted.unmount();
  await waitFor(() => expect(cancelled).toHaveBeenCalledOnce());
  cache.clear();
});
it("falls back to incremental reads when streaming is unsupported", async () => {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const afterValues: number[] = [];
  vi.spyOn(api, "GET").mockImplementation((path, options) => {
    if (String(path).endsWith("/stream"))
      return Promise.resolve({ error: {}, response: new Response(null, { status: 405 }) });
    const after = (options as { params: { query: { after: number } } }).params.query.after;
    afterValues.push(after);
    return Promise.resolve({ data: after === 0 ? [event(1)] : [], response: new Response() });
  });
  const mounted = mount(cache);
  await waitFor(() => expect(mounted.result.current.phase).toBe("polling"));
  expect(afterValues).toEqual([0, 1]);
  expect(mounted.result.current.data).toEqual([event(1)]);
  mounted.unmount();
  cache.clear();
});
it("clears revoked progress and does not reconnect until explicitly retried", async () => {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const get = vi.spyOn(api, "GET").mockImplementation((path) =>
    Promise.resolve(
      String(path).endsWith("/stream")
        ? {
            data: stream("event: access-revoked\ndata: {}\n\n"),
            response: new Response(null, { headers: { "content-type": "text/event-stream" } }),
          }
        : { data: [event(1)], response: new Response() },
    ),
  );
  const mounted = mount(cache);
  await waitFor(() => expect(mounted.result.current.phase).toBe("unavailable"));
  expect(mounted.result.current.data).toEqual([]);
  expect(get).toHaveBeenCalledTimes(2);
  act(() => mounted.result.current.retry());
  await waitFor(() => expect(get).toHaveBeenCalledTimes(4));
  mounted.unmount();
  cache.clear();
});
