import type { PropsWithChildren } from "react";
import { act, renderHook } from "@testing-library/react";
import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";

import { useHealth } from "@/api/queries";

const get = vi.hoisted(() => vi.fn());
vi.mock("@/api/client", () => ({
  api: { GET: get },
  unwrap: async (result: Promise<{ data: unknown }>) => (await result).data,
  isOffline: () => false,
}));

afterEach(() => {
  focusManager.setFocused(undefined);
  vi.useRealTimers();
  vi.restoreAllMocks();
});

it("pauses health polling while hidden and resumes when visible", async () => {
  vi.useFakeTimers();
  get.mockResolvedValue({ data: { status: "ok", database: true } });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const view = renderHook(useHealth, { wrapper });
  await act(() => vi.advanceTimersByTimeAsync(1));
  expect(get).toHaveBeenCalledTimes(1);
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
  await act(() => window.dispatchEvent(new Event("visibilitychange")));
  await act(() => vi.advanceTimersByTimeAsync(180_000));
  expect(get).toHaveBeenCalledTimes(1);
  vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible");
  await act(() => window.dispatchEvent(new Event("visibilitychange")));
  await act(() => vi.advanceTimersByTimeAsync(60_001));
  expect(get.mock.calls.length).toBeGreaterThan(1);
  view.unmount();
  client.clear();
});
