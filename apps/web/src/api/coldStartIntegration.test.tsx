import { act, cleanup, render, renderHook, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { PropsWithChildren } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type * as EnvModule from "@/app/env";
import { useCaptures as useAdminCaptures } from "@/admin/queries";
import { Toasts } from "@/features/notices/Toasts";
import { useSites } from "./queries";
import { apiWaking } from "./waking";
import { API_RETRY_BUDGET_MS, fetchApi } from "./coldStart";

vi.mock("@/app/env", async (original) => {
  const actual = await original<typeof EnvModule>();
  return {
    ...actual,
    env: {
      ...actual.env,
      apiBaseUrl: "https://api.invalid",
      offlineCatalogUrl: "https://catalog.invalid/catalog.json",
    },
  };
});

const transport = vi.fn<typeof fetch>();
let cache: QueryClient;
function Wrapper({ children }: PropsWithChildren) {
  return <QueryClientProvider client={cache}>{children}</QueryClientProvider>;
}
beforeEach(async () => {
  vi.useFakeTimers();
  // Even a parent with Query's default retries must not multiply the catalog budget.
  cache = new QueryClient({ defaultOptions: { queries: { gcTime: 0 } } });
  transport.mockReset();
  vi.stubGlobal("fetch", transport);
  transport.mockResolvedValueOnce(new Response());
  await fetchApi("https://api.invalid/api/v1/health/live");
  transport.mockReset();
});
afterEach(async () => {
  cleanup();
  cache.clear();
  await vi.advanceTimersByTimeAsync(0);
  expect(apiWaking.getSnapshot()).toBe(false);
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("shows a calm notice during retries and clears it when the catalog answers", async () => {
  transport
    .mockResolvedValueOnce(new Response(null, { status: 503 }))
    .mockResolvedValueOnce(Response.json([]));
  const { result } = renderHook(useSites, { wrapper: Wrapper });
  render(<Toasts />);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  await act(() => vi.advanceTimersByTimeAsync(1));
  expect(screen.getByRole("status")).toHaveTextContent(
    "Waking the server… this takes a few seconds after a quiet spell",
  );
  expect(result.current.builtin).toBe(false);
  expect(result.current.isLoading).toBe(true);
  await act(() => vi.advanceTimersByTimeAsync(1001));
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  expect(result.current.data).toEqual([]);
  expect(result.current.builtin).toBe(false);
});

it.each(["network", "timeout", "503"])(
  "falls back only after the %s budget, without query retries",
  async (failure) => {
    transport.mockImplementation(async (input) => {
      if (input === "https://catalog.invalid/catalog.json") return Response.json({ sites: [] });
      if (failure === "503") return Response.json({ detail: "asleep" }, { status: 503 });
      if (failure === "timeout") return new Promise(() => undefined);
      throw new TypeError("offline");
    });
    const { result } = renderHook(useSites, { wrapper: Wrapper });
    render(<Toasts />);
    await act(() => vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS - 1));
    expect(result.current.builtin).toBe(false);
    expect(transport.mock.calls.every(([input]) => input instanceof Request)).toBe(true);
    await act(() => vi.advanceTimersByTimeAsync(3));
    expect(result.current.builtin).toBe(true);
    expect(result.current.data.length).toBeGreaterThan(0);
    expect(
      transport.mock.calls.some(([input]) => input === "https://catalog.invalid/catalog.json"),
    ).toBe(true);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    const calls = transport.mock.calls.length;
    await act(() => vi.advanceTimersByTimeAsync(2000));
    expect(transport).toHaveBeenCalledTimes(calls);
    transport.mockResolvedValue(Response.json([]));
    await act(() => vi.advanceTimersByTimeAsync(3002));
    expect(result.current.builtin).toBe(false);
    expect(result.current.data).toEqual([]);
  },
);

it.each([useSites, useAdminCaptures])("cancels a catalog retry on unmount", async (hook) => {
  transport.mockRejectedValue(new TypeError("offline"));
  const view = renderHook(
    () => {
      hook();
    },
    { wrapper: Wrapper },
  );
  await act(() => vi.advanceTimersByTimeAsync(1));
  expect(apiWaking.getSnapshot()).toBe(true);
  view.unmount();
  await act(() => vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS));
  expect(apiWaking.getSnapshot()).toBe(false);
  expect(transport).toHaveBeenCalledTimes(1);
});

it("uses the same retry transport for the admin's first GET", async () => {
  transport
    .mockResolvedValueOnce(new Response(null, { status: 503 }))
    .mockResolvedValueOnce(Response.json([]));
  const { result } = renderHook(useAdminCaptures, { wrapper: Wrapper });
  await act(() => vi.advanceTimersByTimeAsync(1002));
  expect(result.current.data).toEqual([]);
  expect(transport).toHaveBeenCalledTimes(2);
});

it("keeps the built-in ranch available to existing and new observers throughout recovery", async () => {
  transport.mockImplementation((input) => {
    if (input === "https://catalog.invalid/catalog.json")
      return Promise.resolve(Response.json({ sites: [] }));
    return Promise.reject(new TypeError("offline"));
  });
  const { result } = renderHook(useSites, { wrapper: Wrapper });
  render(<Toasts />);
  await act(() => vi.advanceTimersByTimeAsync(API_RETRY_BUDGET_MS + 2));
  expect(result.current.builtin).toBe(true);

  let answer!: (response: Response) => void;
  transport.mockImplementation(() => new Promise<Response>((resolve) => (answer = resolve)));
  await act(() => vi.advanceTimersByTimeAsync(5000));
  expect(result.current.data?.some((site) => site.slug === "demo-ranch")).toBe(true);
  expect(result.current.builtin).toBe(true);
  expect(result.current.isLoading).toBe(false);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  const newcomer = renderHook(useSites, { wrapper: Wrapper });
  expect(newcomer.result.current.data?.some((site) => site.slug === "demo-ranch")).toBe(true);
  expect(newcomer.result.current.isLoading).toBe(false);

  await act(async () => {
    answer(Response.json([]));
    await vi.advanceTimersByTimeAsync(1);
  });
  expect(result.current.builtin).toBe(false);
  expect(result.current.data).toEqual([]);
  expect(newcomer.result.current.data).toEqual([]);
});
