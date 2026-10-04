/**
 * Words asked of the agent while it is answering wait their turn (useAgentCommand).
 *
 * The command box closes and clears as a row runs, and a sentence of work keeps the agent
 * busy until its draft is in. A second ask meanwhile -- a refinement typed while the plan was
 * being drafted -- used to be dropped without a word.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { useAgentCommand } from "@/features/mission/useAgentCommand";
import type * as Intents from "@/lib/intents";
import { useMission } from "@/state/mission";

const intents = vi.hoisted(() => ({ runIntent: vi.fn<(input: string) => Promise<string>>() }));
vi.mock("@/lib/intents", async (importOriginal) => ({
  ...(await importOriginal<typeof Intents>()),
  runIntent: intents.runIntent,
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("asking the agent while it answers", () => {
  beforeEach(() => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useMission.setState({ log: [] });
    intents.runIntent.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  it("answers every ask, in order, and none is dropped", async () => {
    let finishDraft: (reply: string) => void = () => undefined;
    intents.runIntent
      .mockImplementationOnce(
        () =>
          new Promise<string>((resolve) => {
            finishDraft = resolve;
          }),
      )
      .mockImplementation((input: string) => Promise.resolve(`Done: ${input}.`));
    const { result } = renderHook(() => useAgentCommand(), { wrapper });

    act(() => void result.current.ask("mow Z-21 by Friday"));
    await waitFor(() => expect(result.current.busy).toBe(true));
    // Typed while the draft is being written: heard at once, answered after it.
    act(() => void result.current.ask("two mowers"));
    act(() => void result.current.ask("  "));
    expect(intents.runIntent).toHaveBeenCalledTimes(1);
    expect(useMission.getState().log.map((e) => `${e.role}: ${e.text}`)).toEqual([
      "you: mow Z-21 by Friday",
      "you: two mowers",
    ]);

    act(() => finishDraft("Drafting a plan."));
    await waitFor(() => expect(result.current.busy).toBe(false));
    expect(intents.runIntent.mock.calls.map((call) => call[0])).toEqual([
      "mow Z-21 by Friday",
      "two mowers",
    ]);
    expect(useMission.getState().log.map((e) => `${e.role}: ${e.text}`)).toEqual([
      "you: mow Z-21 by Friday",
      "you: two mowers",
      "agent: Drafting a plan.",
      "agent: Done: two mowers.",
    ]);
  });
});
