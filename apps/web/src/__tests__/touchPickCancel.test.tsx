/**
 * Waiting for a point on the map on a phone: a phone has no Escape key, so the words never
 * say "Esc" there, and the plan card has a Cancel that ends the wait on any device.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import { PlanCard } from "@/features/mission/PlanCard";
import { planFromText } from "@/features/mission/planFlow";
import { anywhereProject } from "@/missions/anywhere";
import { useMission } from "@/state/mission";

const scene = vi.hoisted(() => {
  let settle: ((point: null) => void) | null = null;
  return {
    areas: {
      pickGround: vi.fn(
        () =>
          new Promise<null>((done) => {
            settle = done;
          }),
      ),
      cancelPick: vi.fn(() => settle?.(null)),
      edit: vi.fn(),
      editing: null,
    },
    events: { on: () => () => undefined },
  };
});
vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

function touchScreen(coarse: boolean) {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: coarse && query === "(pointer: coarse)",
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}

const asScene = scene as unknown as CesiumSceneManager;

beforeEach(() => {
  vi.spyOn(api, "GET").mockResolvedValue({
    data: { configured: false, provider: "rules", model: null },
    response: new Response(),
  });
  useMission.setState({ composer: null, landGround: null });
  useMission.getState().setProject(anywhereProject());
});

afterEach(() => {
  cleanup();
  scene.areas.cancelPick();
  useMission.setState({ composer: null });
  vi.restoreAllMocks();
});

describe("waiting for the ground under a finger", () => {
  it("says tap and points at Cancel on a phone, and keeps Esc for a keyboard", async () => {
    touchScreen(true);
    const touch = await planFromText("3D scan this field", asScene);
    expect(touch).toBe(
      "Tap the ground you mean on the map and I'll outline it, or press Cancel on the plan.",
    );
    expect(useMission.getState().composer?.status).toBe("awaiting-ground");
    scene.areas.cancelPick();

    touchScreen(false);
    expect(await planFromText("3D scan this field", asScene)).toBe(
      "Click the ground you mean on the map and I'll outline it. Esc to cancel.",
    );
  });

  it("gives the plan card a Cancel that ends the wait, and no Esc on a phone", async () => {
    touchScreen(true);
    await planFromText("3D scan this field", asScene);
    render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <PlanCard project={useMission.getState().project!} />
      </QueryClientProvider>,
    );
    const prompt = screen.getByTestId("plan-awaiting-ground");
    expect(prompt).toHaveTextContent("Tap the ground you mean on the map.");
    expect(prompt).not.toHaveTextContent(/Esc/);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
      await Promise.resolve();
    });
    await waitFor(() => expect(useMission.getState().composer?.status).toBe("idle"));
    expect(screen.queryByTestId("plan-awaiting-ground")).not.toBeInTheDocument();
  });
});
