import { readFileSync } from "node:fs";
import type * as LiveSpark from "@/view/liveSpark";
import type * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { afterEach, expect, it, vi } from "vitest";
import { showLive } from "@/view/live";
import { fetchApi } from "@/api/coldStart";

const state = vi.hoisted(() => ({
  render: vi.fn(),
  disposeRenderer: vi.fn(),
  disposeSpark: vi.fn(),
  stopConsole: vi.fn(),
  updateConsole: vi.fn(),
  loop: null as ((time: number) => void) | null,
  spark: null as unknown as { sorting: boolean },
}));
vi.mock("@/api/coldStart", () => ({ fetchApi: vi.fn() }));
vi.mock("@/view/liveConsole", () => ({
  liveConsole: () => ({ stop: state.stopConsole, update: state.updateConsole }),
}));
vi.mock("three", async () => {
  const three = await vi.importActual<typeof THREE>("three");
  return {
    ...three,
    WebGLRenderer: class {
      domElement = document.createElement("canvas");
      setPixelRatio = vi.fn();
      setSize = vi.fn();
      setClearColor = vi.fn();
      render = state.render;
      dispose = state.disposeRenderer;
      setAnimationLoop(callback: ((time: number) => void) | null) {
        state.loop = callback;
      }
    },
  };
});
vi.mock("@/view/liveSpark", async () => {
  const actual = await vi.importActual<typeof LiveSpark>("@/view/liveSpark");
  const three = await vi.importActual<typeof THREE>("three");
  return {
    ...actual,
    consoleSpark: () => {
      const spark = Object.assign(new three.Group(), {
        autoUpdate: true,
        enableDriveLod: true,
        sorting: true,
        sortDirty: true,
        updateTimeoutId: -1,
        sortTimeoutId: -1,
        sortWorker: null,
        lodWorker: null,
        dispose: state.disposeSpark,
      });
      state.spark = spark;
      return spark;
    },
  };
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  document.body.replaceChildren();
});

it("stops UI immediately, ignores a late poll body and releases GPU only after its readback", async () => {
  vi.useFakeTimers();
  document.body.innerHTML = readFileSync("view.html", "utf8");
  const disposeControls = vi.spyOn(OrbitControls.prototype, "dispose");
  let resolveBody!: (body: unknown) => void;
  const body = new Promise<unknown>((resolve) => {
    resolveBody = resolve;
  });
  const json = vi.fn(() => body);
  vi.mocked(fetchApi).mockResolvedValue({ ok: true, status: 200, json } as unknown as Response);
  const viewer = showLive("c1");
  await Promise.resolve();
  expect(json).toHaveBeenCalled();
  const queuedFrame = state.loop!;
  viewer.stop();
  viewer.stop();
  expect(state.loop).toBeNull();
  expect(document.querySelector("#live")?.getAttribute("hidden")).not.toBeNull();
  expect(document.querySelector('#live canvas[aria-label="Training viewer"]')).toBeNull();
  expect(disposeControls).toHaveBeenCalledTimes(1);
  expect(state.stopConsole).toHaveBeenCalledTimes(1);
  expect(vi.mocked(fetchApi).mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
  expect(state.disposeSpark).not.toHaveBeenCalled();
  expect(state.disposeRenderer).not.toHaveBeenCalled();
  resolveBody({ captureName: "late response", status: "complete" });
  await Promise.resolve();
  queuedFrame(100);
  document.dispatchEvent(new Event("visibilitychange"));
  await vi.advanceTimersByTimeAsync(100);
  expect(document.getElementById("live-name")?.textContent).toBe("Live");
  expect(state.render).not.toHaveBeenCalled();
  expect(state.updateConsole).not.toHaveBeenCalled();
  expect(fetchApi).toHaveBeenCalledTimes(1);
  document.getElementById("live")!.dataset.tiles = "new-session";
  state.spark.sorting = false;
  await vi.advanceTimersByTimeAsync(32);
  expect(state.disposeSpark).toHaveBeenCalledTimes(1);
  expect(state.disposeRenderer).toHaveBeenCalledTimes(1);
  expect(document.getElementById("live")!.dataset.tiles).toBe("new-session");
});
