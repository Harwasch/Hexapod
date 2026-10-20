/**
 * As captured's page address (state/asCaptured.ts): `?asCaptured=1` turns it on for the visit,
 * nothing else does, and nothing is kept. Its own file: each case loads the store afresh
 * (`vi.resetModules`), which would hand later imports in the same file another store.
 */
import { describe, expect, it, vi } from "vitest";

describe("As captured from the page address", () => {
  it("is off unless the page address asks (`?asCaptured=1`), and not persisted", async () => {
    for (const [search, on] of [
      ["?asCaptured=1", true],
      ["?renderer=cesium&asCaptured=1", true],
      ["?asCaptured=0", false],
      ["?asCaptured=true", false],
      ["", false],
    ] as const) {
      window.history.replaceState(null, "", `/${search}`);
      vi.resetModules();
      const fresh = await import("@/state/asCaptured");
      expect(fresh.useAsCaptured.getState().on, search).toBe(on);
    }
    // Turned on, it leaves nothing on the device for the next visit.
    const { useAsCaptured } = await import("@/state/asCaptured");
    useAsCaptured.getState().set(true);
    for (let i = 0; i < window.localStorage.length; i += 1) {
      const key = window.localStorage.key(i) ?? "";
      expect(`${key}=${window.localStorage.getItem(key) ?? ""}`).not.toMatch(/asCaptured/i);
    }
    window.history.replaceState(null, "", "/");
    vi.resetModules();
  });
});
