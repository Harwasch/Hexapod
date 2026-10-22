/**
 * As captured's page address (state/asCaptured.ts): `?asCaptured=1` turns it on for the visit,
 * nothing else does, the switch keeps the address in step, and nothing is kept on the device.
 * Its own file: each case loads the store afresh (`vi.resetModules`), which would hand later
 * imports in the same file another store.
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

  it("keeps the page address in step with the switch, replacing it rather than adding history", async () => {
    window.history.replaceState(null, "", "/?renderer=cesium#site=yard");
    vi.resetModules();
    const { useAsCaptured } = await import("@/state/asCaptured");
    const depth = window.history.length;
    useAsCaptured.getState().set(true);
    expect(new URLSearchParams(window.location.search).get("asCaptured")).toBe("1");
    expect(new URLSearchParams(window.location.search).get("renderer")).toBe("cesium");
    expect(window.location.hash).toBe("#site=yard");
    // Set again: the same address, still one entry.
    useAsCaptured.getState().set(true);
    expect(window.location.search).toBe("?renderer=cesium&asCaptured=1");
    useAsCaptured.getState().set(false);
    expect(window.location.search).toBe("?renderer=cesium");
    expect(window.location.hash).toBe("#site=yard");
    expect(window.history.length).toBe(depth);
    // The address it leaves is the one a reload or a shared link opens on.
    useAsCaptured.getState().set(true);
    vi.resetModules();
    const fresh = await import("@/state/asCaptured");
    expect(fresh.useAsCaptured.getState().on).toBe(true);
    window.history.replaceState(null, "", "/");
    vi.resetModules();
  });
});
