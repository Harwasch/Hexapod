import { describe, expect, it, vi } from "vitest";

import { runIntent, type IntentContext } from "@/lib/intents";

function ctx(overrides: Partial<IntentContext> = {}): IntentContext {
  return {
    flyToPlace: vi.fn((q: string) => Promise.resolve(`Flying to ${q}.`)),
    flyToSite: vi.fn(() => null),
    toggleLayer: vi.fn((q: string, v: boolean | null) =>
      q === "vegetation" ? `veg ${String(v)}` : null,
    ),
    selectMachine: vi.fn((id: string) => (id === "TR-07" ? "Locating TR-07." : null)),
    selectZone: vi.fn((id: string) => (id === "Z-14" ? "Zone Z-14." : null)),
    openPlan: vi.fn((q: string | null) => `plan ${q ?? "list"}`),
    setView: vi.fn((v: string) => `view ${v}`),
    startMeasure: vi.fn((m: string) => `measure ${m}`),
    camera: vi.fn((a: string) => `camera ${a}`),
    openSettings: vi.fn(() => "settings"),
    ...overrides,
  };
}

describe("runIntent", () => {
  it("routes machines, zones and layers", async () => {
    expect(await runIntent("where is TR-07", ctx())).toBe("Locating TR-07.");
    expect(await runIntent("show me tr-99", ctx())).toContain("don't know a machine");
    expect(await runIntent("open Z-14", ctx())).toBe("Zone Z-14.");
    expect(await runIntent("show vegetation", ctx())).toBe("veg true");
    expect(await runIntent("hide the vegetation layer", ctx())).toBe("veg false");
  });

  it("routes camera, measurement, views and settings", async () => {
    expect(await runIntent("measure the area", ctx())).toBe("measure area");
    expect(await runIntent("how far is it", ctx())).toBe("measure distance");
    expect(await runIntent("top-down", ctx())).toBe("camera top-down");
    expect(await runIntent("reset north", ctx())).toBe("camera north");
    expect(await runIntent("show the fleet", ctx())).toBe("view fleet");
    expect(await runIntent("open the plan for thistle", ctx())).toBe("plan thistle");
    expect(await runIntent("settings", ctx())).toBe("settings");
  });

  it("falls back to sites, then places", async () => {
    const c = ctx({
      flyToSite: vi.fn((q: string) => (q.includes("demo") ? "Flying to demo." : null)),
    });
    expect(await runIntent("fly to the demo site", c)).toBe("Flying to demo.");
    expect(await runIntent("fly to Yosemite", c)).toBe("Flying to yosemite.");
    expect(await runIntent("   ", c)).toContain("Say where");
  });
});
