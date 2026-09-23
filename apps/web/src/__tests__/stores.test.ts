import { describe, expect, it } from "vitest";

import { useLayers } from "@/state/layers";
import { useMeasurements } from "@/state/measurements";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

describe("stores", () => {
  it("layer runtime updates and removals are immutable", () => {
    useLayers.getState().ensure("a", { visible: true });
    useLayers.getState().update("a", { loadState: "ready" });
    expect(useLayers.getState().runtime.a).toMatchObject({
      visible: true,
      loadState: "ready",
      opacity: 1,
    });
    useLayers.getState().remove("a");
    expect(useLayers.getState().runtime.a).toBeUndefined();
  });

  it("site runtime tracks representation and assets", () => {
    useSites.getState().setRepresentation("s1", "mesh");
    useSites.getState().updateAsset("x", { loadState: "loading" });
    expect(useSites.getState().representation.s1).toBe("mesh");
    expect(useSites.getState().assets.x?.loadState).toBe("loading");
    useSites.getState().clearAssets(["x"]);
    expect(useSites.getState().assets.x).toBeUndefined();
  });

  it("ui panel toggling is exclusive and measurement mode opens the panel", () => {
    const ui = useUi.getState();
    ui.togglePanel("layers");
    expect(useUi.getState().activePanel).toBe("layers");
    ui.togglePanel("layers");
    expect(useUi.getState().activePanel).toBeNull();
    ui.setMeasureMode("distance");
    expect(useUi.getState().activePanel).toBe("measure");
  });

  it("measurements upsert by id", () => {
    const m = {
      id: "m1",
      mode: "distance" as const,
      points: [],
      complete: true,
      createdAt: 1,
      distance2dM: 1,
    };
    useMeasurements.getState().upsert(m);
    useMeasurements.getState().upsert({ ...m, distance2dM: 2 });
    expect(useMeasurements.getState().items).toHaveLength(1);
    expect(useMeasurements.getState().items[0]?.distance2dM).toBe(2);
    useMeasurements.getState().clear();
  });

  it("settings persist defaults and reset", () => {
    useSettings.getState().set({ quality: "ultra", units: "imperial" });
    expect(useSettings.getState().quality).toBe("ultra");
    useSettings.getState().reset();
    expect(useSettings.getState().quality).toBe("balanced");
    expect(useSettings.getState().units).toBe("metric");
  });
});
