import { afterEach, expect, it, vi } from "vitest";
import {
  classifyDevice,
  deviceSplatBudget,
  deviceSplatCeiling,
  OPTIONS_STORAGE,
} from "@/lib/detail";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  localStorage.clear();
});
it("requires all recent-iPhone signals and keeps unknown handhelds conservative", () => {
  const signals = {
    handheld: true,
    iphone: true,
    shortSide: 390,
    longSide: 844,
    pixelRatio: 3,
    cores: 6,
    maxTextureSize: 16384,
  };
  expect(classifyDevice(signals)).toBe("recentIPhone");
  for (const change of [
    { iphone: false },
    { shortSide: 375 },
    { pixelRatio: 2 },
    { cores: 2 },
    { maxTextureSize: 8192 },
  ])
    expect(classifyDevice({ ...signals, ...change })).toBe("handheld");
  expect(classifyDevice({ ...signals, handheld: false })).toBe("desktop");
});
it("maps saved Light/Standard/Full to each class and preserves desktop memory budgets", () => {
  for (const [i, choice] of [200000, 400000, 800000].entries()) {
    localStorage.setItem(OPTIONS_STORAGE, JSON.stringify({ detail: choice }));
    expect(deviceSplatBudget("recentIPhone")).toBe([750000, 1500000, 2500000][i]);
    expect(deviceSplatBudget("handheld")).toBe(choice);
    expect(deviceSplatBudget("desktop")).toBe([1500000, 3000000, 6000000][i]);
    expect(deviceSplatCeiling("recentIPhone")).toBe([1500000, 3000000, 3500000][i]);
    expect(deviceSplatCeiling("handheld")).toBe(choice * 2);
  }
});

it("retains desktop Standard budgets at lower reported memory and defaults invalid Detail", () => {
  localStorage.setItem(OPTIONS_STORAGE, JSON.stringify({ detail: "invalid" }));
  for (const [deviceMemory, expected] of [
    [8, 3000000],
    [4, 1500000],
    [2, 800000],
  ]) {
    vi.stubGlobal("navigator", { deviceMemory });
    expect(deviceSplatBudget("desktop")).toBe(expected);
  }
  expect(deviceSplatBudget("handheld")).toBe(400000);
  expect(deviceSplatBudget("recentIPhone")).toBe(1500000);
});
