import { describe, expect, it } from "vitest";

import { Handover } from "@/cesium/scanView/handover";

function setup(maxWaitMs = 1500) {
  const onScreen = new Set<string>();
  const drawn = new Set<string>();
  const handover = new Handover<string>(
    {
      add: (m) => onScreen.add(m),
      remove: (m) => onScreen.delete(m),
      isDrawn: (m) => drawn.has(m),
    },
    maxWaitMs,
  );
  return { onScreen, drawn, handover };
}

describe("Handover", () => {
  it("keeps a replaced tile on screen until its replacements have drawn", () => {
    const { onScreen, drawn, handover } = setup();
    handover.show("parent", 0);
    drawn.add("parent");
    handover.tick(10);
    handover.show("a", 20);
    handover.show("b", 20);
    handover.hide("parent", 20);
    handover.tick(30);
    expect([...onScreen].sort()).toEqual(["a", "b", "parent"]);
    drawn.add("a");
    handover.tick(40);
    expect(onScreen.has("parent")).toBe(true);
    drawn.add("b");
    handover.tick(50);
    expect([...onScreen].sort()).toEqual(["a", "b"]);
  });

  it("gives up waiting after the longest wait", () => {
    const { onScreen, drawn, handover } = setup(100);
    handover.show("parent", 0);
    drawn.add("parent");
    handover.tick(0);
    handover.show("child", 0);
    handover.hide("parent", 0);
    handover.tick(50);
    expect(onScreen.has("parent")).toBe(true);
    handover.tick(150);
    expect(onScreen.has("parent")).toBe(false);
  });

  it("a tile swapped back before it left is never re-added", () => {
    const adds: string[] = [];
    const drawn = new Set(["parent"]);
    const handover = new Handover<string>({
      add: (m) => adds.push(m),
      remove: () => undefined,
      isDrawn: (m) => drawn.has(m),
    });
    handover.show("parent", 0);
    handover.tick(0);
    handover.show("child", 1);
    handover.hide("parent", 1);
    handover.show("parent", 2);
    expect(adds).toEqual(["parent", "child"]);
    expect(handover.retained).toBe(0);
  });

  it("removes at once a tile that was never drawn, and forgets before disposal", () => {
    const { onScreen, handover } = setup();
    handover.show("a", 0);
    handover.hide("a", 1);
    expect(onScreen.has("a")).toBe(false);
    handover.show("b", 0);
    handover.tick(1);
    handover.show("c", 2);
    handover.hide("b", 2);
    handover.forget("b");
    expect(onScreen.has("b")).toBe(false);
  });
});
