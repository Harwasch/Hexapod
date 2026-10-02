import { describe, expect, it } from "vitest";

import { FADE_MS, Handover } from "@/cesium/scanView/handover";

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

  it("fades new detail in, and takes the old off only once it is drawn and in", () => {
    const onScreen = new Set<string>();
    const drawn = new Set<string>();
    const alpha = new Map<string, number>();
    const handover = new Handover<string>({
      add: (m) => onScreen.add(m),
      remove: (m) => onScreen.delete(m),
      isDrawn: (m) => drawn.has(m),
      fade: (m, a) => alpha.set(m, a),
    });
    handover.show("parent", 0);
    drawn.add("parent");
    handover.tick(FADE_MS);
    expect(alpha.get("parent")).toBe(1);
    handover.show("child", 1000);
    handover.hide("parent", 1000);
    expect(alpha.get("child")).toBe(0);
    handover.tick(1000 + FADE_MS / 2);
    expect(alpha.get("child")).toBeCloseTo(0.5);
    handover.tick(1000 + FADE_MS);
    expect(alpha.get("child")).toBe(1);
    // Faded in, but not drawn yet: the parent stays.
    expect(onScreen.has("parent")).toBe(true);
    drawn.add("child");
    handover.tick(1000 + FADE_MS + 16);
    expect(onScreen.has("parent")).toBe(false);
  });

  it("says what the next frames need: a frame per fade step, one after a retire, a deadline", () => {
    const drawn = new Set<string>();
    const handover = new Handover<string>(
      {
        add: () => undefined,
        remove: () => undefined,
        isDrawn: (m) => drawn.has(m),
        fade: () => undefined,
      },
      1000,
    );
    handover.show("parent", 0);
    drawn.add("parent");
    // Fading in: a frame every display frame until it is in.
    expect(handover.tick(10)).toEqual({ changed: true, animating: true, nextAt: null });
    expect(handover.tick(FADE_MS)).toEqual({ changed: true, animating: false, nextAt: null });
    // Settled and in: nothing more to draw, however long it rests.
    expect(handover.tick(5000)).toEqual({ changed: false, animating: false, nextAt: null });
    expect(handover.busy).toBe(false);
    handover.show("child", 6000);
    handover.hide("parent", 6000);
    // The child fades in; the parent waits for it, and gives up at the longest wait.
    const step = handover.tick(6000 + FADE_MS);
    expect(step.animating).toBe(false);
    expect(step.nextAt).toBe(6000 + 1000 + 1);
    // Not drawn yet: waiting costs no frames (the renderer asks for one once it has drawn).
    expect(handover.tick(6500)).toEqual({ changed: false, animating: false, nextAt: 7001 });
    drawn.add("child");
    // Drawn: the parent comes off, and that change wants one more frame.
    expect(handover.tick(6600)).toEqual({ changed: true, animating: false, nextAt: null });
    expect(handover.busy).toBe(false);
  });
});
