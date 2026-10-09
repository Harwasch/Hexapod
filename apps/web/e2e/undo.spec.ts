/**
 * Undo and redo in the app (state/history.ts, features/shell/undoHotkeys.ts): the keys bound
 * by the shell and printed on the shortcut sheet, the line that says what was undone with the
 * step back beside it, a text field keeping its own Ctrl+Z, a layer switched in Layers
 * switched back, and a corner taken off an area on the map put back. The selection card's
 * Hide and Show only on a drawn scan are e2e/undoYard.spec.ts.
 */

import type { Page } from "@playwright/test";

import { expect, inApp, STAGED_SCAN, stageObjects, test } from "./fixtures";

/** Presses a key with nothing focused, the way a person would from the map. */
async function fromMap(app: Page, key: string): Promise<void> {
  await app.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await app.keyboard.press(key);
}

/** Somewhere with no site, the globe drawn under the whole view (as e2e/app.spec.ts has it). */
async function overFarmland(app: Page): Promise<void> {
  interface Twin {
    viewer: { canvas: HTMLCanvasElement };
    camera: {
      isMoving: boolean;
      cancelFlight: () => void;
      setView: (lon: number, lat: number, h: number, hd: number, p: number) => void;
    };
    scene: {
      camera: { getPickRay: (p: { x: number; y: number }) => unknown };
      globe: { pick: (ray: unknown, scene: unknown) => unknown };
    };
  }
  await app.evaluate(() => {
    const twin = (window as unknown as { __twin?: Twin }).__twin;
    twin?.camera.cancelFlight();
    twin?.camera.setView(-119.9, 36.6, 1500, 0, -60);
  });
  await expect
    .poll(
      () =>
        app.evaluate(() => {
          const twin = (window as unknown as { __twin?: Twin }).__twin;
          if (!twin || twin.camera.isMoving) return false;
          const { canvas } = twin.viewer;
          return [0.2, 0.5, 0.8].every((fx) =>
            [0.2, 0.5, 0.8].every((fy) => {
              const ray = twin.scene.camera.getPickRay({
                x: canvas.clientWidth * fx,
                y: canvas.clientHeight * fy,
              });
              return ray !== undefined && twin.scene.globe.pick(ray, twin.scene) !== undefined;
            }),
          );
        }),
      { timeout: 60_000 },
    )
    .toBe(true);
  await app.waitForTimeout(1000);
}

/** The area editor's corner handles the map shows, where they are on the screen. */
function corners(app: Page): Promise<{ id: string; x: number; y: number }[]> {
  return app.evaluate(() => {
    interface Twin {
      viewer: {
        canvas: HTMLCanvasElement;
        clock: { currentTime: unknown };
        entities: { values: { id: string; position?: { getValue: (t: unknown) => unknown } }[] };
      };
      scene: { cartesianToCanvasCoordinates: (p: unknown) => { x: number; y: number } | undefined };
    }
    const twin = (window as unknown as { __twin?: Twin }).__twin;
    if (!twin) return [];
    const rect = twin.viewer.canvas.getBoundingClientRect();
    const out: { id: string; x: number; y: number }[] = [];
    for (const entity of twin.viewer.entities.values) {
      if (!entity.id.startsWith("area-edit:vertex:")) continue;
      const position = entity.position?.getValue(twin.viewer.clock.currentTime);
      const at = position ? twin.scene.cartesianToCanvasCoordinates(position) : undefined;
      if (!at) continue;
      const x = rect.left + at.x;
      const y = rect.top + at.y;
      if (document.elementFromPoint(x, y) === twin.viewer.canvas) out.push({ id: entity.id, x, y });
    }
    return out;
  });
}

/** How many corners the area being edited has, on the map. */
function cornerCount(app: Page): Promise<number> {
  return app.evaluate(() => {
    const twin = (
      window as unknown as { __twin?: { viewer: { entities: { values: { id: string }[] } } } }
    ).__twin;
    return (twin?.viewer.entities.values ?? []).filter((e) => e.id.startsWith("area-edit:vertex:"))
      .length;
  });
}

function hidden(app: Page): Promise<number[]> {
  return inApp(
    app,
    `return [...(useInstances.getState().assets["${STAGED_SCAN}"]?.hidden ?? [])].sort((a, b) => a - b);`,
  );
}

test.describe("undo and redo", () => {
  test("Ctrl+Z takes back a hidden object and says so, Ctrl+Shift+Z hides it again, and a text field keeps its own", async ({
    app,
  }) => {
    await app.getByTestId("onboarding-explore").click();
    await stageObjects(app);
    const said = app
      .getByTestId("hud")
      .getByRole("status")
      .filter({ hasText: /Undid|Redid|Nothing/ });

    await fromMap(app, "Control+z");
    await expect(said).toHaveText("Nothing to undo");

    // The objects panel's eye on the conifer (2), as the panel calls the store.
    await inApp(app, `useInstances.getState().setObjectsHidden("${STAGED_SCAN}", [2], true);`);
    expect(await hidden(app)).toEqual([2, 3]);
    await fromMap(app, "Control+z");
    await expect.poll(() => hidden(app)).toEqual([]);
    await expect(said).toContainText("Undid: Hide Conifer");
    await fromMap(app, "Control+Shift+Z");
    await expect(said).toContainText("Redid: Hide Conifer");
    // The line's own button goes back again (at once: the line goes after a few seconds).
    await said.getByRole("button", { name: "Undo" }).click();
    await expect.poll(() => hidden(app)).toEqual([]);
    await fromMap(app, "Control+y");
    await expect.poll(() => hidden(app)).toEqual([2, 3]);
    await expect(said).toContainText("Redid: Hide Conifer");

    // In the command box Ctrl+Z is the field's: the words come back, nothing else changes.
    const input = app.getByRole("combobox", { name: "Search, run an action or ask the agent" });
    await input.click();
    await app.keyboard.type("conifer");
    await input.press("Control+a");
    await input.press("Backspace");
    await expect(input).toHaveValue("");
    await input.press("Control+z");
    await expect(input).toHaveValue("conifer");
    expect(await hidden(app)).toEqual([2, 3]);
    await expect(said).not.toContainText("Undid");
  });

  test("a layer switched in Layers is switched back by Ctrl+Z, and both keys are on the sheet", async ({
    app,
  }) => {
    await app.getByTestId("tool-layers").click();
    await expect(app.getByTestId("layers-panel")).toBeVisible();
    await app.getByTestId("layers-filter").fill("world");
    const card = app.getByTestId("layer-card-esa-worldcover-2021");
    const toggle = card.getByRole("switch");
    await expect(toggle).toHaveAttribute("aria-checked", "false");
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-checked", "true", { timeout: 20_000 });
    // From the switch itself: a switch has no text to take back.
    await app.keyboard.press("Control+z");
    await expect(toggle).toHaveAttribute("aria-checked", "false", { timeout: 20_000 });
    await expect(
      app.getByTestId("hud").getByRole("status").filter({ hasText: "Undid" }),
    ).toHaveText(/Undid: Show ESA WorldCover/);
    await app.keyboard.press("Control+Shift+Z");
    await expect(toggle).toHaveAttribute("aria-checked", "true", { timeout: 20_000 });

    await fromMap(app, "Shift+?");
    const general = app.getByTestId("shortcut-sheet").getByRole("region", { name: "General" });
    await expect(general).toContainText("Undo");
    await expect(general).toContainText(/Redo.*Shift.*Z.*or.*Y/);
  });

  test("a corner taken off an area on the map is put back by Ctrl+Z, and off again by Ctrl+Shift+Z", async ({
    app,
  }) => {
    // Software GL draws the globe under the view slowly (`overFarmland` waits for it).
    test.setTimeout(150_000);
    await app.getByTestId("onboarding-explore").click({ force: true });
    await overFarmland(app);
    await app.mouse.click(720, 450, { button: "right" });
    await app.getByRole("menuitem", { name: "Plan here" }).click();
    await expect(app.getByTestId("plan-ground")).toContainText("Test Field", { timeout: 20_000 });
    await expect.poll(async () => (await corners(app)).length).toBeGreaterThan(0);
    const before = await cornerCount(app);
    const [corner] = await corners(app);
    if (!corner) throw new Error("no corner on screen");
    // A right-click on a corner removes it.
    await app.mouse.click(corner.x, corner.y, { button: "right" });
    await expect.poll(() => cornerCount(app)).toBe(before - 1);
    await fromMap(app, "Control+z");
    await expect.poll(() => cornerCount(app)).toBe(before);
    await expect(
      app.getByTestId("hud").getByRole("status").filter({ hasText: "Undid" }),
    ).toContainText("Undid: Reshape");
    await fromMap(app, "Control+Shift+Z");
    await expect.poll(() => cornerCount(app)).toBe(before - 1);
  });
});
