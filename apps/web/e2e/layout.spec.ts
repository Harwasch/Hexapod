/**
 * The guard for "cards and panels overlap each other".
 *
 * The HUD is a grid of fixed regions (`.hud` in `app.css`, rules in `state/layout.ts`), so
 * no two floating surfaces should ever be drawn over each other, and none should hang off
 * the screen. This opens the panel combinations people actually reach — one tool panel at a
 * time, a selection with its feeds and the agent open, Plan and Fleet, the captures panel
 * mid-job with its phone QR code, the offline state — at a desktop, a laptop and a phone
 * size, and after each one asserts exactly that on the bounding boxes.
 *
 * A "surface" is any glass element in the HUD that is not inside another one, plus the data
 * credits. The map's own markers and zone chips are pinned to the world, not the screen, so
 * they are not surfaces. Modal sheets are checked separately: they cover the HUD on purpose,
 * behind a scrim, and only have to fit on screen.
 */
import type { Page } from "@playwright/test";

import { expect, mockApi, test, type MockOptions } from "./fixtures";

const VIEWPORTS = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "laptop", width: 1280, height: 720 },
  { name: "phone", width: 390, height: 844 },
] as const;

interface Surface {
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Reads every visible surface once the HUD has stopped moving. */
async function settledSurfaces(page: Page): Promise<Surface[]> {
  const read = () =>
    page.evaluate(() => {
      const candidates = Array.from(
        document.querySelectorAll<HTMLElement>(
          ".hud .glass, [data-testid='credits'] .cesium-viewer-bottom",
        ),
      ).filter((el) => !el.closest(".mc-overlays"));
      const opacity = (el: HTMLElement | null): number => {
        let value = 1;
        for (let node = el; node; node = node.parentElement)
          value *= Number(getComputedStyle(node).opacity);
        return value;
      };
      const visible = candidates.filter((el) => {
        const box = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return box.width > 1 && box.height > 1 && style.visibility !== "hidden";
      });
      return visible
        .filter((el) => !visible.some((other) => other !== el && other.contains(el)))
        .map((el) => {
          const box = el.getBoundingClientRect();
          const name =
            el.dataset.testid ??
            el.closest("[data-testid]")?.getAttribute("data-testid") ??
            el.getAttribute("aria-label") ??
            el.className;
          return {
            name,
            x: Math.round(box.x),
            y: Math.round(box.y),
            width: Math.round(box.width),
            height: Math.round(box.height),
            opacity: Math.round(opacity(el) * 100),
          };
        });
    });
  // Settled means no finite animation is running (enter/exit transitions; spinners loop
  // forever and are ignored) and two reads in a row agree, including opacity, so a panel
  // still fading out after its replacement arrived is neither counted nor missed.
  const animating = () =>
    page.evaluate(
      () =>
        document
          .getAnimations()
          .filter(
            (a) =>
              a.playState === "running" &&
              Number.isFinite(Number(a.effect?.getComputedTiming().endTime)),
          ).length,
    );
  let previous = "";
  for (let attempt = 0; attempt < 40; attempt++) {
    await page.waitForTimeout(300);
    if ((await animating()) > 0) continue;
    const next = JSON.stringify(await read());
    if (next === previous) break;
    previous = next;
  }
  const surfaces = JSON.parse(previous) as (Surface & { opacity: number })[];
  return surfaces.filter((surface) => surface.opacity > 5);
}

async function expectNoOverlap(page: Page, label: string): Promise<void> {
  const surfaces = await settledSurfaces(page);
  const viewport = page.viewportSize()!;
  const problems: string[] = [];
  for (const [i, a] of surfaces.entries()) {
    if (
      a.x < -1 ||
      a.y < -1 ||
      a.x + a.width > viewport.width + 1 ||
      a.y + a.height > viewport.height + 1
    )
      problems.push(`${a.name} leaves the screen (${a.x},${a.y} ${a.width}×${a.height})`);
    for (const b of surfaces.slice(i + 1)) {
      const w = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
      const h = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
      if (w > 1 && h > 1) problems.push(`${a.name} overlaps ${b.name} by ${w}×${h}`);
    }
  }
  expect(problems, `${label}: ${surfaces.map((s) => s.name).join(", ")}`).toEqual([]);
}

async function boot(page: Page, options: MockOptions = {}, onboarding = false): Promise<void> {
  await mockApi(page, options);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addInitScript((dismissed) => {
    window.localStorage.setItem(
      "twin.settings.v1",
      JSON.stringify({
        state: { onboardingDismissed: dismissed, quality: "performance", reducedMotion: true },
        version: 2,
      }),
    );
  }, !onboarding);
  await page.goto("/");
  await expect(page.getByTestId("status-bar")).toContainText("Alt", { timeout: 60_000 });
}

/** Presses a global shortcut with nothing focused, the way a person would from the map. */
async function shortcut(page: Page, key: string): Promise<void> {
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await page.keyboard.press(key);
}

async function loadDemo(page: Page): Promise<void> {
  await page.getByTestId("onboarding-demo").click();
  await expect(page.getByTestId("project-card")).toContainText("Blackrock Mesa", {
    timeout: 30_000,
  });
  await expect(page.getByTestId("onboarding")).toHaveCount(0);
}

const TOOLS = [
  ["l", "layers-panel"],
  ["s", "sites-panel"],
  ["u", "captures-panel"],
  ["m", "measure-panel"],
  ["c", "compare-panel"],
  ["b", "bookmarks-panel"],
] as const;

for (const viewport of VIEWPORTS) {
  test.describe(`no floating surfaces overlap at ${viewport.name} size`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test("welcome, each tool panel, and a dense map view", async ({ page }) => {
      test.setTimeout(240_000);
      await boot(page, {}, true);
      await expect(page.getByTestId("onboarding")).toBeVisible();
      await expectNoOverlap(page, "welcome");
      await loadDemo(page);
      await expectNoOverlap(page, "demo loaded");

      for (const [key, panel] of TOOLS) {
        await shortcut(page, key);
        await expect(page.getByTestId(panel)).toBeVisible();
        await expectNoOverlap(page, panel);
      }
      await shortcut(page, "b");

      // What a busy operator has open at once: a machine, its cameras, a panel, the agent.
      await page.getByTestId("command-input").fill("where is TR-04");
      await page.getByTestId("command-input").press("Enter");
      await expect(page.getByTestId("selection-card")).toBeVisible();
      await expectNoOverlap(page, "selection with the agent");
      await page.getByTestId("selection-card").getByRole("button", { name: "Camera" }).click();
      await expect(page.getByTestId("feeds-panel")).toBeVisible();
      await expectNoOverlap(page, "selection with feeds and agent");
      await shortcut(page, "l");
      await expect(page.getByTestId("layers-panel")).toBeVisible();
      await expectNoOverlap(page, "layers with selection, feeds and agent");
      if (viewport.name !== "phone") {
        await shortcut(page, "d");
        await expect(page.getByTestId("dev-panel")).toBeVisible();
        await expectNoOverlap(page, "everything in the right dock");
      }
    });

    test("plan and fleet windows, and the plan composer", async ({ page }) => {
      test.setTimeout(240_000);
      await boot(page, {}, true);
      await loadDemo(page);
      await shortcut(page, "l");
      await page.getByTestId("view-tab-plan").click();
      await expect(page.getByTestId("plans-panel")).toBeVisible();
      // The window and a tool panel share the left dock; one replaces the other.
      await expect(page.getByTestId("layers-panel")).toHaveCount(0);
      await expectNoOverlap(page, "plans");
      await page.getByTestId("plan-thistle").click();
      await expect(page.getByTestId("plan-detail")).toBeVisible();
      await expectNoOverlap(page, "plan detail");
      await page.getByTestId("view-tab-fleet").click();
      await expect(page.getByTestId("fleet-panel")).toBeVisible();
      await expectNoOverlap(page, "fleet");
      await page.getByTestId("command-input").fill("Mow Z-21 weekly with one mower");
      await page.getByTestId("command-input").press("Enter");
      await expect(page.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
      await expectNoOverlap(page, "plan composer with the agent open");
    });

    test("captures mid-job with the phone QR code, and the write-token prompt", async ({
      page,
    }) => {
      test.setTimeout(240_000);
      await boot(page, { workerRuns: true, writeToken: "letmein" });
      await shortcut(page, "u");
      await page.getByTestId("capture-file-input").setInputFiles({
        name: "orchard.mp4",
        mimeType: "video/mp4",
        buffer: Buffer.alloc(8, "x"),
      });
      await expect(page.getByTestId("write-token-form")).toBeVisible();
      await expectNoOverlap(page, "write-token prompt");
      await page.getByTestId("write-token-input").fill("letmein");
      await page.getByTestId("write-token-save").click();
      const card = page.getByTestId("capture-card-capture-1");
      await card.getByTestId("capture-process").click();
      await expect(card.getByTestId("capture-stage").first()).toBeVisible({ timeout: 30_000 });
      await expectNoOverlap(page, "captures with a running job");
      await page.getByTestId("capture-from-phone").click();
      await expect(page.getByTestId("handoff-qr")).toBeVisible();
      await expectNoOverlap(page, "captures with the phone QR code");
    });

    test("offline, with the captures panel and a sheet", async ({ page }) => {
      test.setTimeout(240_000);
      await boot(page, { apiDown: true });
      await expect(page.getByTestId("notice-api-offline")).toBeVisible();
      await expectNoOverlap(page, "offline");
      await shortcut(page, "u");
      await expect(page.getByTestId("captures-offline")).toBeVisible();
      await expectNoOverlap(page, "offline captures");

      await page.getByTestId("tool-settings").click();
      const sheet = page.getByTestId("settings-sheet");
      await expect(sheet).toBeVisible();
      const box = (await sheet.boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.y).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(viewport.width + 1);
      expect(box.y + box.height).toBeLessThanOrEqual(viewport.height + 1);
    });
  });
}
