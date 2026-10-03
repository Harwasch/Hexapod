/**
 * The guard for "cards and panels overlap each other".
 *
 * The HUD is a grid of fixed regions (`.hud` in `app.css`, rules in `state/layout.ts`), so
 * no two floating surfaces should ever be drawn over each other, and none should hang off
 * the screen. This opens the panel combinations people actually reach — one tool panel at a
 * time, a selection with its feeds and the agent open, the Plan / Fleet drawer with a machine
 * picked from it, Add mid-job with its phone QR code, the offline state — at a desktop, a
 * laptop and a phone size, and after each one asserts exactly that on the bounding boxes.
 *
 * A "surface" is any glass element in the HUD that is not inside another one, plus the data
 * credits. The map's own markers and zone chips are pinned to the world, not the screen, so
 * they are not surfaces. Modal sheets are checked separately: they cover the HUD on purpose,
 * behind a scrim, and only have to fit on screen. So do the popovers (`[data-hud-popover]`:
 * the command box's results, the site switcher, the agent's activity log, the phone's More
 * sheet): they open over the regions when asked for and close with Escape or a click away,
 * so they are held to the screen's edges but not to the regions'. The data credits are never
 * a popover: on a phone they are a strip of their own, held to the same rules as the rest.
 */
import type { Page } from "@playwright/test";

import { addGoogleCredit, expect, mockApi, test, type MockOptions } from "./fixtures";

const VIEWPORTS = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "laptop", width: 1280, height: 720 },
  { name: "phone", width: 390, height: 844 },
] as const;

interface Surface {
  name: string;
  popover: boolean;
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
            popover: el.closest("[data-hud-popover]") !== null,
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
      if (a.popover || b.popover) continue;
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
  await expect(page.getByTestId("status-line")).toBeVisible({ timeout: 60_000 });
}

/** Presses a global shortcut with nothing focused, the way a person would from the map. */
async function shortcut(page: Page, key: string): Promise<void> {
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await page.keyboard.press(key);
}

/** Map / Plan / Fleet: the tabs at the top right, or on a phone the bar at the bottom. */
async function showView(page: Page, view: "map" | "plan" | "fleet"): Promise<void> {
  const phone = page.getByTestId(`phone-tab-${view}`);
  if (await phone.isVisible()) await phone.click();
  else await page.getByTestId(`view-tab-${view}`).click();
}

/** A rail tool, or on a phone the same tool under More. */
async function tool(page: Page, id: "layers" | "measure" | "add" | "settings"): Promise<void> {
  const more = page.getByTestId("phone-tab-more");
  if (await more.isVisible()) {
    await more.click();
    await page.getByTestId(`more-${id}`).click();
  } else await page.getByTestId(`tool-${id}`).click();
}

/** Says something in the command box; on a phone its search button opens it first. */
async function ask(page: Page, text: string): Promise<void> {
  const trigger = page.getByTestId("command-trigger");
  if (await trigger.isVisible()) await trigger.click();
  await page.getByTestId("command-input").fill(text);
  await page.getByTestId("command-input").press("Enter");
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
  ["c", "compare-controls"],
  ["u", "add-panel"],
  ["m", "measure-panel"],
  ["s", "site-switcher"],
  ["v", "site-switcher"],
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
        await expectNoOverlap(page, `${key}: ${panel}`);
      }
      await shortcut(page, "v");
      await expect(page.getByTestId("site-switcher")).toHaveCount(0);
      await shortcut(page, "m");
      await expect(page.getByTestId("measure-panel")).toHaveCount(0);

      // What a busy operator has open at once: a machine, its cameras, a panel, the agent.
      await ask(page, "where is TR-04");
      await expect(page.getByTestId("selection-card")).toBeVisible();
      await expectNoOverlap(page, "selection with the agent");
      await page.getByTestId("selection-card").getByRole("button", { name: "Camera" }).click();
      await expect(page.getByTestId("feeds-panel")).toBeVisible();
      await expectNoOverlap(page, "selection with feeds and agent");
      await shortcut(page, "l");
      await expect(page.getByTestId("layers-panel")).toBeVisible();
      await expectNoOverlap(page, "layers with selection, feeds and agent");
      // The agent's full log opens above the status line and stays on screen.
      await shortcut(page, "a");
      await expect(page.getByTestId("agent-stream")).toBeVisible();
      await expectNoOverlap(page, "agent activity log");
      await shortcut(page, "a");
      await expect(page.getByTestId("agent-stream")).toHaveCount(0);
      if (viewport.name !== "phone") {
        await shortcut(page, "d");
        await expect(page.getByTestId("dev-panel")).toBeVisible();
        await expectNoOverlap(page, "everything in the right dock");
      }
    });

    test("the plan and fleet drawer, a machine picked from it, and the plan composer", async ({
      page,
    }) => {
      test.setTimeout(240_000);
      await boot(page, {}, true);
      await loadDemo(page);
      await shortcut(page, "l");
      await showView(page, "plan");
      await expect(page.getByTestId("plans-panel")).toBeVisible();
      // A tool panel and the drawer replace each other, so the map between them keeps its width.
      await expect(page.getByTestId("layers-panel")).toHaveCount(0);
      await expectNoOverlap(page, "plans");
      await page.getByTestId("plan-thistle").click();
      await expect(page.getByTestId("plan-detail")).toBeVisible();
      await expectNoOverlap(page, "plan detail");
      await showView(page, "fleet");
      await expect(page.getByTestId("fleet-panel")).toBeVisible();
      await expectNoOverlap(page, "fleet");
      // Several rows of the table show, on a phone too.
      const rows = await page.locator('[data-testid^="fleet-row-"]').evaluateAll((els) => {
        const sheet = document.querySelector(".hud-drawer .mc-window__body");
        const bottom = sheet?.getBoundingClientRect().bottom ?? 0;
        return els.filter((el) => el.getBoundingClientRect().bottom <= bottom + 1).length;
      });
      expect(rows).toBeGreaterThanOrEqual(viewport.name === "phone" ? 3 : 5);
      // A machine picked from the table: its card opens beside the drawer, which stays.
      await page.getByTestId("fleet-row-TR-04").click();
      await expect(page.getByTestId("selection-card")).toContainText("TR-04 Kestrel");
      await expectNoOverlap(page, "fleet with a machine selected");
      if (viewport.name !== "phone") await expect(page.getByTestId("fleet-panel")).toBeVisible();
      else {
        // One sheet on a phone: the card is in front; the Fleet tab brings the table back.
        await showView(page, "fleet");
        await expect(page.getByTestId("fleet-panel")).toBeVisible();
      }
      await ask(page, "Mow Z-21 weekly with one mower");
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

      await tool(page, "settings");
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

test.describe("the phone layout", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("tab bar, More, full-screen search, one-row status and the credits strip", async ({
    page,
  }) => {
    test.setTimeout(240_000);
    await boot(page, {}, true);
    await loadDemo(page);
    // Map / Plan / Fleet / More along the bottom; the rail and the view tabs give way to it.
    await expect(page.getByTestId("phone-tabs")).toBeVisible();
    await expect(page.getByTestId("tool-layers")).toBeHidden();
    await expect(page.getByTestId("view-tabs")).toBeHidden();
    const tabs = await page.getByTestId("phone-tabs").boundingBox();
    const status = await page.getByTestId("status-line").boundingBox();
    const corner = await page.getByTestId("map-corner").boundingBox();
    if (!tabs || !status || !corner) throw new Error("the phone's bottom rows are not laid out");
    // The status line is one row, directly above the tab bar, with the compass beside it.
    expect(status.y + status.height).toBeLessThanOrEqual(tabs.y);
    expect(status.height).toBeLessThan(56);
    expect(Math.abs(corner.y - status.y)).toBeLessThan(12);
    await expect(page.getByTestId("nav-north")).toBeVisible();

    // It shows less than it says, and what it shows is whole: the counts and the simulated
    // tag, then the agent's spinner and "+N". The labels, the agent's sentence and the word
    // after "+N" are out of view but not out of the live regions or the button's name.
    const fleet = page.getByTestId("status-fleet");
    await expect(fleet.getByText("simulated")).toBeVisible();
    await expect(page.getByTestId("status-line").getByRole("status")).toContainText(
      "Fleet: 4 working · 2 need attention simulated",
    );
    await expect(page.getByTestId("status-agent")).toHaveText(/^Agent: \S.{9,}$/);
    await expect(page.locator(".status-line__more")).toHaveText(/^\+\d+ tasks?$/);
    const shown = await page.evaluate(() => {
      const width = (selector: string) =>
        document.querySelector(selector)?.getBoundingClientRect().width ?? -1;
      const clip = document.querySelector("[data-testid='status-fleet'] .status-line__clip")!;
      return {
        counts: (clip.lastChild?.textContent ?? "").trim(),
        clipped: clip.scrollWidth > clip.clientWidth,
        prefix: width(".status-line__prefix"),
        sentence: width("[data-testid='status-agent'] .status-line__clip"),
        word: width(".status-line__more-word"),
        more: width(".status-line__more"),
      };
    });
    expect(shown).toMatchObject({ counts: "4 working · 2 need attention", clipped: false });
    expect(shown.prefix).toBeLessThanOrEqual(1);
    expect(shown.sentence).toBeLessThanOrEqual(1);
    expect(shown.word).toBeLessThanOrEqual(1);
    expect(shown.more).toBeGreaterThan(8);

    // The data credits are on screen with no tap: a strip of their own, one thin line at the
    // right edge just above the status line, the Cesium ion logo and "Data attribution" in
    // view, over nothing. Google's logo, when Photorealistic 3D Tiles are on, joins the line.
    const chip = page.getByTestId("credits").locator(".cesium-viewer-bottom");
    await expect(chip).toBeVisible();
    await expect(chip.locator(".cesium-credit-logoContainer img")).toBeVisible();
    await expect(page.getByRole("button", { name: /credits/i })).toHaveCount(0);
    const expectStrip = async (label: string) => {
      const strip = (await chip.boundingBox())!;
      expect(strip.height).toBeLessThan(30);
      expect(strip.y + strip.height).toBeLessThanOrEqual(status.y);
      expect(Math.abs(strip.x + strip.width - (corner.x + corner.width))).toBeLessThan(2);
      await expectNoOverlap(page, label);
    };
    await expectStrip("phone credits strip");
    await addGoogleCredit(page);
    await expect(chip.getByRole("img", { name: "Google" })).toBeVisible();
    await expect
      .poll(async () => (await chip.getByRole("img", { name: "Google" }).boundingBox())?.width)
      .toBe(98);
    await expectStrip("phone credits strip with Google's logo");
    // "Data attribution" opens the dialog with every credit in full.
    await chip.getByRole("button", { name: "Data attribution" }).click();
    const dialog = page.getByRole("dialog", { name: "Data attribution" });
    await expect(dialog).toBeVisible();
    await page.getByRole("button", { name: "Close data attribution" }).click();
    await expect(dialog).toBeHidden();

    // More holds the four tools.
    await page.getByTestId("phone-tab-more").click();
    await expect(page.getByTestId("phone-more").getByRole("button")).toHaveText([
      "Layers",
      "Measure",
      "Add",
      "Settings",
    ]);
    await expectNoOverlap(page, "phone More sheet");
    await page.getByTestId("more-layers").click();
    await expect(page.getByTestId("layers-panel")).toBeVisible();
    await expect(page.getByTestId("phone-more")).toHaveCount(0);

    // Search is a button that opens the command box over the whole screen.
    await expect(page.getByTestId("command-input")).toBeHidden();
    await page.getByTestId("command-trigger").click();
    const input = page.getByTestId("command-input");
    await expect(input).toBeFocused();
    const box = await page.getByTestId("command-box").boundingBox();
    expect(box?.width).toBeGreaterThanOrEqual(388);
    expect(box?.height).toBeGreaterThanOrEqual(840);
    await expect(page.getByTestId("command-results")).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(input).toBeHidden();
  });
});
