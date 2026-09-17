import type { Page } from "@playwright/test";

import { expect, mockApi, test } from "./fixtures";

/** Ids of the plan overlay entities the map is drawing (passes, step markers, route). */
function planEntityIds(app: Page): Promise<string[]> {
  return app.evaluate(() => {
    const twin = (
      window as unknown as { __twin?: { viewer: { entities: { values: { id: string }[] } } } }
    ).__twin;
    return (twin?.viewer.entities.values ?? [])
      .map((entity) => entity.id)
      .filter((id) => id.startsWith("mission:plan:"))
      .sort();
  });
}

test.describe("boot", () => {
  test("app boots, viewer initializes and onboarding shows", async ({ app }) => {
    await expect(app.getByTestId("onboarding")).toBeVisible();
    await expect(app.getByRole("heading", { name: "Explore the living world" })).toBeVisible();
    await expect(app.getByTestId("status-bar")).toContainText("Planet");
    const errors: string[] = [];
    app.on("pageerror", (e) => errors.push(e.message));
    await app.waitForTimeout(500);
    expect(errors).toEqual([]);
  });

  test("API failure does not crash the viewer", async ({ page }) => {
    await mockApi(page, { apiDown: true });
    await page.goto("/");
    await expect(page.locator("canvas").first()).toBeVisible();
    await expect(page.getByTestId("notice-api-offline")).toBeVisible();
    await page.getByTestId("tool-sites").click();
    await expect(page.getByTestId("site-card-cesium-splat-demo")).toBeVisible();
    await expect(page.getByText("Built-in demo (API offline)")).toBeVisible();
  });
});

test.describe("catalog", () => {
  test("site catalog loads and fly-to-site works", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    await app.getByTestId("tool-sites").click();
    await expect(app.getByTestId("sites-panel")).toBeVisible();
    const card = app.getByTestId("site-card-cesium-splat-demo");
    await expect(card).toContainText("Cesium Gaussian splat demo");
    await expect(card).toContainText("Splat");
    await card.click();
    await expect(app.getByTestId("representation-switcher")).toBeVisible({ timeout: 30_000 });
    await expect(app.getByTestId("status-bar")).not.toContainText("Planet");
  });

  test("representation switch keeps the camera", async ({ app }) => {
    await app.getByTestId("onboarding-demo").click();
    const switcher = app.getByTestId("representation-switcher");
    await expect(switcher).toBeVisible({ timeout: 30_000 });
    await expect(app.getByTestId("status-bar")).toContainText("Splat");
    const readCamera = () =>
      app.evaluate(() => {
        interface Handle {
          viewer: {
            camera: {
              positionWC: { x: number; y: number; z: number };
              heading: number;
              pitch: number;
            };
          };
        }
        const camera = (window as unknown as { __twin?: Handle }).__twin?.viewer.camera;
        if (!camera) throw new Error("scene handle missing");
        return {
          x: camera.positionWC.x,
          y: camera.positionWC.y,
          z: camera.positionWC.z,
          heading: camera.heading,
          pitch: camera.pitch,
        };
      });
    // Wait for the arrival flight to settle before sampling the pose.
    await expect
      .poll(
        async () => {
          const a = await readCamera();
          await app.waitForTimeout(600);
          const b = await readCamera();
          return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
        },
        { timeout: 30_000 },
      )
      .toBeLessThan(0.01);
    const before = await readCamera();
    await switcher.getByRole("radio", { name: /Mesh/ }).click();
    await expect(app.getByTestId("status-bar")).toContainText("Mesh");
    await app.waitForTimeout(800);
    // Cesium re-normalises the camera frame every frame; compare within float noise.
    const after = await readCamera();
    expect(Math.hypot(after.x - before.x, after.y - before.y, after.z - before.z)).toBeLessThan(
      1e-6,
    );
    expect(Math.abs(after.heading - before.heading)).toBeLessThan(1e-9);
    expect(Math.abs(after.pitch - before.pitch)).toBeLessThan(1e-9);
    await expect(switcher.getByRole("radio", { name: /Points/ })).toBeDisabled();
  });

  test("layer toggle updates runtime state and About sheet shows provenance", async ({ app }) => {
    await app.getByTestId("tool-layers").click();
    await expect(app.getByTestId("layers-panel")).toBeVisible();
    await app.getByTestId("layers-filter").fill("world");
    const card = app.getByTestId("layer-card-esa-worldcover-2021");
    await expect(card).toBeVisible();
    await expect(app.getByTestId("layer-card-openstreetmap")).toHaveCount(0);
    await card.getByRole("switch").click();
    await expect(card.getByRole("switch")).toHaveAttribute("aria-checked", "true", {
      timeout: 20_000,
    });
    await expect(card).toContainText("Opacity", { timeout: 20_000 });
    await card.getByRole("button", { name: /About ESA WorldCover/ }).click();
    const about = app.getByTestId("layer-about");
    await expect(about).toBeVisible();
    await expect(about).toContainText("CC BY 4.0");
    await expect(about).toContainText("© ESA WorldCover project 2021");
    await expect(about).toContainText("Tree cover");
  });
});

test.describe("interaction", () => {
  test("clicking the world opens the inspector", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    await app.waitForTimeout(3500);
    await app.mouse.click(720, 450);
    await expect(app.getByTestId("inspector")).toBeVisible({ timeout: 15_000 });
    await expect(app.getByTestId("inspector-position")).toContainText("°");
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("inspector")).toHaveCount(0);
  });

  test("measurement tool can be entered and exited", async ({ app }) => {
    await app.getByTestId("tool-measure").click();
    await expect(app.getByTestId("measure-panel")).toBeVisible();
    await app.getByTestId("measure-distance").click();
    await expect(app.getByTestId("measure-active")).toContainText("Two clicks");
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("measure-active")).toHaveCount(0);
    await app.getByTestId("measure-area").click();
    await expect(app.getByTestId("measure-active")).toBeVisible();
    await app.getByTestId("measure-area").click();
    await expect(app.getByTestId("measure-active")).toHaveCount(0);
  });

  test("command palette and keyboard shortcuts drive the UI", async ({ app }) => {
    await app.keyboard.press("Control+k");
    const palette = app.getByTestId("command-palette");
    await expect(palette).toBeVisible();
    await app.getByTestId("palette-input").fill("settings");
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("settings-sheet")).toBeVisible();
    await expect(app.getByRole("radiogroup", { name: "Quality preset" })).toBeVisible();
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("settings-sheet")).toHaveCount(0);
    await app.keyboard.press("l");
    await expect(app.getByTestId("layers-panel")).toBeVisible();
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("layers-panel")).toHaveCount(0);
  });

  test("major UI is keyboard accessible", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    await app.getByTestId("search-input").focus();
    // Tab through the top-right controls to the tool rail; every stop must be a real control.
    for (let i = 0; i < 12; i++) {
      await app.keyboard.press("Tab");
      if (await app.getByTestId("tool-layers").evaluate((el) => el === document.activeElement))
        break;
    }
    await expect(app.getByTestId("tool-layers")).toBeFocused();
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("layers-panel")).toBeVisible();
    await app.keyboard.press("Tab");
    await expect(app.getByTestId("tool-sites")).toBeFocused();
    const rail = app.getByRole("toolbar", { name: "Tools" });
    await expect(rail.getByRole("button")).toHaveCount(8);
    for (const button of await rail.getByRole("button").all()) {
      expect(await button.getAttribute("aria-label")).toBeTruthy();
    }
  });
});

test.describe("add data", () => {
  test("validates input before submitting and persists a site", async ({ page }) => {
    let created: unknown = null;
    await mockApi(page, { onCreateSite: (body) => (created = body) });
    await page.addInitScript(() =>
      window.localStorage.setItem(
        "twin.settings.v1",
        JSON.stringify({ state: { onboardingDismissed: true }, version: 1 }),
      ),
    );
    await page.goto("/");
    await expect(page.getByTestId("status-bar")).toContainText("Alt");
    await page.getByTestId("tool-add-data").click();
    const sheet = page.getByTestId("add-data");
    await expect(sheet).toBeVisible();
    await page.getByTestId("site-submit").click();
    await expect(sheet.getByText("Give it a name")).toBeVisible();
    await expect(sheet.getByText("Paste or upload a GeoJSON Polygon footprint.")).toBeVisible();
    await expect(sheet.getByText(/numeric Cesium ion asset ID/)).toBeVisible();
    expect(created).toBeNull();

    await page.getByTestId("site-name-input").fill("Orchard block A");
    await page.getByTestId("asset-id-input").fill("12abc");
    await page.getByTestId("footprint-input").fill('{"type":"Point","coordinates":[0,0]}');
    await page.getByTestId("site-submit").click();
    await expect(sheet.getByText("Footprint must be a Polygon or MultiPolygon")).toBeVisible();
    expect(created).toBeNull();

    await page.getByTestId("asset-id-input").fill("4547222");
    await page
      .getByTestId("footprint-input")
      .fill(
        '{"type":"Polygon","coordinates":[[[-122.14,47.64],[-122.13,47.64],[-122.13,47.65],[-122.14,47.65],[-122.14,47.64]]]}',
      );
    await expect(sheet.getByText(/Area/)).toBeVisible();
    await page.getByTestId("site-submit").click();
    await expect(sheet).toHaveCount(0);
    expect(created).toMatchObject({
      name: "Orchard block A",
      assets: [
        { representation: "gaussian-splat", source: { type: "cesium-ion", assetId: 4547222 } },
      ],
    });
    await expect(page.getByText("Orchard block A added")).toBeVisible();
  });

  test("layer form rejects bad URL templates", async ({ app }) => {
    await app.getByTestId("tool-add-data").click();
    await app.getByTestId("add-tab-imagery").click();
    await app.getByTestId("layer-name").fill("My tiles");
    await app.getByTestId("layer-url").fill("https://tiles.example.com/{z}/{x}.png");
    await app.getByTestId("layer-submit").click();
    await expect(app.getByText("The template must contain {y}.")).toBeVisible();
    await app.getByTestId("layer-url").fill("javascript:alert(1)");
    await app.getByTestId("layer-submit").click();
    await expect(app.getByText("Only http(s) URLs are supported.")).toBeVisible();
  });
});

test.describe("mission control", () => {
  test("view tabs switch between map, plan and fleet; plan detail shows on map", async ({
    app,
  }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.getByTestId("project-card")).toContainText("Blackrock Mesa", {
      timeout: 30_000,
    });
    await expect(app.locator('[data-testid^="machine-marker-"]')).toHaveCount(6);
    await app.getByTestId("view-tab-plan").click({ force: true });
    await expect(app.getByTestId("plans-panel")).toBeVisible();
    await expect(app.getByTestId("representation-switcher")).toHaveCount(0);
    await app.getByTestId("plan-thistle").click({ force: true });
    await expect(app.getByTestId("plan-detail")).toContainText("Remove all invasive star thistle");
    await expect(app.getByTestId("plan-show-on-map")).toBeVisible();
    await app.waitForTimeout(700);
    // The window animates in; a positional click can land on the map mid-transition.
    await app.getByTestId("plan-show-on-map").dispatchEvent("click");
    // The plan window stays open; the plan's zones are drawn on the map with passes and markers.
    await expect.poll(async () => (await planEntityIds(app)).length).toBeGreaterThanOrEqual(4);
    await app.getByTestId("view-tab-fleet").click({ force: true });
    await expect(app.getByTestId("fleet-panel")).toContainText("TR-07 Harrier");
    await app.getByTestId("toggle-work-log").click({ force: true });
    await expect(app.getByTestId("work-log")).toContainText("Treatment log");
    await app.getByTestId("fleet-row-TR-04").click({ force: true });
    await expect(app.getByTestId("selection-card").last()).toContainText("TR-04 Kestrel");
    await expect(app.getByTestId("fleet-panel")).toHaveCount(0);
  });

  test("the agent drafts a plan from a goal and the operator approves it", async ({ app }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.getByTestId("project-card")).toContainText("Blackrock Mesa", {
      timeout: 30_000,
    });
    // The demo agent has simulated threads running, so its card shows; an idle one never says so.
    await expect(app.getByText("Agent idle")).toHaveCount(0);
    await app.getByTestId("view-tab-plan").click({ force: true });
    await app.getByTestId("plan-new").dispatchEvent("click");
    await expect(app.getByTestId("plan-composer")).toBeVisible();
    await app.getByTestId("compose-zone-Z-21").dispatchEvent("click");
    await app.getByTestId("plan-goal").fill("Mow Z-21 weekly with one mower");
    await app.getByTestId("plan-draft-submit").dispatchEvent("click");
    await expect(app.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
    await expect(app.getByTestId("plan-review")).toContainText(
      /Drafted by Claude|Rule-based draft/,
    );
    await expect(app.getByTestId("plan-review")).toContainText("Z-21");
    await expect(app.getByTestId("plan-review")).toContainText("ASSUMPTIONS");
    await expect(app.getByTestId("plan-schedule")).toBeVisible();
    await expect(app.getByTestId("agent-stream")).toContainText("Drafted");
    // The draft is drawn on the map: coverage passes for Z-21, a step marker, no route (one zone).
    await expect
      .poll(() => planEntityIds(app))
      .toEqual(["mission:plan:passes:Z-21", "mission:plan:step:Z-21"]);
    await app.getByTestId("plan-refine").fill("use TR-12 as well");
    await app.getByText("Redraft").dispatchEvent("click");
    await expect(app.getByTestId("plan-changes")).toContainText("Added machine TR-12", {
      timeout: 15_000,
    });
    await app.getByTestId("plan-approve").dispatchEvent("click");
    await expect(app.getByTestId("plan-detail")).toContainText("Mow Z-21 weekly with one mower");
    await expect(app.getByTestId("plan-detail")).toContainText("Scheduled");
    await expect(app.getByTestId("plan-detail")).toContainText("STEPS");
    await app.getByTestId("plan-lifecycle").dispatchEvent("click");
    await expect(app.getByTestId("plan-detail")).toContainText("Dispatched");
    // Revising keeps the plan's identity and lifecycle and adds a revision to its history.
    await app.getByText("Revise with agent").dispatchEvent("click");
    await expect(app.getByTestId("plan-composer")).toBeVisible();
    await app.getByTestId("plan-goal").fill("Mow Z-21 weekly with two mowers");
    await app.getByTestId("plan-draft-submit").dispatchEvent("click");
    await expect(app.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
    await app.getByTestId("plan-approve").dispatchEvent("click");
    await expect(app.getByTestId("plan-detail")).toContainText("rev 2", { timeout: 15_000 });
    await expect(app.getByTestId("plan-detail")).toContainText("Dispatched");
    await app.getByText("All plans").dispatchEvent("click");
    await expect(app.getByTestId("plans-panel")).toContainText("Mow Z-21 weekly with two mowers");
    await expect(app.getByTestId("plans-panel")).not.toContainText("with one mower");
  });

  test("command bar drives the agent stream and layer pills toggle overlays", async ({ app }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.getByTestId("project-card")).toContainText("Blackrock Mesa", {
      timeout: 30_000,
    });
    await app.getByTestId("command-input").fill("where is TR-07");
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("agent-stream")).toContainText("Locating TR-07 Harrier");
    await expect(app.getByTestId("selection-card").last()).toContainText("Service due");
    await app.getByTestId("command-input").fill("hide zones");
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("agent-stream")).toContainText("Zones off");
    await expect(app.getByTestId("pill-zones")).toHaveAttribute("aria-pressed", "false");
    await expect(app.locator('[data-testid^="zone-chip-"]')).toHaveCount(0);
    await app.getByTestId("pill-zones").click({ force: true });
    await expect(app.getByTestId("pill-zones")).toHaveAttribute("aria-pressed", "true");
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("selection-card")).toHaveCount(0);
  });
});
