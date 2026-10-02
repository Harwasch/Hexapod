import type { Page } from "@playwright/test";

import { addOnScreenCredit, demoSite, expect, mockApi, test } from "./fixtures";

/** Ids of the plan overlay entities the map is drawing (passes, step markers, route). */
function entityIds(app: Page): Promise<string[]> {
  return app.evaluate(() => {
    const twin = (
      window as unknown as { __twin?: { viewer: { entities: { values: { id: string }[] } } } }
    ).__twin;
    return (twin?.viewer.entities.values ?? []).map((entity) => entity.id).sort();
  });
}

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
    // At globe scale with no site, the status line sums up the catalog; the camera readouts
    // are developer telemetry and stay off unless Settings › Advanced turns them on.
    await expect(app.getByTestId("status-fleet")).toContainText("1 site");
    await expect(app.getByTestId("status-bar")).toHaveCount(0);
    const errors: string[] = [];
    app.on("pageerror", (e) => errors.push(e.message));
    await app.waitForTimeout(500);
    expect(errors).toEqual([]);
  });

  test("the scene clock advances, so a pure function of it is not a frozen frame", async ({
    app,
  }) => {
    // The Living Survey computes every frame as a pure function of `viewer.clock.currentTime`.
    // That is what makes it reproducible, and it also means a stopped clock and a motion model
    // with no motion in it are indistinguishable on screen — a tree that bends once and sits
    // there. The model was the culprit once; this pins the other candidate so the two can never
    // be confused again. `requestRenderMode` is on, so the question is real: the clock must tick
    // even on the frames the scene declines to draw.
    const readClock = () =>
      app.evaluate(() => {
        const twin = (
          window as unknown as {
            __twin?: {
              viewer: { clock: { currentTime: { dayNumber: number; secondsOfDay: number } } };
            };
          }
        ).__twin;
        const time = twin?.viewer.clock.currentTime;
        if (!time) throw new Error("scene handle missing");
        return time.dayNumber * 86400 + time.secondsOfDay;
      });
    const first = await readClock();
    await app.waitForTimeout(1000);
    const second = await readClock();
    // Roughly a second of scene time for a second of wall time, with wide tolerance: headless
    // GL here is SwiftShader and frames are slow, so the clock is checked for advancing rather
    // than for keeping time.
    expect(second - first).toBeGreaterThan(0.2);
    expect(second - first).toBeLessThan(30);
  });

  test("API failure does not crash the viewer", async ({ page }) => {
    await mockApi(page, { apiDown: true });
    await page.goto("/");
    await expect(page.locator("canvas").first()).toBeVisible();
    await expect(page.getByTestId("notice-api-offline")).toBeVisible();
    // Sites are in the site switcher, behind the site's name at the top left.
    await page.getByTestId("project-card").getByRole("button").first().click();
    await expect(page.getByTestId("site-row-cesium-splat-demo")).toBeVisible();
    await expect(page.getByText("Built-in demo (API offline)")).toBeVisible();
  });
});

test.describe("catalog", () => {
  test("site catalog loads and fly-to-site works", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    // `s` opens the site switcher on its list of sites.
    await app.keyboard.press("s");
    await expect(app.getByTestId("site-switcher")).toBeVisible();
    const card = app.getByTestId("site-row-cesium-splat-demo");
    // One name per site: its project's, not the catalog record's.
    await expect(card).toContainText("Blackrock Mesa");
    await expect(card).not.toContainText("Cesium Gaussian splat demo");
    await expect(card).toContainText("Splat");
    await expect(card).toBeFocused();
    await card.click();
    await expect(app.getByTestId("site-switcher")).toHaveCount(0);
    const strip = app.getByTestId("representation-switcher");
    await expect(strip).toBeVisible({ timeout: 30_000 });
    await expect(strip).toContainText("Blackrock Mesa");
    await expect(strip).not.toContainText("Cesium Gaussian splat demo");
    // At the site the status line reports the site's own fleet, not the catalog's.
    await expect(app.getByTestId("status-fleet")).toContainText("Fleet:");
    await expect(app.getByTestId("status-fleet")).not.toContainText("site");
  });

  test("a fly-to whose site record fails says so beside the switcher, and Retry flies on", async ({
    app,
  }) => {
    // The site's record answers 503 (a cold or failing API). The camera still leaves on the
    // click, for the catalog summary's pose; the site never becomes active, so the pill has to
    // follow the flight's site to say why no model came.
    const record = `**/api/v1/sites/${demoSite.id}`;
    await app.route(record, (route) =>
      route.fulfill({ status: 503, contentType: "application/json", body: '{"detail":"cold"}' }),
    );
    await app.getByTestId("onboarding-explore").click();
    await app.keyboard.press("s");
    await app.getByTestId("site-row-cesium-splat-demo").click();
    const load = app.getByTestId("site-load");
    await expect(load).toContainText("Couldn’t load the 3D model", { timeout: 30_000 });
    await expect(app.getByTestId("representation-switcher")).toHaveCount(0);
    // The API recovers; Retry flies there again and the site loads.
    await app.unroute(record);
    await load.getByRole("button", { name: "Retry" }).click();
    await expect(app.getByTestId("representation-switcher")).toBeVisible({ timeout: 30_000 });
    await expect(app.getByText("Couldn’t load the 3D model")).toHaveCount(0);
  });

  test("representation switch keeps the camera", async ({ app }) => {
    await app.getByTestId("onboarding-demo").click();
    const switcher = app.getByTestId("representation-switcher");
    await expect(switcher).toBeVisible({ timeout: 30_000 });
    await expect(switcher.getByRole("radio", { name: /Splat/ })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    // Which engine draws the splat is a developer's comparison, in Settings › Advanced.
    await expect(switcher.getByTestId("splat-renderer")).toHaveCount(0);
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
    await expect(switcher.getByRole("radio", { name: /Mesh/ })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await app.waitForTimeout(800);
    // Cesium re-normalises the camera frame every frame; compare within float noise.
    const after = await readCamera();
    expect(Math.hypot(after.x - before.x, after.y - before.y, after.z - before.z)).toBeLessThan(
      1e-6,
    );
    expect(Math.abs(after.heading - before.heading)).toBeLessThan(1e-9);
    expect(Math.abs(after.pitch - before.pitch)).toBeLessThan(1e-9);
    await expect(switcher.getByRole("radio", { name: /Points/ })).toBeDisabled();
    // The fixture's mesh lives at example.invalid, so it never loads: the failure is said
    // beside the switch, with a way to try again, and trying again asks for it again.
    const load = app.getByTestId("site-load");
    await expect(load).toContainText("Couldn’t load the 3D model", { timeout: 30_000 });
    // Said once: the pill, not also a toast naming the asset's file.
    await expect(app.getByText(/failed to load/)).toHaveCount(0);
    const tilesetRequests: string[] = [];
    app.on("request", (request) => {
      if (request.url().includes("example.invalid/mesh")) tilesetRequests.push(request.url());
    });
    await load.getByRole("button", { name: "Retry" }).click();
    await expect.poll(() => tilesetRequests.length, { timeout: 15_000 }).toBeGreaterThan(0);
    await expect(load).toContainText("Couldn’t load the 3D model", { timeout: 30_000 });
    // Back to the splat: the mesh's failure is not said about it. (Ion is out of reach here,
    // so the splat is being asked for again; it used to keep the mesh's error from the start.)
    await switcher.getByRole("radio", { name: /Splat/ }).click();
    await expect(load).toContainText("Loading 3D model", { timeout: 5_000 });
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

/**
 * Publishes a Living Survey status from the scene, exactly as `LivingSurveyManager` does.
 *
 * The badge hangs off the `living` scene event, mirrored into the store by `SceneBridge`, and
 * this drives that real path end to end — the app's own emitter, its own subscription, its own
 * render. What it does not do is deform a tileset: making the manager publish `animating: true`
 * for real needs a rigged splat site loaded, which under SwiftShader is a two-minute test.
 * That half is covered where it belongs, in `livingSurveyScene.spec.ts`, which asserts the
 * scene sets `animating` from wind × attachment and clears it at calm. Between the two, the
 * chain from a moving tree to a visible badge is covered without either test pretending to do
 * the other's job.
 */
async function publishLiving(app: Page, animating: boolean): Promise<void> {
  await app.evaluate((moving) => {
    const twin = (
      window as unknown as { __twin?: { events: { emit: (name: string, value: unknown) => void } } }
    ).__twin;
    twin?.events.emit("living", {
      wind: { strength: moving ? 0.12 : 0, bearingDeg: 250 },
      animating: moving,
      sites: [
        {
          // The mocked catalog's demo site, standing in for a rigged one: the badge resolves
          // the name it shows from the catalog by id, and that is part of what is asserted.
          siteId: "11111111-1111-4111-8111-111111111111",
          siteSlug: "cesium-splat-demo",
          assetId: "22222222-2222-4222-8222-222222222222",
          phase: "ready",
          numSplats: 2000,
          displaced: moving,
          rigSourceNote: "synthetic tree, 6.0 m, 33 nodes (tools/captures/synthetic_tree.py)",
          maxDisplacementM: moving ? 0.195 : 0,
          sortStaleness: moving ? 9.7 : 0,
          motionPath: "gpu",
          cpuReason: null,
          motionMs: moving ? 0.42 : null,
          applyMs: moving ? 0.05 : null,
        },
      ],
    });
  }, animating);
}

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

  test("the command box and keyboard shortcuts drive the UI", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    // ⌘K / Ctrl+K focuses the one box; with nothing typed it lists every action and its key.
    await app.keyboard.press("Control+k");
    const input = app.getByRole("combobox", { name: "Search, run an action or ask the agent" });
    await expect(input).toBeFocused();
    const results = app.getByRole("listbox", { name: "Results" });
    await expect(results.getByRole("option", { name: /Saved views/ })).toContainText("B");
    await app.keyboard.type("settings");
    // Enter runs the highlighted row: the Settings action, not the agent.
    await expect(results.getByRole("option", { selected: true })).toContainText("Settings");
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("settings-sheet")).toBeVisible();
    await expect(app.getByRole("radiogroup", { name: "Quality preset" })).toBeVisible();

    // The developer readouts live behind Settings › Advanced, off until turned on.
    await expect(app.getByTestId("status-bar")).toHaveCount(0);
    await app.getByTestId("settings-advanced").getByText("Advanced").click();
    await app.getByRole("switch", { name: "Show developer readouts" }).click();
    await expect(app.getByTestId("status-bar")).toContainText("Alt");
    await expect(app.getByRole("radiogroup", { name: "Splat renderer" })).toBeVisible();
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("settings-sheet")).toHaveCount(0);

    // "/" also focuses the box; Escape closes its list without running anything.
    await app.keyboard.press("/");
    await expect(input).toBeFocused();
    await app.keyboard.type("compare");
    await expect(results).toBeVisible();
    await app.keyboard.press("Escape");
    await expect(results).toHaveCount(0);
    await expect(app.getByTestId("compare-controls")).toHaveCount(0);

    // Local matches are listed above geocoded places: the highlighted best match is on top,
    // and a place that arrives a beat later lands below it rather than above.
    await app.route(/nominatim\.openstreetmap\.org/, (route) =>
      route.fulfill({
        json: [
          {
            display_name: "Layers Lake, Ontario",
            boundingbox: ["45.0", "45.1", "-79.1", "-79.0"],
          },
        ],
      }),
    );
    await app.keyboard.press("/");
    await app.keyboard.type("layers");
    await expect(results.getByRole("option", { name: /Layers Lake/ })).toBeVisible();
    const options = results.getByRole("option");
    await expect(options.first()).toHaveAttribute("aria-selected", "true");
    await expect(options.first()).toContainText("Layers");
    await expect(options.first()).not.toContainText("Layers Lake");
    await app.keyboard.press("Escape");

    await app.keyboard.press("l");
    await expect(app.getByTestId("layers-panel")).toBeVisible();
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("layers-panel")).toHaveCount(0);

    // "?" lists every key, from the same registry the box shows its keys from.
    await app.keyboard.press("Shift+?");
    const sheet = app.getByTestId("shortcut-sheet");
    await expect(sheet).toBeVisible();
    await expect(sheet.getByRole("region", { name: "Camera" })).toContainText("Reset north");
    await expect(sheet).toContainText("Walk / explore mode");
    await app.keyboard.press("Escape");
    await expect(sheet).toHaveCount(0);
  });

  test("major UI is keyboard accessible", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    await app.getByTestId("command-input").focus();
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
    await expect(app.getByTestId("tool-measure")).toBeFocused();
    const rail = app.getByRole("toolbar", { name: "Tools" });
    // Four tools, each labelled in words on the button itself.
    await expect(rail.getByRole("button")).toHaveText(["Layers", "Measure", "Add", "Settings"]);
    await expect(rail.getByRole("button", { name: "Layers" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    // The developer console left the rail for Settings › Advanced (and `D`).
    await expect(app.getByTestId("tool-dev")).toHaveCount(0);
    await app.getByTestId("tool-settings").click();
    await app.getByTestId("settings-advanced").getByText("Advanced").click();
    await app.getByRole("switch", { name: "Developer tools" }).click();
    await expect(app.getByTestId("dev-panel")).toBeVisible();
  });
});

test.describe("the HUD over the map", () => {
  test("the site switcher's menu opens over the tool rail, not under it", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    const rail = app.getByRole("toolbar", { name: "Tools" });
    await expect(rail).toBeVisible();
    await app.getByTestId("project-card").getByRole("button").first().click();
    const menu = app.getByRole("dialog", { name: "Switch site" });
    await expect(menu).toBeVisible();

    // The menu drops down the left edge, over the tool rail below the top bar. Hit-test the
    // overlap rather than reading z-index: what matters is which element takes the click.
    const owner = await app.evaluate(() => {
      const menu = document.querySelector(".mc-project__menu");
      const railRect = document.querySelector(".tool-rail")?.getBoundingClientRect();
      if (!menu || !railRect) return "missing";
      const menuRect = menu.getBoundingClientRect();
      const left = Math.max(menuRect.left, railRect.left);
      const right = Math.min(menuRect.right, railRect.right);
      const top = Math.max(menuRect.top, railRect.top);
      const bottom = Math.min(menuRect.bottom, railRect.bottom);
      // The menu has to be long enough to reach the rail for there to be anything to test.
      if (right - left < 2 || bottom - top < 2) return "no-overlap";
      const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
      if (!hit) return "nothing";
      // Anything but the menu on top — the rail, or the region wrapping it — is the bug.
      return menu.contains(hit) ? "menu" : `${hit.tagName.toLowerCase()}.${hit.className}`;
    });
    expect(owner).toBe("menu");
  });

  test("the site switcher saves, opens and deletes the site's views, and leads to the other pages", async ({
    app,
  }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.getByTestId("project-card")).toContainText("Blackrock Mesa", {
      timeout: 30_000,
    });
    // `b` opens the switcher at its saved views, ready to name the current one.
    await app.keyboard.press("b");
    const switcher = app.getByTestId("site-switcher");
    await expect(switcher).toContainText("Saved views · Blackrock Mesa");
    await expect(switcher.getByTestId("saved-view-Overview")).toBeVisible();
    await expect(switcher.getByRole("textbox", { name: "View name" })).toBeFocused();
    await expect(switcher.getByRole("link", { name: /Scan gallery/ })).toHaveAttribute(
      "href",
      "/view.html",
    );
    await expect(switcher.getByRole("link", { name: /Data console/ })).toHaveAttribute(
      "href",
      "/admin.html",
    );
    await switcher.getByTestId("saved-view-Overview").click();
    await expect(switcher).toHaveCount(0);
  });

  for (const viewport of [
    { name: "desktop", width: 1440, height: 900 },
    { name: "phone", width: 390, height: 844 },
  ]) {
    test.describe(`at ${viewport.name} size`, () => {
      test.use({ viewport: { width: viewport.width, height: viewport.height } });

      test("attribution stays on screen, and the setup advice is not what is on it", async ({
        app,
      }) => {
        await app.getByTestId("onboarding-explore").click();
        // Cesium ion's terms and Google Photorealistic 3D Tiles' terms both require the
        // credit to remain visible. It is a chip in the bottom bar (on a phone, a strip of its
        // own above the status line); it is never removed, hidden or folded behind a button.
        const credits = app.getByTestId("credits").locator(".cesium-viewer-bottom");
        await expect(credits).toBeVisible();
        await expect(credits.locator(".cesium-credit-logoContainer img")).toBeVisible();

        // What must NOT be on screen is CesiumJS's default-token setup advice, which is a
        // paragraph and inflates the chip into a slab over the globe.
        await expect(credits).not.toContainText("default ion access token");
        const box = await credits.boundingBox();
        expect(box?.height ?? 0).toBeLessThan(48);

        // It is demoted, not deleted: the "Data attribution" dialog still carries it, and the
        // dialog itself is reachable — it is hosted on <body>, over the HUD, not under it.
        const expand = credits.getByRole("button", { name: "Data attribution" });
        await expect(expand).toBeVisible();
        await expand.click();
        const dialog = app.getByRole("dialog", { name: "Data attribution" });
        await expect(dialog).toBeVisible();
        await expect(dialog).toContainText("default ion access token");
        await app.getByRole("button", { name: "Close data attribution" }).click();
        await expect(dialog).toBeHidden();

        // A provider's linked credit reads in the accent on the glass, in either theme and
        // when hovered: not the browser's default blue, nor the white and #48b that CesiumJS's
        // own stylesheet (appended after the app's) would give it.
        await addOnScreenCredit(
          app,
          '<a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors</a>',
        );
        const link = credits.getByRole("link", { name: "© OpenStreetMap contributors" });
        await expect(link).toBeVisible();
        const expectAccent = async () => {
          const colours = await link.evaluate((el) => {
            const root = document.documentElement;
            const probe = document.body.appendChild(document.createElement("span"));
            probe.style.color = "var(--accent-strong)";
            const read = () => [getComputedStyle(el).color, getComputedStyle(probe).color];
            const dark = read();
            root.dataset.theme = "light";
            const light = read();
            delete root.dataset.theme;
            probe.remove();
            return { dark, light };
          });
          expect(colours.dark[0]).toBe(colours.dark[1]);
          expect(colours.light[0]).toBe(colours.light[1]);
          expect(colours.light[1]).not.toBe(colours.dark[1]);
        };
        await expectAccent();
        await link.hover();
        await expectAccent();
        const withLink = await credits.boundingBox();
        expect(withLink?.height ?? 0).toBeLessThan(48);
      });
    });
  }

  // Raw `page`, not the `app` fixture: that fixture seeds `twin.settings.v1` on every
  // navigation, which would wipe the dismissal this test reloads to check.
  test("the evaluation-token notice can be dismissed and stays dismissed", async ({ page }) => {
    await mockApi(page);
    // The notice is for whoever deployed this, so it rides the developer readouts. Seeded
    // only once: a reload must find the dismissal this test is about, not a fresh seed.
    await page.addInitScript(() => {
      if (!window.localStorage.getItem("twin.settings.v1"))
        window.localStorage.setItem(
          "twin.settings.v1",
          JSON.stringify({ state: { devReadouts: true }, version: 3 }),
        );
    });
    await page.goto("/");
    await expect(page.getByTestId("cesium-viewport")).toBeVisible();
    const notice = page.getByTestId("notice-default-token");
    await expect(notice).toBeVisible();
    await page.getByTestId("notice-default-token-dismiss").click();
    await expect(notice).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId("cesium-viewport")).toBeVisible();
    await expect(page.getByTestId("status-bar")).toContainText("Alt");
    await expect(page.getByTestId("notice-default-token")).toHaveCount(0);
  });
});

test.describe("living survey", () => {
  test("the Simulated badge is on screen exactly while the motion is", async ({ app }) => {
    await app.getByTestId("onboarding-explore").click();
    const badge = app.getByTestId("simulated-badge");
    // Nothing is moving, so nothing is claimed.
    await expect(badge).toHaveCount(0);

    await publishLiving(app, true);
    await expect(badge).toBeVisible();
    await expect(badge).toContainText("Simulated motion");
    await expect(badge).toContainText("Modelled wind");
    // The name the site is known by, so a viewer can tell which thing on screen is modelled.
    await expect(badge).toContainText("Cesium Gaussian splat demo");
    // Announced rather than merely coloured: the amber says nothing to a screen reader.
    await expect(badge).toHaveAttribute("role", "status");
    // The claims it must never make.
    await expect(badge).not.toContainText("m/s");
    await expect(badge).not.toContainText("wind from");
    await expect(badge).not.toContainText("measured");

    await publishLiving(app, false);
    await expect(badge).toHaveCount(0);
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
    await expect(page.getByTestId("status-line")).toBeVisible();
    await page.getByTestId("tool-add").click();
    const sheet = page.getByTestId("add-panel");
    await expect(sheet).toBeVisible();
    // Uploading leads; linking something already hosted is the second tab.
    await expect(sheet.getByTestId("capture-dropzone")).toBeVisible();
    await expect(page.getByTestId("site-submit")).toHaveCount(0);
    await sheet.getByRole("radio", { name: /Link a source/ }).click();
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
    // "Link a hosted source" from the command box opens Add on its second tab.
    await app.getByTestId("command-input").fill("hosted source");
    await app.getByTestId("command-input").press("Enter");
    await expect(app.getByTestId("add-panel")).toBeVisible();
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
    // A drawer, not a window over the map: the tool panel it replaced is gone, the map stays.
    await expect(app.getByTestId("layers-panel")).toHaveCount(0);
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
    // The Fleet drawer stays open beside the map; the machine's card opens next to it.
    await expect(app.getByTestId("selection-card").last()).toContainText("TR-04 Kestrel");
    await expect(app.getByTestId("fleet-panel")).toBeVisible();
    const drawer = await app.getByTestId("fleet-panel").boundingBox();
    const card = await app.getByTestId("selection-card").last().boundingBox();
    if (!drawer || !card) throw new Error("the drawer or the card is not laid out");
    expect(card.x + card.width).toBeLessThanOrEqual(drawer.x + 1);
    // The map beside it still takes clicks: another machine's marker selects that machine.
    await app.getByTestId("machine-marker-TR-12").dispatchEvent("click");
    await expect(app.getByTestId("selection-card").last()).toContainText("TR-12 Pipit");
    await expect(app.getByTestId("fleet-panel")).toBeVisible();
  });

  test("the agent drafts a plan from a sentence in the bar and the operator approves it", async ({
    app,
  }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.getByTestId("project-card")).toContainText("Blackrock Mesa", {
      timeout: 30_000,
    });
    // The demo agent has simulated threads running, so its card shows; an idle one never says so.
    await expect(app.getByText("Agent idle")).toHaveCount(0);
    await app.getByTestId("command-input").fill("Mow Z-21 weekly with one mower");
    await app.getByTestId("command-input").press("Enter");
    await expect(app.getByTestId("plan-card")).toBeVisible({ timeout: 15_000 });
    await expect(app.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
    await expect(app.getByTestId("plan-source")).toContainText(/Claude|Rule-based/);
    await expect(app.getByTestId("plan-review")).toContainText("Z-21");
    await expect(app.getByTestId("plan-schedule")).toBeVisible();
    // The status line carries the agent's latest line; the chevron opens the whole thread.
    await app.getByTestId("agent-stream-toggle").click();
    await expect(app.getByTestId("agent-stream")).toContainText("Drafted");
    await app.getByTestId("agent-stream-toggle").click();
    await expect(app.getByTestId("agent-stream")).toHaveCount(0);
    // The draft is drawn on the map: coverage passes for Z-21, a step marker, no route (one zone).
    await expect
      .poll(() => planEntityIds(app))
      .toEqual(["mission:plan:passes:Z-21", "mission:plan:step:Z-21"]);
    // Plain words in the bar change the open draft.
    await app.getByTestId("command-input").fill("use TR-12 as well");
    await app.getByTestId("command-input").press("Enter");
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
    await expect(app.getByTestId("plan-card")).toBeVisible();
    await app.getByTestId("command-input").fill("Mow Z-21 weekly with two mowers");
    await app.getByTestId("command-input").press("Enter");
    await expect(app.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
    await app.getByTestId("plan-approve").dispatchEvent("click");
    await expect(app.getByTestId("plan-detail")).toContainText("rev 2", { timeout: 15_000 });
    await expect(app.getByTestId("plan-detail")).toContainText("Dispatched");
    await app.getByText("All plans").dispatchEvent("click");
    await expect(app.getByTestId("plans-panel")).toContainText("Mow Z-21 weekly with two mowers");
    await expect(app.getByTestId("plans-panel")).not.toContainText("with one mower");
  });

  test("a plan can be made anywhere: click the ground, the agent outlines it and drafts", async ({
    app,
  }) => {
    await app.getByTestId("onboarding-explore").click({ force: true });
    // Somewhere with no site, close enough that the ground under a click is real.
    await app.evaluate(() => {
      const twin = (
        window as unknown as {
          __twin?: {
            camera: {
              cancelFlight: () => void;
              setView: (lon: number, lat: number, h: number, hd: number, p: number) => void;
            };
          };
        }
      ).__twin;
      twin?.camera.cancelFlight();
      twin?.camera.setView(-119.9, 36.6, 1500, 0, -60);
    });
    await app.getByTestId("view-tab-plan").click({ force: true });
    await expect(app.getByTestId("plans-panel")).toContainText("Anywhere", { timeout: 15_000 });
    await app.getByTestId("command-input").fill("3D scan this field into a splat");
    await app.getByTestId("command-input").press("Enter");
    await expect(app.getByTestId("plan-awaiting-ground")).toBeVisible({ timeout: 15_000 });
    await expect(app.getByTestId("status-agent")).toContainText("Click the ground");
    // One click on the map: the mapped field under it becomes the ground, corners on the map.
    const canvas = app.locator("canvas").first();
    const box = await canvas.boundingBox();
    if (!box) throw new Error("no canvas");
    await app.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.6);
    await expect(app.getByTestId("plan-ground")).toContainText("Test Field", { timeout: 20_000 });
    await expect(app.getByTestId("plan-ground")).toContainText("OpenStreetMap");
    await expect.poll(() => entityIds(app)).toContain("area-edit:vertex:0");
    await expect(app.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
    await expect.poll(() => planEntityIds(app)).toContain("mission:plan:passes:A-01");
    // What the agent assumed is a row of chips; tapping one changes the plan.
    await expect(app.getByTestId("assume-deadline")).toContainText("14 days");
    await app.getByTestId("assume-passes").dispatchEvent("click");
    await app.getByTestId("assume-passes-two").dispatchEvent("click");
    await expect(app.getByTestId("plan-changes")).toContainText("Machine-hours 147 → 294", {
      timeout: 15_000,
    });
    await expect(app.getByTestId("assume-passes")).toContainText("Two passes");
    // A dragged corner redrafts for the new outline: the ground row's acreage changes.
    const groundBefore = await app.getByTestId("plan-ground").textContent();
    await app.evaluate(() => {
      const twin = (
        window as unknown as {
          __twin?: { events: { emit: (name: string, payload: unknown) => void } };
        }
      ).__twin;
      twin?.events.emit("area-edit", {
        zoneId: "A-01",
        footprint: {
          type: "Polygon",
          coordinates: [
            [
              [-119.91, 36.59],
              [-119.89, 36.59],
              [-119.89, 36.61],
              [-119.91, 36.61],
              [-119.91, 36.59],
            ],
          ],
        },
      });
    });
    // The drafting state can pass within a frame on a fast mock, so assert the outcome.
    await expect(app.getByTestId("plan-ground")).not.toHaveText(groundBefore ?? "", {
      timeout: 30_000,
    });
    await expect(app.getByTestId("plan-review")).toBeVisible({ timeout: 30_000 });
    await app.getByTestId("plan-approve").dispatchEvent("click");
    await expect(app.getByTestId("plan-detail")).toContainText("3D scan this field into a splat");
    await expect(app.getByTestId("plan-detail")).toContainText("A-01 Test Field");
  });

  test("the agent answers on the status line and the layer favourites toggle overlays", async ({
    app,
  }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.getByTestId("project-card")).toContainText("Blackrock Mesa", {
      timeout: 30_000,
    });
    // The site's fleet, counted the way the Fleet window counts it, and labelled simulated.
    await expect(app.getByTestId("status-fleet")).toContainText(
      "Fleet: 4 working · 2 need attention",
    );
    await expect(app.getByTestId("status-fleet")).toContainText("simulated");
    // An instruction highlights "Ask the agent", so Enter asks rather than searches.
    await app.getByTestId("command-input").fill("where is TR-07");
    await expect(app.getByTestId("command-row-agent")).toHaveAttribute("aria-selected", "true");
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("status-agent")).toContainText("Locating TR-07 Harrier");
    await expect(app.getByTestId("selection-card").last()).toContainText("Service due");
    await app.getByTestId("command-input").fill("hide zones");
    await app.keyboard.press("Enter");
    await expect(app.getByTestId("status-agent")).toContainText("Zones off");
    await expect(app.locator('[data-testid^="zone-chip-"]')).toHaveCount(0);
    // The quick toggles are the favourites at the top of Layers, not pills over the map.
    await expect(app.getByTestId("layer-favourites")).toHaveCount(0);
    await app.getByTestId("tool-layers").click();
    await expect(app.getByTestId("pill-zones")).toHaveAttribute("aria-pressed", "false");
    await app.getByTestId("pill-zones").click({ force: true });
    await expect(app.getByTestId("pill-zones")).toHaveAttribute("aria-pressed", "true");
    await expect(app.locator('[data-testid^="zone-chip-"]').first()).toBeAttached();
    await app.keyboard.press("Escape");
    await expect(app.getByTestId("selection-card")).toHaveCount(0);
  });

  test("from far away a site is one pin with its name and a count", async ({ app }) => {
    await app.getByTestId("onboarding-demo").click();
    await expect(app.locator('[data-testid^="machine-marker-"]')).toHaveCount(6, {
      timeout: 30_000,
    });
    const setHeight = (height: number) =>
      app.evaluate((h) => {
        const twin = (
          window as unknown as {
            __twin?: {
              camera: {
                cancelFlight: () => void;
                setView: (lon: number, lat: number, h: number, hd: number, p: number) => void;
              };
            };
          }
        ).__twin;
        twin?.camera.cancelFlight();
        twin?.camera.setView(-122.138, 47.6445, h, 0, -90);
      }, height);
    await setHeight(400_000);
    const pin = app.getByTestId("site-pin");
    await expect(pin).toBeVisible({ timeout: 15_000 });
    await expect(pin).toContainText("Blackrock Mesa");
    await expect(pin).toContainText("9");
    await expect(app.locator('[data-testid^="machine-marker-"]')).toHaveCount(0);
    await expect(app.locator('[data-testid^="zone-chip-"]')).toHaveCount(0);
    await setHeight(3_000);
    await expect(app.getByTestId("site-pin")).toHaveCount(0, { timeout: 15_000 });
    await expect(app.locator('[data-testid^="machine-marker-"]')).toHaveCount(6);
  });
});
