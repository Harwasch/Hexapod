/**
 * The landing page (landing.html): it loads without CesiumJS, its hero scene starts or gives way
 * to a still, the nav scrolls to its sections, the theme toggles, and the request form validates
 * and says plainly that it sends nothing yet.
 */
import { expect, test, type Page } from "@playwright/test";

import { BRAND } from "../src/landing/brand";

test.use({ colorScheme: "light" });

/** Requests for anything of CesiumJS's, which this page must never make. */
function watchCesium(page: Page): string[] {
  const seen: string[] = [];
  page.on("request", (request) => {
    if (/cesium/i.test(request.url())) seen.push(request.url());
  });
  return seen;
}

/**
 * Everything that could carry the form's contents off the page: requests other than GETs,
 * and any navigation of the page itself (a native GET submit is one, with the fields in its
 * query string). Read `left()` after the page has loaded.
 */
function watchLeaks(page: Page) {
  const sent: string[] = [];
  const navigations: string[] = [];
  page.on("request", (request) => {
    if (request.method() !== "GET") sent.push(`${request.method()} ${request.url()}`);
  });
  page.on("framenavigated", (frame) => {
    if (frame === page.mainFrame()) navigations.push(frame.url());
  });
  return {
    /** Call once the page has loaded: later navigations are the ones that count. */
    arm: () => navigations.splice(0),
    left: () => ({ sent, navigations, search: new URL(page.url()).search }),
  };
}

const NOTHING_LEFT = { sent: [], navigations: [], search: "" };

test("loads without CesiumJS, with the brand from brand.ts", async ({ page }) => {
  const cesium = watchCesium(page);
  await page.goto("/landing.html");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    /Robots for\s*everything outside\./i,
  );
  await expect(page.locator("header .brand")).toHaveText(BRAND);
  await expect(page.locator("footer [data-brand]")).toHaveText(BRAND);
  await expect(page).toHaveTitle(new RegExp(`^${BRAND.replace(/[[\]]/g, "\\$&")}`));
  // Let the scenes' modules load too: the check is that none of them reaches the engine.
  await expect(page.locator("#top")).toHaveAttribute("data-scene", /live|fallback/, {
    timeout: 60_000,
  });
  expect(cesium).toEqual([]);
});

test("the hero scene renders and places its crews' labels", async ({ page }) => {
  await page.goto("/landing.html");
  // Software WebGL (playwright.config.ts) is enough for the scene; the fallback has its own test.
  await expect(page.locator("#top")).toHaveAttribute("data-scene", "live", { timeout: 60_000 });
  await expect(page.locator("canvas.gl")).toHaveClass(/\bon\b/);
  await expect(page.locator(".hero-still.lt")).toBeHidden();
  // A label is placed from the scene's own projection, every frame: one in view means frames ran.
  await expect
    .poll(
      () =>
        page.$$eval("#zl a", (links) =>
          links.some((a) => Number.parseFloat(getComputedStyle(a).opacity) > 0.1),
        ),
      { timeout: 60_000 },
    )
    .toBe(true);
  // The examples strip follows the camera: one example is current.
  await expect(page.locator('#ex button[aria-current="true"]')).toHaveCount(1, { timeout: 60_000 });
});

test("without WebGL the hero falls back to a still and the examples scroll to their cards", async ({
  page,
}) => {
  // A browser without WebGL: every WebGL context request comes back empty, 2D still works.
  await page.addInitScript(() => {
    type GetContext = (this: HTMLCanvasElement, type: string, ...rest: unknown[]) => unknown;
    const original = Reflect.get(HTMLCanvasElement.prototype, "getContext") as GetContext;
    const withoutWebgl: GetContext = function (type, ...rest) {
      return type.startsWith("webgl") ? null : original.call(this, type, ...rest);
    };
    Reflect.set(HTMLCanvasElement.prototype, "getContext", withoutWebgl);
  });
  await page.goto("/landing.html");
  const hero = page.locator("#top");
  await expect(hero).toHaveAttribute("data-scene", "fallback", { timeout: 60_000 });
  await expect(page.locator(".hero-still.lt")).toBeVisible();
  await expect(page.locator("canvas.gl")).toBeHidden();
  await page.getByRole("button", { name: "Solar farm" }).click();
  await expect(page.locator("#use-solar")).toBeInViewport();
  // The form needs no WebGL of its own: acres alone complete a request.
  await expect(page.locator("form#f").getByLabel("Acres")).toBeEditable();
});

test("the nav scrolls to each section", async ({ page }) => {
  await page.goto("/landing.html");
  const nav = page.getByRole("navigation", { name: "Main" });
  for (const [name, id] of [
    ["Fleet", "fleet"],
    ["Capabilities", "capabilities"],
    ["App", "app"],
    ["Use cases", "uses"],
  ] as const) {
    await nav.getByRole("link", { name, exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`#${id}$`));
    await expect(page.locator(`#${id} h2`)).toBeInViewport();
  }
  await page.locator("header").getByRole("link", { name: "Request a site scan" }).click();
  await expect(page.locator("#start h2")).toBeInViewport();
});

test("the toggle switches between light and dark", async ({ page }) => {
  await page.goto("/landing.html");
  const toggle = page.getByRole("button", { name: "Switch to dark mode" });
  const background = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(await background()).toBe("rgb(255, 255, 255)");
  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(await background()).toBe("rgb(10, 11, 10)");
  await page.getByRole("button", { name: "Switch to light mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(await background()).toBe("rgb(255, 255, 255)");
});

test("the request form validates, and sends nothing while it is not connected", async ({
  page,
}) => {
  const leaks = watchLeaks(page);
  await page.goto("/landing.html#start");
  leaks.arm();
  const form = page.locator("form#f");
  const name = form.getByLabel("Name");
  const email = form.getByLabel("Email");
  const submit = form.getByRole("button", { name: "Request a site scan" });
  const status = form.getByRole("status");

  // Nothing found and nothing typed: every required part is called out.
  await expect(form.getByLabel("Acres")).toHaveValue("");
  await submit.click();
  await expect(page.locator("#f-name-err")).toHaveText("Enter your name.");
  await expect(page.locator("#f-email-err")).toBeVisible();
  await expect(page.locator("#pk-err")).toBeVisible();
  await expect(name).toHaveAttribute("aria-invalid", "true");
  await expect(name).toBeFocused();
  await expect(status).toBeHidden();

  // A malformed email is still an error; the others clear as they are fixed.
  await name.fill("Ada Lovelace");
  await email.fill("ada@");
  await form.getByLabel("Acres").fill("12");
  await submit.click();
  await expect(page.locator("#f-name-err")).toBeHidden();
  await expect(name).not.toHaveAttribute("aria-invalid", "true");
  await expect(page.locator("#pk-err")).toBeHidden();
  await expect(page.locator("#f-email-err")).toContainText("name@example.com");
  await expect(email).toBeFocused();

  // Complete: the page says it is not connected, and nothing left the browser.
  await email.fill("ada@example.com");
  await submit.click();
  await expect(page.locator("#f-email-err")).toBeHidden();
  await expect(status).toBeVisible();
  await expect(status).toContainText("not connected yet, so nothing was sent");
  await expect(status).toContainText("12 acres");
  // What would be sent, exactly: no land was found, so no boundary and no parcels.
  expect(JSON.parse((await status.getAttribute("data-request")) ?? "null")).toEqual({
    name: "Ada Lovelace",
    email: "ada@example.com",
    notes: "Area (acres): 12",
    boundary: null,
    parcelIds: [],
  });
  expect(leaks.left()).toEqual(NOTHING_LEFT);
});

test("when the site finder fails, the form still validates and still sends nothing", async ({
  page,
}) => {
  // The finder module, swapped for one that throws as the page starts it.
  await page.route(
    (url) => url.pathname.endsWith("/src/landing/finder.ts"),
    (route) =>
      route.fulfill({
        contentType: "text/javascript",
        body: 'export const startFinder = () => { throw new Error("test: no finder"); };',
      }),
  );
  const leaks = watchLeaks(page);
  await page.goto("/landing.html#start");
  leaks.arm();
  const form = page.locator("form#f");
  await expect(form.getByRole("button", { name: "Find my land" })).toBeDisabled();
  const submit = form.getByRole("button", { name: "Request a site scan" });
  await form.getByLabel("Name").fill("Ada Lovelace");
  await form.getByLabel("Email").fill("ada@example.com");
  await submit.click();
  await expect(page.locator("#pk-err")).toBeVisible();
  await form.getByLabel("Acres").fill("30");
  await submit.click();
  await expect(form.getByRole("status")).toContainText("nothing was sent");
  await expect(form.getByRole("status")).toContainText("30 acres");
  expect(leaks.left()).toEqual(NOTHING_LEFT);
});

test.describe("without JavaScript", () => {
  test.use({ javaScriptEnabled: false });

  test("a submit carries none of the form's contents", async ({ page }) => {
    const leaks = watchLeaks(page);
    await page.goto("/landing.html");
    leaks.arm();
    const form = page.locator("form#f");
    await form.getByLabel("Name").fill("Ada Lovelace");
    await form.getByLabel("Email").fill("ada@example.com");
    await form.getByLabel("Acres").fill("30");
    await form.getByRole("button", { name: "Request a site scan" }).click();
    await page.waitForLoadState("load");
    const { sent, navigations, search } = leaks.left();
    expect(sent).toEqual([]);
    expect(search).toBe("");
    // A native submit of a form with no named fields may reload the page; it carries nothing.
    for (const url of navigations) expect(new URL(url).search).toBe("");
  });
});

test.describe("with reduced motion", () => {
  test.use({ reducedMotion: "reduce" });

  test("the hero holds still, and an example cuts to its crew instead of flying", async ({
    page,
  }) => {
    await page.goto("/landing.html");
    await expect(page.locator("#top")).toHaveAttribute("data-scene", "live", { timeout: 60_000 });
    // The crews' labels are placed from the camera every frame: still labels, still camera.
    const labels = () =>
      page.$$eval("#zl a", (links) => links.map((a) => a.style.transform).join(" "));
    await expect.poll(labels, { timeout: 60_000 }).toMatch(/translate/);
    const before = await labels();
    await page.waitForTimeout(1500);
    expect(await labels()).toBe(before);
    await page.getByRole("button", { name: "Quarry" }).click();
    await expect.poll(labels, { timeout: 30_000 }).not.toBe(before);
    const after = await labels();
    await page.waitForTimeout(1500);
    expect(await labels()).toBe(after);
  });
});

test("fits a phone without scrolling sideways", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/landing.html");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  for (const id of ["top", "fleet", "uses", "start"]) {
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true);
  }
});
