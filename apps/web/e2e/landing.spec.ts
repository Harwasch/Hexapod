/**
 * The landing page (landing.html): it loads without CesiumJS, its hero scene starts or gives way
 * to a still, in any of the looks under review, the nav scrolls to its sections, the theme
 * toggles, the app section's points switch its screenshots, and the request form validates and
 * sends.
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

/** What the form posted to the scan-request route, answered as the API answers: 202. */
async function stubRequests(page: Page, status = 202) {
  const bodies: Record<string, unknown>[] = [];
  await page.route("**/api/v1/scan-requests", async (route) => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill(
      status === 202
        ? { status, json: { status: "received" } }
        : { status, json: { title: "Too many requests", status } },
    );
  });
  return bodies;
}

/** The one request a complete submit is meant to make. */
const SCAN_REQUEST = /^POST http:\/\/127\.0\.0\.1:\d+\/api\/v1\/scan-requests$/;

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
    ["Mission Control", "app"],
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

test("the request form validates, sends nothing until complete, then sends the request", async ({
  page,
}) => {
  const bodies = await stubRequests(page);
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
  // Two refused submits, and nothing has left the page.
  expect(leaks.left()).toEqual(NOTHING_LEFT);

  // Complete: one request to our API, exactly this, and the page says it was received.
  await email.fill("ada@example.com");
  await submit.click();
  await expect(page.locator("#f-email-err")).toBeHidden();
  await expect(status).toContainText("We have your request (12 acres)");
  await expect(status).toContainText("will reply to ada@example.com");
  await expect(status).toHaveAttribute("data-state", "received");
  expect(bodies).toEqual([
    {
      name: "Ada Lovelace",
      email: "ada@example.com",
      notes: "Area (acres): 12",
      boundary: null,
      parcelIds: [],
      website: "",
    },
  ]);
  const { sent, navigations, search } = leaks.left();
  expect(sent).toHaveLength(1);
  expect(sent[0]).toMatch(SCAN_REQUEST);
  expect({ navigations, search }).toEqual({ navigations: [], search: "" });
  await expect(submit).toBeEnabled();
});

test("a refused send says what to do, and the form can send again", async ({ page }) => {
  const bodies = await stubRequests(page, 429);
  await page.goto("/landing.html#start");
  const form = page.locator("form#f");
  await form.getByLabel("Name").fill("Ada Lovelace");
  await form.getByLabel("Email").fill("ada@example.com");
  await form.getByLabel("Acres").fill("12");
  const submit = form.getByRole("button", { name: "Request a site scan" });
  await submit.click();
  const status = form.getByRole("status");
  await expect(status).toContainText("Wait a minute, then send it again");
  await expect(status).toHaveAttribute("data-state", "error");
  await expect(submit).toBeEnabled();
  expect(bodies).toHaveLength(1);
  // The honeypot is off screen and out of the tab order, and was sent empty.
  await expect(page.locator("#f-website")).toHaveAttribute("tabindex", "-1");
  expect(bodies[0]?.website).toBe("");
});

test("when the site finder fails, the form still validates and sends on acres", async ({
  page,
}) => {
  const bodies = await stubRequests(page);
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
  await expect(form.getByRole("status")).toContainText("We have your request (30 acres)");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]?.boundary).toBeNull();
  const { sent } = leaks.left();
  expect(sent).toHaveLength(1);
  expect(sent[0]).toMatch(SCAN_REQUEST);
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

test("shows no placeholder furniture: no counters, spec sheets or coming-soon cards", async ({
  page,
}) => {
  await page.goto("/landing.html");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  // The simulation's robot counter, the fleet's placeholder spec lists, the "Next" cards.
  await expect(page.getByText("Crews working")).toHaveCount(0);
  await expect(page.locator("#fleet dl")).toHaveCount(0);
  await expect(page.locator("#uses .tag", { hasText: /^Next$/ })).toHaveCount(0);
  // Repair is a capability, beside the others.
  await expect(page.locator("#capabilities").getByText("Repair", { exact: true })).toBeVisible();
});

// One look with real light, shadows and the finish pass, one with scan dots: the renderer's two
// new shader paths, each in both themes' versions. A shader that fails to compile logs, it does
// not throw.
for (const [id, lightStage, darkStage] of [
  ["golden", "#EFDDBD", "#121A2A"],
  ["scan", "#F3F5F7", "#070B0F"],
] as const) {
  test(`the hero renders in the ${id} look, and turns over with the theme`, async ({ page }) => {
    const shaderErrors: string[] = [];
    page.on("console", (message) => {
      if (/shader|WebGLProgram/i.test(message.text())) shaderErrors.push(message.text());
    });
    await page.goto(`/landing.html?look=${id}`);
    const hero = page.locator("#top");
    await expect(hero).toHaveAttribute("data-look", id);
    await expect(hero).toHaveAttribute("data-scene", "live", { timeout: 60_000 });
    await expect(page.locator('#ex button[aria-current="true"]')).toHaveCount(1, {
      timeout: 60_000,
    });
    const stage = () => hero.evaluate((el) => getComputedStyle(el).getPropertyValue("--stage"));
    expect(await stage()).toBe(lightStage);
    await page.getByRole("button", { name: "Switch to dark mode" }).click();
    await expect.poll(stage).toBe(darkStage);
    await expect(page.locator(".stage")).toHaveAttribute("data-theme", "dark");
    expect(shaderErrors).toEqual([]);
  });
}

test("an unknown look is the line drawing, with no review switcher", async ({ page }) => {
  await page.goto("/landing.html?look=nonsense");
  await expect(page.locator("#top")).toHaveAttribute("data-look", "line");
  await expect(page.locator(".review")).toHaveCount(0);
});

test("with ?review, the switcher changes the look", async ({ page }) => {
  await page.goto("/landing.html?review");
  const review = page.getByRole("complementary", { name: "Review options" });
  await review.getByRole("combobox", { name: "Look" }).selectOption("sensor");
  // The switcher reloads the page; wait for the new address, not for the scene to finish loading.
  await page.waitForURL(/look=sensor/, { waitUntil: "commit" });
  await expect(page.locator("#top")).toHaveAttribute("data-look", "sensor");
});

test("the app section's points switch its screenshot", async ({ page }) => {
  await page.goto("/landing.html");
  const app = page.locator("#app");
  await expect(app.locator("h2")).toHaveText("Mission Control");
  await expect(app.locator("#shot-land")).toBeVisible();
  await app.getByRole("button", { name: /^Plan/ }).click();
  await expect(app.locator("#shot-plan")).toBeVisible();
  await expect(app.locator("#shot-land")).toBeHidden();
  await expect(app.getByRole("button", { name: /^Plan/ })).toHaveAttribute("aria-pressed", "true");
  // Simulated things are labelled as simulated.
  await expect(app.locator("figcaption")).toContainText("simulated");
  await expect(app.getByText("Keep the record")).toHaveCount(0);
});

test("fits a phone without scrolling sideways", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/landing.html");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  for (const id of ["top", "fleet", "app", "uses", "start"]) {
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true);
  }
});
