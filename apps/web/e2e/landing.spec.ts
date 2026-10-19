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
  // The site picker is a 2D canvas, so the form still works.
  await expect(page.locator("#pk-total")).toHaveText(/acres/);
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
  const sent: string[] = [];
  page.on("request", (request) => {
    if (request.method() !== "GET") sent.push(`${request.method()} ${request.url()}`);
  });
  await page.goto("/landing.html#start");
  const form = page.locator("form#f");
  const name = form.getByLabel("Name");
  const email = form.getByLabel("Email");
  const submit = form.getByRole("button", { name: "Request a site scan" });
  const status = form.getByRole("status");

  // Nothing selected and nothing typed: every required part is called out.
  await form.getByRole("button", { name: "Clear" }).click();
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
  expect(sent).toEqual([]);
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
