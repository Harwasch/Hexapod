/**
 * "Find my land" on the landing page's request form: the site finder (finder.html) opens in a
 * dialog, runs the app's Descent search and parcel picker on the app's globe, asks the public
 * parcel route (stubbed here), and hands the chosen land back to the form. The landing page
 * itself still loads no CesiumJS: the globe is the finder frame's.
 */
import { expect, test, type Frame, type Page, type Route } from "@playwright/test";

test.use({ reducedMotion: "reduce", colorScheme: "light" });

const PARCEL_ID = "ca-parcel/06001/2-51-1";
const ATTRIBUTION = "Alameda County Information Technology Department";
/** A statement the source's terms require with its data, as the route returns Kern's. */
const NOTICE = {
  source: "Alameda County parcels",
  text: "SUMMARY:\n\nA test disclaimer that must travel with these records.",
};

/** A small square around a point, as a county service would outline a parcel. */
function square(point: number[], half = 0.0008) {
  const [lon = 0, lat = 0] = point;
  return {
    type: "Polygon",
    coordinates: [
      [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
      ],
    ],
  };
}

interface Stubs {
  /** Bodies posted to the public parcel route. */
  asked: { point: { coordinates: number[] } }[];
  /** Non-GET requests other than that route: anything else leaving the page. */
  sent: string[];
  /** CesiumJS requests made by the landing page itself (not the finder frame). */
  landingCesium: string[];
  /** Uncaught errors in either page. */
  errors: string[];
  /** Bodies posted to the scan-request route. */
  requests: Record<string, unknown>[];
}

async function stub(page: Page, answer: "parcels" | "uncovered"): Promise<Stubs> {
  const stubs: Stubs = { asked: [], sent: [], landingCesium: [], errors: [], requests: [] };
  page.on("pageerror", (error) => stubs.errors.push(error.message));
  page.on("request", (request) => {
    const url = request.url();
    // Allowed out: the public parcel route, and the Descent's own Overpass query for the mapped
    // area at the searched spot (a read-only public lookup; it carries nothing of the form).
    const expected =
      url.includes("/api/v1/public/parcels/near") ||
      url.includes("/api/v1/scan-requests") ||
      url.startsWith("https://overpass-api.de/");
    if (request.method() !== "GET" && !expected) stubs.sent.push(`${request.method()} ${url}`);
    if (/cesium/i.test(url) && request.frame() === page.mainFrame()) stubs.landingCesium.push(url);
  });
  // The globe offline, as the app's own e2e keeps it (fixtures.ts).
  await page.route(/https:\/\/(api|assets|tile)\.cesium\.com\/.*/, (route) => route.abort());
  await page.route(/https:\/\/tile\.openstreetmap\.org\/.*/, (route) => route.abort());
  await page.route(/https:\/\/services\.terrascope\.be\/.*/, (route) => route.abort());
  await page.route(/https:\/\/(dev\.virtualearth|ecn\.t[0-9]\.tiles\.virtualearth)\.net\/.*/, (r) =>
    r.abort(),
  );
  await page.route("https://overpass-api.de/**", (route) =>
    route.fulfill({ json: { elements: [] } }),
  );
  await page.route("https://nominatim.openstreetmap.org/**", (route) =>
    route.fulfill({ json: [] }),
  );
  await page.route("**/api/v1/**", (route) =>
    route.fulfill({ status: 404, json: { title: "Not found", status: 404 } }),
  );
  await page.route("**/api/v1/scan-requests", async (route: Route) => {
    stubs.requests.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill({ status: 202, json: { status: "received" } });
  });
  await page.route("**/api/v1/public/parcels/near", async (route: Route) => {
    const body = route.request().postDataJSON() as Stubs["asked"][number];
    stubs.asked.push(body);
    if (answer === "uncovered") {
      await route.fulfill({
        json: {
          status: "uncovered",
          message:
            "Parcel records are available for California. Here, draw the land or import a boundary file.",
          candidates: [],
          truncated: false,
        },
      });
      return;
    }
    await route.fulfill({
      json: {
        status: "available",
        message:
          "Parcels from Alameda County records near your click. They show county assessor mapping, not surveyed property lines.",
        truncated: false,
        notices: [NOTICE],
        candidates: [
          {
            id: PARCEL_ID,
            label: "Parcel 2-51-1",
            geometry: square(body.point.coordinates),
            distanceM: 0,
            properties: {},
            source: {
              method: "parcel",
              meaning: "recorded-parcel",
              label: "Alameda County assessor parcel",
              recordId: "2-51-1",
              attribution: ATTRIBUTION,
            },
          },
        ],
      },
    });
  });
  return stubs;
}

/** Opens the finder and waits for its globe; returns the finder's frame. */
async function openFinder(page: Page): Promise<Frame> {
  await page.goto("/landing.html#start");
  await page.getByRole("button", { name: "Find my land" }).click();
  // In the page, inside the request form: not a pop-up over it.
  const region = page.locator("form#f").getByRole("region", { name: "Find your land" });
  await expect(region).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Hide the map" })).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  await expect
    .poll(() => page.frames().some((frame) => frame.url().includes("/finder.html")))
    .toBe(true);
  const frame = page.frames().find((candidate) => candidate.url().includes("/finder.html"));
  if (!frame) throw new Error("the finder frame did not load");
  // The scene manager, which the dev server puts on `window` (cesium/CesiumViewport.tsx).
  await frame.waitForFunction("'__twin' in window && Boolean(window.__twin)", null, {
    timeout: 60_000,
  });
  return frame;
}

/**
 * Goes to a place by coordinates (no geocoder) and waits for the camera to be down over it and
 * still. Height and "not gliding" alone are not enough: the finder opens low over the app's
 * default site, so both can be true before the flight has left.
 */
async function goTo(frame: Frame, lat: number, lon: number) {
  // The app's Descent search, asking the finder's own question.
  await expect(frame.getByLabel("Where is your land?")).toBeVisible();
  await frame.getByTestId("descent-input").fill(`${lat}, ${lon}`);
  await frame.getByTestId("descent-submit").click();
  await frame.waitForFunction(
    `(() => {
      const twin = window.__twin;
      const at = twin.viewer.camera.positionCartographic;
      const degrees = 180 / Math.PI;
      return (
        !twin.camera.gliding &&
        at.height < 5000 &&
        Math.abs(at.latitude * degrees - ${lat}) < 0.03 &&
        Math.abs(at.longitude * degrees - ${lon}) < 0.03
      );
    })()`,
    null,
    { timeout: 60_000 },
  );
}

/** Clicks the finder's map at a fraction of the frame, clear of the cards on the left. */
async function clickMap(page: Page, fx: number, fy: number) {
  const box = await page.locator("#pk-finder iframe").boundingBox();
  if (!box) throw new Error("the finder frame has no box");
  await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
}

test("finds a parcel on the map, selects it, and hands it to the form", async ({ page }) => {
  const stubs = await stub(page, "parcels");
  const frame = await openFinder(page);
  await goTo(frame, 37.8044, -122.2711);
  const picker = frame.getByTestId("land-parcel-picker");
  await expect(picker).toContainText("near your land on the map");

  await clickMap(page, 0.7, 0.5);
  const parcel = picker.getByRole("checkbox", { name: "Parcel 2-51-1" });
  await expect(parcel).toBeVisible({ timeout: 30_000 });
  // The public route, asked once, about the tapped spot in the searched region. Not closer than
  // that: offline (imagery and terrain blocked) the globe's depth pick under a tap is coarse.
  // Measured 2026-10-20 on SwiftShader, the same tap landed 0.01-0.19 degrees from the target
  // from run to run; with live terrain it lands on the tap. Pick precision is the app's own,
  // tested in land.spec.ts; this spec is about the finder's wiring.
  expect(stubs.asked).toHaveLength(1);
  const [lon = 0, lat = 0] = stubs.asked[0]?.point.coordinates ?? [];
  expect(Math.abs(lon + 122.2711)).toBeLessThan(0.5);
  expect(Math.abs(lat - 37.8044)).toBeLessThan(0.5);
  await expect(frame.getByTestId("finder-attribution")).toContainText(ATTRIBUTION);
  // The disclaimer its terms require, shown with the records.
  const notice = frame.getByTestId("finder-notice");
  await expect(notice).toContainText(NOTICE.source);
  await notice.locator("summary").click();
  await expect(notice).toContainText("must travel with these records");
  await parcel.check();
  await picker.getByRole("button", { name: /^Use this/ }).click();

  // Back on the form: the dialog closed, the parcel listed with its source's credit.
  await expect(page.getByRole("region", { name: "Find your land" })).toBeHidden();
  await expect(page.getByRole("button", { name: "Change my land" })).toBeVisible();
  const list = page.locator("#pk-list");
  await expect(list).toContainText("Parcel 2-51-1");
  await expect(list).toContainText("County parcel record");
  await expect(page.locator("#pk-attr")).toContainText(ATTRIBUTION);
  await expect(page.locator("#pk-notices")).toContainText(NOTICE.source);
  await expect(page.getByLabel("Acres")).not.toHaveValue("");

  await page.getByLabel("Name").fill("Ada Lovelace");
  await page.getByLabel("Email").fill("ada@example.com");
  await page.getByLabel("What needs doing?").fill("Broom on the north slope.");
  await page.locator("form#f").getByRole("button", { name: "Request a site scan" }).click();
  const status = page.locator("form#f").getByRole("status");
  await expect(status).toContainText("We have your request (Parcel 2-51-1");
  // Sent to our API as exactly name, email, notes, boundary, parcel ids, and the honeypot.
  const [request] = stubs.requests;
  expect(stubs.requests).toHaveLength(1);
  expect(Object.keys(request ?? {}).sort()).toEqual([
    "boundary",
    "email",
    "name",
    "notes",
    "parcelIds",
    "website",
  ]);
  expect(request?.parcelIds).toEqual([PARCEL_ID]);
  expect((request?.boundary as { type: string }).type).toBe("Polygon");
  expect(request?.notes).toContain("Broom on the north slope.");
  expect(request?.website).toBe("");

  expect(stubs.sent).toEqual([]);
  expect(stubs.landingCesium).toEqual([]);
  expect(stubs.errors).toEqual([]);
});

test("outside California, the outline can be drawn instead", async ({ page }) => {
  const stubs = await stub(page, "uncovered");
  const frame = await openFinder(page);
  await goTo(frame, 47.645, -122.135);
  const picker = frame.getByTestId("land-parcel-picker");
  await clickMap(page, 0.7, 0.5);
  await expect(picker).toContainText("Parcel records are available for California");
  expect(stubs.asked).toHaveLength(1);

  await frame.getByRole("button", { name: "Draw the outline instead" }).click();
  for (const [fx, fy] of [
    [0.6, 0.35],
    [0.85, 0.35],
    [0.85, 0.65],
    [0.6, 0.65],
  ] as const) {
    await clickMap(page, fx, fy);
  }
  const drawbar = frame.getByTestId("land-drawbar");
  await expect(drawbar).toContainText("4 corners");
  await drawbar.getByRole("button", { name: "Done", exact: true }).click();

  await expect(page.getByRole("region", { name: "Find your land" })).toBeHidden();
  await expect(page.locator("#pk-list")).toContainText("Drawn outline");
  await expect(page.locator("#pk-attr")).toBeHidden();
  await expect(page.locator("#pk-notices")).toBeHidden();
  expect(stubs.sent).toEqual([]);
  expect(stubs.errors).toEqual([]);
});
