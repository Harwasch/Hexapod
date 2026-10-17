/**
 * The Descent (features/descent): typed coordinates on the welcome card, one flight down, the
 * arrival card, and "Outline this land" proposing the mapped area the flight found on the way.
 *
 * Software GL draws a few frames a second, so the glide runs on a clock that steps a fixed time
 * per frame (`setGlideClock`), as siteFlight.spec.ts does: the flight is the app's own, only
 * faster. Overpass is the fixture's (a field around any point asked about); e2e aborts terrain,
 * so the ground is the ellipsoid.
 */
import { expect, test } from "./fixtures";

const TARGET = { longitude: -120.6, latitude: 35.62 };

/** Metres between two points, near enough at this scale. */
function distanceM(a: { longitude: number; latitude: number }, b: typeof a): number {
  const dLat = (b.latitude - a.latitude) * 111_320;
  const dLon = (b.longitude - a.longitude) * 111_320 * Math.cos((a.latitude * Math.PI) / 180);
  return Math.hypot(dLat, dLon);
}

test("descends to typed coordinates in one flight, onto the arrival card", async ({ app }) => {
  await app.waitForFunction(() => "__twin" in window && Boolean(window.__twin));
  // A clock that steps 100 ms a frame, whatever the machine draws.
  await app.evaluate(`(() => {
    const twin = window.__twin;
    let virtual = performance.now();
    twin.scene.preRender.addEventListener(() => { virtual += 100; });
    twin.camera.setGlideClock(() => virtual);
  })()`);
  const flightsBefore = await app.evaluate<number>("window.__twin.camera.flights");
  await app.getByTestId("descent-input").fill("35.62, -120.60");
  await app.getByTestId("descent-submit").click();
  // The welcome card goes as the flight leaves.
  await expect(app.getByTestId("onboarding")).toHaveCount(0);
  const arrival = app.getByTestId("descent-arrival");
  await expect(arrival).toBeVisible({ timeout: 60_000 });
  await expect(arrival).toContainText("35.62000° N, 120.60000° W");
  await expect(app.getByTestId("descent-mapped")).toContainText("Test Field", { timeout: 15_000 });
  await expect(app.getByTestId("descent-mapped")).toContainText("Not a legal parcel");
  await app.waitForFunction("!window.__twin.camera.gliding");
  const pose = await app.evaluate<{ longitude: number; latitude: number; altitude: number }>(
    "window.__twin.camera.pose()",
  );
  expect(distanceM(pose, TARGET)).toBeLessThan(2_000);
  expect(pose.altitude).toBeLessThan(2_000);
  // One flight, re-pointed on the way: never a second take-off.
  expect(await app.evaluate<number>("window.__twin.camera.flights")).toBe(flightsBefore + 1);

  if (process.env.VITE_ENABLE_LAND_EXPLORATION === "true") {
    await app.getByRole("button", { name: "Outline this land" }).click();
    const panel = app.getByTestId("land-panel");
    await expect(panel).toBeVisible();
    await expect(panel).toContainText("OpenStreetMap feature");
    await expect(panel.getByRole("textbox", { name: "Name this land" })).toHaveValue("Test Field");
    await expect(arrival).toHaveCount(0);
  } else {
    await expect(app.getByRole("button", { name: "Outline this land" })).toHaveCount(0);
  }
});

test("says where nothing was found, and asks the geocoder only on submit", async ({ app }) => {
  const asked: string[] = [];
  await app.route("https://nominatim.openstreetmap.org/**", (route) => {
    asked.push(new URL(route.request().url()).searchParams.get("q") ?? "");
    return route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
  });
  await app.getByTestId("descent-input").fill("Nowhere in particular");
  await app.waitForTimeout(500);
  expect(asked).toEqual([]);
  await app.getByTestId("descent-submit").click();
  await expect(app.getByRole("alert")).toContainText("No place found for “Nowhere in particular”");
  expect(asked).toEqual(["Nowhere in particular"]);
});
