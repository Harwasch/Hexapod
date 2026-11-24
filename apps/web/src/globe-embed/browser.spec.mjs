import { expect, test } from "@playwright/test";

test("phone globe keeps imagery without an ion token, frames scans and survives reopening", async ({
  page,
}) => {
  const ionRequests = [];
  page.on("request", (request) => {
    if (request.url().includes("api.cesium.com")) ionRequests.push(request.url());
  });
  await page.route("https://tile.openstreetmap.org/**", (route) => route.abort());
  await page.goto("/globe-embed.html");
  await expect(page.locator("canvas")).toBeVisible();
  await expect
    .poll(() => page.locator("canvas").evaluate((canvas) => canvas.width / canvas.clientWidth))
    .toBe(3);
  await expect.poll(() => page.evaluate(() => Boolean(window.globeBridge))).toBe(true);
  await expect(page.locator("#map-status")).toHaveText(/Terrain unavailable|Map offline/);
  // Natural Earth is bundled below the failed online layer. The rendered Earth should
  // occupy many pixels with opaque nonblack colour, even with all remote imagery blocked.
  await expect
    .poll(
      async () => {
        // Inspect the same CSS viewport at every DPR. Device-scale capture at DPR 3
        // encodes/decodes nine times as many pixels before producing this 100x100 sample.
        // Keep the separate backing-store assertion above at full native resolution.
        const rendered = (await page.screenshot({ scale: "css" })).toString("base64");
        return page.evaluate(async (base64) => {
          const image = new Image();
          image.src = `data:image/png;base64,${base64}`;
          await image.decode();
          const output = document.createElement("canvas");
          output.width = 100;
          output.height = 100;
          const context = output.getContext("2d");
          if (!context) return 0;
          context.drawImage(image, 0, 0, 100, 100);
          const data = context.getImageData(0, 0, 100, 100).data;
          let colored = 0;
          for (let i = 0; i < data.length; i += 4) {
            if (
              (data[i] ?? 0) + (data[i + 1] ?? 0) + (data[i + 2] ?? 0) > 100 &&
              (data[i + 3] ?? 0) > 200
            )
              colored++;
          }
          return colored;
        }, rendered);
      },
      {
        // Software WebGL capture can exceed the default 5 s before the first count
        // returns. This bounds rendering + screenshot + decode, not just tile loading.
        timeout: 20_000,
        message: "Bundled globe renders opaque nonblack pixels without remote imagery",
      },
    )
    .toBeGreaterThan(300);
  expect(ionRequests).toEqual([]);
  await page.screenshot({ path: test.info().outputPath("no-token-globe.png") });
  await page.goto("/globe-embed.html?fixture=1");
  await expect.poll(() => page.evaluate(() => Boolean(window.globeBridge))).toBe(true);
  await expect(page.locator("body")).toHaveAttribute("data-scan-state", "ready");
  await page.screenshot({ path: test.info().outputPath("fixture-scan.png") });
});
