/**
 * The standalone page a phone opens after scanning the handoff QR code.
 *
 * This is the one surface the console's specs cannot reach: `upload.html` is a separate
 * Vite entry with no React, no stores and no CesiumJS, so nothing in `app.spec.ts` or
 * `captures.spec.ts` loads a line of it. It also runs on a phone-sized viewport, which is
 * the only place its layout is ever exercised.
 *
 * The tests below drive the real page against a mocked API, and the last one asserts the
 * property the whole separate-entry decision exists for: that the page does not pull in
 * CesiumJS.
 */
import { expect, test } from "@playwright/test";

const PHONE = { width: 390, height: 844 }; // iPhone 14-ish, portrait.

const CAPTURE_ID = "11111111-1111-4111-8111-111111111111";
const FILE_ID = "22222222-2222-4222-8222-222222222222";

/** `version.captureIdHex.expiresAt.ceiling.signature` — the page only parses; the API verifies. */
function token(expiresAt: number, captureHex = CAPTURE_ID.replace(/-/g, "")): string {
  const ceiling = expiresAt + 3600;
  return `v1.${captureHex}.${String(expiresAt)}.${String(ceiling)}.signature-not-checked-here`;
}

const soon = () => Math.floor(Date.now() / 1000) + 600;
const past = () => Math.floor(Date.now() / 1000) - 60;

test.use({ viewport: PHONE });

test.describe("the phone upload page", () => {
  test("a valid link offers a file picker", async ({ page }) => {
    await page.goto(`/upload.html#${token(soon())}`);
    await expect(page.getByRole("heading", { name: "Send a capture" })).toBeVisible();
    await expect(page.locator("#file")).toBeAttached();
    await expect(page.locator("#status")).toContainText("Pick a file");
  });

  test("with no link the page asks for the phone key, once", async ({ page }) => {
    await page.route("**/api/v1/phone/check", async (route) => {
      const good = route.request().headers().authorization === "Bearer abcd-efgh-jkmn";
      await route.fulfill({ status: good ? 204 : 401, body: good ? "" : "{}" });
    });
    await page.goto("/upload.html");
    await expect(page.locator("#keyform")).toBeVisible();
    await expect(page.locator("#form")).toBeHidden();

    await page.locator("#key").fill("abcd-efgh-jkmm");
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.locator("#status")).toContainText("isn't right");

    // Typed the way a phone types it: capitals and a trailing space are forgiven.
    await page.locator("#key").fill("ABCD-EFGH-JKMN ");
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.locator("#form")).toBeVisible();
    await expect(page.locator("#keyform")).toBeHidden();

    // Remembered: a reload goes straight to the picker.
    await page.reload();
    await expect(page.locator("#form")).toBeVisible();
    await expect(page.locator("#keyform")).toBeHidden();
  });

  test("an expired link says so before asking for a file", async ({ page }) => {
    await page.goto(`/upload.html#${token(past())}`);
    await expect(page.locator("#status")).toContainText("expired");
    await expect(page.locator("#form")).toBeHidden();
  });

  test("a malformed token is refused", async ({ page }) => {
    await page.goto("/upload.html#not-a-token");
    await expect(page.locator("#status")).toContainText("not a valid handoff");
    await expect(page.locator("#form")).toBeHidden();
  });

  test("picking a file uploads it part by part and says it is done", async ({ page }) => {
    const puts: string[] = [];

    await page.route("**/api/v1/captures/*/files", async (route) => {
      // The handoff token, not the write token, is what authorises this.
      expect(route.request().headers().authorization).toContain("Bearer v1.");
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          file: { id: FILE_ID, captureId: CAPTURE_ID, filename: "clip.mov", partsTotal: 2 },
          upload: {
            uploadId: "u-1",
            storageKey: `captures/${CAPTURE_ID}/source/${FILE_ID}/clip.mov`,
            partSize: 8,
            partsTotal: 2,
            nextPartNumber: null,
            expiresIn: 3600,
            parts: [
              { partNumber: 1, url: "https://storage.example/part-1" },
              { partNumber: 2, url: "https://storage.example/part-2" },
            ],
          },
        }),
      });
    });

    // Storage, not the API. A presigned URL is the credential, so this must carry no
    // Authorization header at all -- asserted, because an extra header is exactly what
    // makes a SigV4 signature stop matching.
    await page.route("https://storage.example/**", async (route) => {
      expect(route.request().headers().authorization).toBeUndefined();
      puts.push(new URL(route.request().url()).pathname);
      await route.fulfill({
        status: 200,
        // Both headers matter, and the second one is the whole point of
        // infra/cors/upload.json: without Access-Control-Expose-Headers the browser
        // receives the ETag and refuses to show it to the script, and completion --
        // which is a list of part ETags -- becomes impossible. Written out here rather
        // than hidden in a helper because a mock that omits it fails in exactly the way
        // a misconfigured bucket does.
        headers: {
          ETag: '"etag-x"',
          "access-control-allow-origin": "*",
          "access-control-expose-headers": "ETag",
        },
        body: "",
      });
    });

    await page.route("**/api/v1/captures/*/files/*/complete", async (route) => {
      const body = route.request().postDataJSON() as { parts: { partNumber: number }[] };
      expect(body.parts.map((p) => p.partNumber)).toEqual([1, 2]);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ id: FILE_ID, status: "complete" }),
      });
    });

    await page.goto(`/upload.html#${token(soon())}`);
    await page.locator("#file").setInputFiles({
      name: "clip.mov",
      mimeType: "video/quicktime",
      buffer: Buffer.alloc(16, 7),
    });

    await expect(page.locator("#status")).toContainText("on its way", { timeout: 20_000 });
    expect(puts).toEqual(["/part-1", "/part-2"]);
  });

  test("several photos upload one after another into the same capture", async ({ page }) => {
    const registered: string[] = [];
    const completed: string[] = [];

    await page.route("**/api/v1/captures/*/files", async (route) => {
      const { filename } = route.request().postDataJSON() as { filename: string };
      registered.push(filename);
      const id = `${FILE_ID.slice(0, -1)}${String(registered.length)}`;
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          file: { id, captureId: CAPTURE_ID, filename, partsTotal: 1 },
          upload: {
            uploadId: `u-${String(registered.length)}`,
            storageKey: `captures/${CAPTURE_ID}/source/${id}/${filename}`,
            partSize: 8,
            partsTotal: 1,
            nextPartNumber: null,
            expiresIn: 3600,
            parts: [{ partNumber: 1, url: `https://storage.example/${filename}` }],
          },
        }),
      });
    });
    await page.route("https://storage.example/**", async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          ETag: '"etag-x"',
          "access-control-allow-origin": "*",
          "access-control-expose-headers": "ETag",
        },
        body: "",
      });
    });
    await page.route("**/api/v1/captures/*/files/*/complete", async (route) => {
      completed.push(route.request().url());
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "complete" }),
      });
    });

    await page.goto(`/upload.html#${token(soon())}`);
    await page.locator("#file").setInputFiles([
      { name: "a.jpg", mimeType: "image/jpeg", buffer: Buffer.alloc(8, 1) },
      { name: "b.jpg", mimeType: "image/jpeg", buffer: Buffer.alloc(8, 2) },
      { name: "c.jpg", mimeType: "image/jpeg", buffer: Buffer.alloc(8, 3) },
    ]);

    await expect(page.locator("#status")).toContainText("3 files are on their way", {
      timeout: 20_000,
    });
    expect(registered).toEqual(["a.jpg", "b.jpg", "c.jpg"]);
    expect(completed).toHaveLength(3);
  });

  test("with the key, a picked video becomes a placed capture and starts processing", async ({
    page,
    context,
  }) => {
    await context.grantPermissions(["geolocation"]);
    await context.setGeolocation({ latitude: 44.9778, longitude: -93.265, accuracy: 7 });
    await page.addInitScript(() => {
      window.localStorage.setItem("twin.phoneKey", "abcd-efgh-jkmn");
    });

    const created: unknown[] = [];
    const processed: unknown[] = [];
    const tokens: string[] = [];
    await page.route("**/api/v1/phone/captures", async (route) => {
      expect(route.request().headers().authorization).toBe("Bearer abcd-efgh-jkmn");
      created.push(route.request().postDataJSON());
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          capture: { id: CAPTURE_ID, name: "Phone capture", files: [], metadata: {} },
          uploadToken: "h1.first",
        }),
      });
    });
    await page.route("**/api/v1/captures/*/files", async (route) => {
      tokens.push(String(route.request().headers().authorization));
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        // The renewal: every handoff-authorised response carries the next token.
        headers: {
          "x-handoff-token": "h1.second",
          "access-control-expose-headers": "X-Handoff-Token",
        },
        body: JSON.stringify({
          file: { id: FILE_ID, captureId: CAPTURE_ID, filename: "walk.mov", partsTotal: 1 },
          upload: {
            uploadId: "u-1",
            storageKey: "k",
            partSize: 8,
            partsTotal: 1,
            nextPartNumber: null,
            expiresIn: 3600,
            parts: [{ partNumber: 1, url: "https://storage.example/part-1" }],
          },
        }),
      });
    });
    await page.route("https://storage.example/**", async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          ETag: '"etag-x"',
          "access-control-allow-origin": "*",
          "access-control-expose-headers": "ETag",
        },
        body: "",
      });
    });
    await page.route("**/api/v1/captures/*/files/*/complete", async (route) => {
      tokens.push(String(route.request().headers().authorization));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: "complete" }),
      });
    });
    await page.route("**/api/v1/phone/captures/*/process", async (route) => {
      processed.push(route.request().postDataJSON());
      await route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify({ id: "job-1", status: "not-started", steps: [] }),
      });
    });

    await page.goto("/upload.html");
    await page.locator("#file").setInputFiles({
      name: "walk.mov",
      mimeType: "video/quicktime",
      buffer: Buffer.alloc(8, 5),
    });

    await expect(page.locator("#status")).toContainText("processing has started", {
      timeout: 20_000,
    });
    expect(created).toEqual([{ lat: 44.9778, lon: -93.265, accuracyM: 7 }]);
    expect(processed).toEqual([{ recipe: "photo-reconstruct" }]);
    // The first call used the token the capture came with; the next used the renewal.
    expect(tokens).toEqual(["Bearer h1.first", "Bearer h1.second"]);
  });

  test("a location request that never answers does not stop the upload", async ({ page }) => {
    // What an iOS in-app browser, or an unanswered permission prompt, looks like: the
    // callbacks are simply never called.
    await page.addInitScript(() => {
      window.localStorage.setItem("twin.phoneKey", "abcd-efgh-jkmn");
      Object.defineProperty(navigator, "geolocation", {
        value: { getCurrentPosition: () => undefined },
        configurable: true,
      });
    });
    const created: unknown[] = [];
    await page.route("**/api/v1/phone/captures", async (route) => {
      created.push(route.request().postDataJSON());
      // Refused, so this test stops at the point it is about: the capture was asked for.
      await route.fulfill({ status: 409, contentType: "application/json", body: "{}" });
    });

    await page.goto("/upload.html");
    await page.locator("#file").setInputFiles({
      name: "a.jpg",
      mimeType: "image/jpeg",
      buffer: Buffer.alloc(8, 1),
    });
    await expect.poll(() => created.length, { timeout: 15_000 }).toBe(1);
    expect(created[0]).toEqual({});
  });

  test("a part that fails while the phone sleeps is sent again, not lost", async ({ page }) => {
    await page.route("**/api/v1/captures/*/files", async (route) => {
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          file: { id: FILE_ID, captureId: CAPTURE_ID, filename: "a.jpg", partsTotal: 1 },
          upload: {
            uploadId: "u-1",
            storageKey: "k",
            partSize: 8,
            partsTotal: 1,
            nextPartNumber: null,
            expiresIn: 3600,
            parts: [{ partNumber: 1, url: "https://storage.example/part-1" }],
          },
        }),
      });
    });
    let puts = 0;
    await page.route("https://storage.example/**", async (route) => {
      puts += 1;
      // The first attempt dies the way a locked phone kills it: no response at all.
      if (puts === 1) return route.abort("connectionreset");
      await route.fulfill({
        status: 200,
        headers: {
          ETag: '"etag-x"',
          "access-control-allow-origin": "*",
          "access-control-expose-headers": "ETag",
        },
        body: "",
      });
    });
    await page.route("**/api/v1/captures/*/files/*/complete", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
    );

    await page.goto(`/upload.html#${token(soon())}`);
    await page.locator("#file").setInputFiles({
      name: "a.jpg",
      mimeType: "image/jpeg",
      buffer: Buffer.alloc(8, 1),
    });
    await expect(page.locator("#status")).toContainText("on its way", { timeout: 20_000 });
    expect(puts).toBe(2);
  });

  test("your captures show each one's real state and only the action that fits it", async ({
    page,
  }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("twin.phoneKey", "abcd-efgh-jkmn");
    });
    const file = (status: string) => ({ filename: "a.jpg", status, bytes: 8 });
    const phone = { origin: "phone-key" };
    await page.route(
      (url) => url.pathname === "/api/v1/captures",
      (route) =>
        route.fulfill({
          json: [
            {
              id: "done",
              name: "Finished",
              siteId: "site-1",
              status: "complete",
              metadata: phone,
              files: [file("complete")],
            },
            // The capture's own status stays not-started while its run goes: the state
            // has to come from the run, or this offers Process on something training.
            {
              id: "busy",
              name: "Training",
              siteId: null,
              status: "not-started",
              metadata: phone,
              files: [file("complete")],
            },
            {
              id: "half",
              name: "Half sent",
              siteId: null,
              status: "not-started",
              metadata: phone,
              files: [file("complete"), file("in-progress")],
            },
            {
              id: "idle",
              name: "Never run",
              siteId: null,
              status: "not-started",
              metadata: phone,
              files: [file("complete"), file("complete")],
            },
            {
              id: "desk",
              name: "From the desktop",
              siteId: "site-2",
              status: "complete",
              metadata: { origin: "console" },
              files: [file("complete")],
            },
          ],
        }),
    );
    const training = {
      id: "job-busy",
      captureId: "busy",
      recipe: "photo-reconstruct",
      status: "in-progress",
      createdAt: new Date().toISOString(),
      error: null,
      steps: [
        { ordinal: 0, stageId: "normalize", status: "complete", startedAt: null },
        {
          ordinal: 1,
          stageId: "train",
          status: "in-progress",
          startedAt: new Date(Date.now() - 12 * 60_000).toISOString(),
        },
      ],
    };
    await page.route(
      (url) => url.pathname === "/api/v1/jobs",
      (route) => route.fulfill({ json: [training] }),
    );
    await page.route(
      (url) => url.pathname === "/api/v1/recipes",
      (route) =>
        route.fulfill({
          json: {
            recipes: [
              {
                name: "photo-reconstruct",
                stages: ["normalize", "pose", "train", "register"].map((id) => ({ id })),
              },
            ],
          },
        }),
    );
    const processed: string[] = [];
    await page.route("**/api/v1/phone/captures/*/process", async (route) => {
      processed.push(route.request().url());
      await route.fulfill({ status: 202, json: { id: "j", status: "not-started", steps: [] } });
    });

    await page.goto("/upload.html");
    const row = (name: string) => page.locator("#mine-list li", { hasText: name });
    await expect(page.locator("#mine-list li")).toHaveCount(4);

    await expect(row("Finished").getByRole("link", { name: "View in 3D" })).toHaveAttribute(
      "href",
      "/view.html#site-1",
    );
    await expect(row("Training")).toContainText("Training the 3D model");
    await expect(row("Training")).toContainText("Step 2 of 4");
    await expect(row("Training")).toContainText("12 min");
    await expect(row("Training").getByRole("button", { name: "Process" })).toHaveCount(0);
    await expect(row("Training").getByRole("button", { name: "Progress" })).toBeVisible();

    await expect(row("Half sent")).toContainText("Upload didn't finish");
    await expect(row("Half sent").getByRole("button")).toHaveCount(0);

    await row("Never run").getByRole("button", { name: "Process" }).click();
    await expect.poll(() => processed.length).toBe(1);
    expect(processed[0]).toContain("/api/v1/phone/captures/idle/process");
  });

  test("the status panel follows a run through its stages", async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("twin.phoneKey", "abcd-efgh-jkmn");
    });
    await page.route(
      (url) => url.pathname === "/api/v1/captures",
      (route) =>
        route.fulfill({
          json: [
            {
              id: "busy",
              name: "Training",
              siteId: null,
              status: "not-started",
              metadata: { origin: "phone-key" },
              files: [{ filename: "a.jpg", status: "complete", bytes: 8 }],
            },
          ],
        }),
    );
    let polls = 0;
    await page.route(
      (url) => url.pathname === "/api/v1/jobs",
      async (route) => {
        polls += 1;
        // One failed answer must not end the polling: that was the "Queued" forever bug.
        if (polls === 2) return route.fulfill({ status: 502, body: "" });
        const started = new Date(Date.now() - 60_000).toISOString();
        await route.fulfill({
          json: [
            {
              id: "job",
              captureId: "busy",
              recipe: "photo-reconstruct",
              status: "in-progress",
              createdAt: started,
              error: null,
              steps: [{ ordinal: 0, stageId: "pose", status: "in-progress", startedAt: started }],
            },
          ],
        });
      },
    );
    await page.route(
      (url) => url.pathname === "/api/v1/recipes",
      (route) => route.fulfill({ json: { recipes: [] } }),
    );

    await page.goto("/upload.html");
    await page.locator("#mine-list li").getByRole("button", { name: "Progress" }).click();
    await expect(page.locator("#status")).toContainText("Working out where each photo was taken");
    await expect(page.locator("#detail")).toContainText("running");
  });

  test("a bucket that hides the ETag is reported as the CORS problem it is", async ({ page }) => {
    await page.route("**/api/v1/captures/*/files", async (route) => {
      await route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          file: { id: FILE_ID, captureId: CAPTURE_ID, filename: "clip.mov", partsTotal: 1 },
          upload: {
            uploadId: "u-1",
            storageKey: "k",
            partSize: 8,
            partsTotal: 1,
            nextPartNumber: null,
            expiresIn: 3600,
            parts: [{ partNumber: 1, url: "https://storage.example/part-1" }],
          },
        }),
      });
    });
    // Deliberately no access-control-expose-headers: this is a real bucket
    // misconfiguration, and the operator needs to be told which rule is missing rather
    // than watching the upload stall.
    await page.route("https://storage.example/**", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { ETag: '"etag-x"', "access-control-allow-origin": "*" },
        body: "",
      });
    });

    await page.goto(`/upload.html#${token(soon())}`);
    await page.locator("#file").setInputFiles({
      name: "clip.mov",
      mimeType: "video/quicktime",
      buffer: Buffer.alloc(8, 7),
    });

    await expect(page.locator("#status")).toContainText("ExposeHeaders", { timeout: 20_000 });
    await expect(page.locator("#status")).toContainText("infra/cors/upload.json");
  });

  test("the page does not load CesiumJS", async ({ page }) => {
    const scripts: string[] = [];
    page.on("request", (request) => {
      if (request.resourceType() === "script") scripts.push(request.url());
    });

    await page.goto(`/upload.html#${token(soon())}`);
    await expect(page.getByRole("heading", { name: "Send a capture" })).toBeVisible();

    // The entire reason upload.html is a separate entry: a phone on cellular should not
    // download a 3D globe to pick one file.
    expect(scripts.filter((url) => /cesium/i.test(url))).toEqual([]);
  });
});
