import type { Page } from "@playwright/test";

import { expect, mockApi, test, type CaptureMockState, type MockOptions } from "./fixtures";

/** Boots the app with the capture mock configured, without the shared `app` fixture. */
async function boot(page: Page, options: MockOptions = {}): Promise<CaptureMockState> {
  const state = await mockApi(page, options);
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "twin.settings.v1",
      JSON.stringify({
        state: { onboardingDismissed: true, quality: "performance" },
        version: 2,
      }),
    );
  });
  await page.goto("/");
  await expect(page.locator("canvas").first()).toBeVisible();
  return state;
}

async function openPanel(page: Page): Promise<void> {
  await page.getByTestId("tool-captures").click();
  await expect(page.getByTestId("captures-panel")).toBeVisible();
}

/** A file of `bytes` bytes, which the mock's 4-byte parts cut into `bytes / 4` parts. */
function payload(bytes: number) {
  return {
    name: "orchard.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.alloc(bytes, "x"),
  };
}

test.describe("the Captures panel", () => {
  test("a dropped file uploads part by part and the capture appears", async ({ page }) => {
    const state = await boot(page);
    await openPanel(page);

    await page.getByTestId("capture-file-input").setInputFiles(payload(8));

    const card = page.getByTestId("capture-card-capture-1");
    await expect(card).toBeVisible();
    await expect(card.getByTestId("capture-status")).toHaveAttribute("data-status", "not-started");
    // The badge says it in plain words; the API's value is kept on the attribute.
    await expect(card.getByTestId("capture-status")).toHaveText("Ready");
    await expect(card.getByTestId("capture-file")).toContainText("orchard.mp4");
    // The bar is a real progressbar, and it ends where the file ends.
    const bar = card.getByRole("progressbar").first();
    await expect(bar).toHaveAttribute("aria-valuenow", "8");
    await expect(bar).toHaveAttribute("aria-valuemax", "8");

    // The DOM is full before `.../complete` has necessarily answered, so the wire-level
    // assertions wait for the upload to actually finish rather than for it to look done.
    await expect.poll(() => state.completed.length).toBe(1);
    // Two 4-byte parts went to storage, and neither carried the write token: the
    // presigned URL is the credential, and an extra Authorization header breaks SigV4.
    expect(state.partPuts.map((put) => put.partNumber)).toEqual([1, 2]);
    expect(state.partPuts.every((put) => put.bytes === 4)).toBe(true);
    expect(state.partPuts.every((put) => put.authorization === null)).toBe(true);
    // The ETags the browser read back off each part PUT are what completion is made of.
    expect(state.completed).toEqual([
      [
        { partNumber: 1, etag: "etag-1" },
        { partNumber: 2, etag: "etag-2" },
      ],
    ]);
  });

  test("an upload longer than one presigned window walks the windows", async ({ page }) => {
    // 20 bytes at 4 bytes a part is 5 parts; a window is 2, so three windows are needed.
    const state = await boot(page, { partSize: 4, windowParts: 2 });
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(20));

    await expect(page.getByTestId("capture-card-capture-1")).toBeVisible();
    await expect.poll(() => state.completed.length, { message: "the file completes" }).toBe(1);
    expect(state.partPuts.map((put) => put.partNumber)).toEqual([1, 2, 3, 4, 5]);
    // Windows after the first are asked for by `nextPartNumber`, never from part 1 again.
    expect(state.presigns).toEqual([3, 5]);
    expect(state.completed[0]?.map((part) => part.partNumber)).toEqual([1, 2, 3, 4, 5]);
  });

  test("a part that fails is retried from where it stopped, not from the start", async ({
    page,
  }) => {
    // 12 bytes at 4 bytes a part is three parts, and the second one is refused once.
    const state = await boot(page, { failPartOnce: 2 });
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(12));

    const card = page.getByTestId("capture-card-capture-1");
    await expect(card.getByTestId("capture-upload-error")).toContainText("Object storage refused");
    expect(state.completed).toHaveLength(0);

    await card.getByTestId("capture-retry-upload").click();
    await expect.poll(() => state.completed.length).toBe(1);

    // Part 1 is not sent twice: the resume re-presigns the window at part 2 and carries
    // on. The API's progress watermark never moves backwards either.
    expect(state.partPuts.map((put) => put.partNumber)).toEqual([1, 2, 2, 3]);
    expect(state.presigns[0]).toBe(2);
    expect(state.completed[0]?.map((part) => part.partNumber)).toEqual([1, 2, 3]);
    // One file row, not two: the retry resumed the registered upload.
    expect((state.captures[0] as { files: unknown[] }).files).toHaveLength(1);
  });

  test("a 401 on a write asks for the token, and the upload then succeeds", async ({ page }) => {
    const state = await boot(page, { writeToken: "letmein" });
    await openPanel(page);

    // Nothing is asked for up front — a deployment with no token leaves writes open.
    await expect(page.getByTestId("write-token-form")).toHaveCount(0);

    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    await expect(page.getByTestId("write-token-form")).toBeVisible();
    await expect(page.getByTestId("capture-error")).toContainText("write token");
    expect(state.unauthorized).toContain("POST /api/v1/captures");
    expect(state.captures).toHaveLength(0);

    await page.getByTestId("write-token-input").fill("letmein");
    await page.getByTestId("write-token-save").click();

    await expect(page.getByTestId("capture-card-capture-1")).toBeVisible();
    await expect(page.getByTestId("write-token-form")).toHaveCount(0);
    await expect.poll(() => state.completed.length).toBe(1);
    // Stored where the other preferences are, not in a VITE_ variable in the bundle.
    const stored = await page.evaluate(() => window.localStorage.getItem("twin.settings.v1"));
    expect(stored).toContain("letmein");
  });

  test("a job with no worker free to claim it says it is queued", async ({ page }) => {
    // A7 built the worker, so this is no longer "nothing runs it": it is the window
    // between the API inserting the row and a worker claiming it, which is a real state
    // and has to read as waiting rather than as broken. The mock holds the worker off.
    const state = await boot(page);
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    const card = page.getByTestId("capture-card-capture-1");
    await expect(card.getByTestId("capture-process")).toBeVisible();
    await card.getByTestId("capture-process").click();

    await expect(card.getByTestId("capture-job")).toBeVisible();
    await expect(card.getByTestId("capture-job-status")).toHaveAttribute(
      "data-status",
      "not-started",
    );
    await expect(card.getByTestId("capture-job-queued")).toBeVisible();
    await expect(card.getByTestId("capture-stage")).toHaveCount(0);
    expect(state.jobs[0]).toMatchObject({ recipe: "photo-reconstruct", status: "not-started" });
  });

  test("a running job can be cancelled from the card", async ({ page }) => {
    const state = await boot(page, { workerRuns: true });
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    const card = page.getByTestId("capture-card-capture-1");
    await card.getByTestId("capture-process").click();

    await expect(card.getByTestId("capture-job-status")).toHaveAttribute(
      "data-status",
      "in-progress",
      {
        timeout: 40_000,
      },
    );
    await card.getByTestId("capture-job-cancel").click();

    await expect(card.getByTestId("capture-job-status")).toHaveAttribute(
      "data-status",
      "cancelled",
    );
    // It stays cancelled: the worker stopped, it did not carry on to the next stage.
    await expect(card.getByTestId("capture-job-status")).toHaveAttribute(
      "data-status",
      "cancelled",
      {
        timeout: 10_000,
      },
    );
    expect(state.jobs[0]).toMatchObject({ status: "cancelled" });
  });

  test("a failed job is retried from the stage that failed, keeping the ones before it", async ({
    page,
  }) => {
    await boot(page, { workerRuns: true, workerFailsAt: "train" });
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    const card = page.getByTestId("capture-card-capture-1");
    await card.getByTestId("capture-process").click();

    await expect(card.getByTestId("capture-job-status")).toHaveAttribute("data-status", "error", {
      timeout: 40_000,
    });
    await expect(card.getByTestId("capture-job-error")).toContainText("will not be retried");
    const retry = card.getByTestId("capture-job-retry");
    await expect(retry).toHaveText(/Retry from Train/);
    // Starting over from scratch is not offered: the earlier stages' work is still there.
    await expect(card.getByTestId("capture-process")).toHaveCount(0);

    await retry.click();

    // The two stages that succeeded are kept, and the run picks up at the third.
    await expect(card.getByTestId("capture-job-status")).toHaveAttribute(
      "data-status",
      "complete",
      {
        timeout: 40_000,
      },
    );
    await expect(card.getByTestId("capture-stage")).toHaveCount(5);
  });

  test("a stage's log opens in the card, read from object storage", async ({ page }) => {
    await boot(page, { workerRuns: true });
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    const card = page.getByTestId("capture-card-capture-1");
    await card.getByTestId("capture-process").click();

    await expect(card.getByTestId("capture-stage").first()).toContainText("Normalize");
    await card.getByTestId("capture-stage-log-toggle").first().click();

    await expect(card.getByTestId("capture-stage-log").first()).toContainText("wrote 1 artifact");
  });

  test("once a worker runs it, the card becomes a live stage list", async ({ page }) => {
    await boot(page, { workerRuns: true });
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    const card = page.getByTestId("capture-card-capture-1");
    await expect(card.getByTestId("capture-process")).toBeVisible();
    await card.getByTestId("capture-process").click();

    // The mock worker advances one stage per poll of the job list.
    await expect(card.getByTestId("capture-stage").first()).toContainText("Normalize");
    await expect(card.getByTestId("capture-job-status")).toHaveAttribute(
      "data-status",
      "complete",
      {
        timeout: 40_000,
      },
    );
    await expect(card.getByTestId("capture-stage")).toHaveCount(5);
    await expect(card.getByTestId("capture-fly-to")).toBeVisible();
  });

  test("a new capture from phone makes an empty capture and shows its QR code", async ({
    page,
  }) => {
    const state = await boot(page);
    await openPanel(page);

    await page.getByTestId("capture-from-phone").click();

    // One capture, with no files, waiting for the phone -- and the code is already up,
    // without a second click on the card's own "Add from phone".
    const qr = page.getByTestId("handoff-qr");
    await expect(qr).toBeVisible();
    await expect.poll(() => state.handoffs).toEqual(["capture-1"]);
    expect(state.captures).toHaveLength(1);
    const created = state.captures[0] as { files: unknown[]; metadata: Record<string, unknown> };
    expect(created.files).toEqual([]);
    // No recipe: nothing has been picked, so the card proposes one from what arrives.
    expect(created.metadata.origin).toBe("phone");
    expect(created.metadata.recipe).toBeUndefined();
    expect(typeof created.metadata.lat).toBe("number");
    await expect(
      page.getByTestId("capture-card-capture-1").getByTestId("capture-status"),
    ).toHaveAttribute("data-status", "awaiting-files");

    // Closing it brings the button back rather than minting another code on its own.
    await page.getByRole("button", { name: "Close the phone handoff" }).first().click();
    await expect(page.getByTestId("capture-from-phone")).toBeVisible();
    expect(state.handoffs).toEqual(["capture-1"]);
  });

  test("a video added to a capture that began as a splat is reconstructed, not packaged", async ({
    page,
  }) => {
    const state = await boot(page);
    const now = new Date().toISOString();
    // A splat was dropped (so `splat-ingest` was proposed and stored), then a phone sent
    // a video into the same capture. The files decide the recipe, not the proposal.
    state.captures.push({
      id: "capture-1",
      slug: "capture-1",
      name: "Mixed",
      kind: "gaussian-splat",
      description: null,
      status: "not-started",
      siteId: null,
      device: null,
      sensor: null,
      capturedAt: null,
      temporalExtent: null,
      georefMethod: null,
      scaleSource: null,
      uncertaintyM: null,
      license: null,
      provenance: null,
      attribution: [],
      metadata: { recipe: "splat-ingest", origin: "console", lat: 1, lon: 2 },
      files: [
        {
          id: "file-1",
          captureId: "capture-1",
          filename: "walkaround.mov",
          contentType: "video/quicktime",
          bytes: 8,
          checksum: null,
          storageKey: "captures/capture-1/source/file-1/walkaround.mov",
          status: "complete",
          uploadId: null,
          partsCompleted: 1,
          partsTotal: 1,
          createdAt: now,
          updatedAt: now,
        },
      ],
      createdAt: now,
      updatedAt: now,
    });
    await openPanel(page);

    const card = page.getByTestId("capture-card-capture-1");
    await card.getByTestId("capture-process").click();
    await expect.poll(() => state.jobs.length).toBe(1);
    expect(state.jobs[0]).toMatchObject({ recipe: "photo-reconstruct" });
  });

  test("with the API offline the drop zone says so and refuses files", async ({ page }) => {
    await boot(page, { apiDown: true });
    await openPanel(page);
    await expect(page.getByTestId("captures-offline")).toBeVisible();
    await expect(page.getByTestId("capture-dropzone")).toHaveClass(/dropzone--disabled/);
    await expect(page.getByTestId("capture-file-input")).toBeDisabled();
  });
});
