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
    await expect(card.getByTestId("capture-status")).toHaveText("not-started");
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

  test("a queued job says it is queued, because nothing runs it yet", async ({ page }) => {
    // No worker exists until A7, so this is not a simulation of failure: it is what the
    // real system does today, and the panel has to read as waiting rather than broken.
    const state = await boot(page);
    await openPanel(page);
    await page.getByTestId("capture-file-input").setInputFiles(payload(8));
    const card = page.getByTestId("capture-card-capture-1");
    await expect(card.getByTestId("capture-process")).toBeVisible();
    await card.getByTestId("capture-process").click();

    await expect(card.getByTestId("capture-job")).toBeVisible();
    await expect(card.getByTestId("capture-job-status")).toHaveText("not-started");
    await expect(card.getByTestId("capture-job-queued")).toBeVisible();
    await expect(card.getByTestId("capture-stage")).toHaveCount(0);
    expect(state.jobs[0]).toMatchObject({ recipe: "photo-reconstruct", status: "not-started" });
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
    await expect(card.getByTestId("capture-job-status")).toHaveText("complete", {
      timeout: 40_000,
    });
    await expect(card.getByTestId("capture-stage")).toHaveCount(5);
    await expect(card.getByTestId("capture-fly-to")).toBeVisible();
  });

  test("with the API offline the drop zone says so and refuses files", async ({ page }) => {
    await boot(page, { apiDown: true });
    await openPanel(page);
    await expect(page.getByTestId("captures-offline")).toBeVisible();
    await expect(page.getByTestId("capture-dropzone")).toHaveClass(/dropzone--disabled/);
    await expect(page.getByTestId("capture-file-input")).toBeDisabled();
  });
});
