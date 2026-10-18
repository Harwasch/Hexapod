import { expect, test } from "@playwright/test";

test("voice sends final commands directly and stops on pause", async ({ page }) => {
  await page.route("**/api/v1/worlds/**", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "No GPU configured in this interface test." }),
    }),
  );
  await page.addInitScript(() => {
    interface Result {
      resultIndex: number;
      results: { 0: { transcript: string }; isFinal: boolean }[];
    }
    class Speech {
      continuous = false;
      interimResults = false;
      lang = "en-US";
      onresult: ((event: Result) => void) | null = null;
      onerror = null;
      onend: (() => void) | null = null;
      start() {
        (window as unknown as { testSpeech: Speech }).testSpeech = this;
      }
      abort() {
        this.onend?.();
      }
    }
    (window as unknown as { SpeechRecognition: typeof Speech }).SpeechRecognition = Speech;
  });
  await page.goto("/worlds.html");
  await page.getByRole("button", { name: "Try interaction preview", exact: true }).click();
  await page.getByRole("button", { name: "Use voice input", exact: true }).click();
  await expect(
    page.getByRole("checkbox", { name: "Send spoken commands immediately" }),
  ).toBeChecked();
  await page.getByRole("checkbox", { name: "Keep listening between commands" }).check();
  await page.getByRole("button", { name: "Enable voice", exact: true }).click();
  await page.getByRole("button", { name: "Use voice input", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop voice input" })).toBeVisible();
  await page.evaluate(() => {
    const speech = (window as unknown as { testSpeech: { onresult: (event: unknown) => void } })
      .testSpeech;
    speech.onresult({
      resultIndex: 0,
      results: [{ 0: { transcript: "Make the sky purple" }, isFinal: true }],
    });
  });
  await page.getByRole("button", { name: "Session details", exact: true }).click();
  await expect(page.locator(".wp-current-prompt")).toHaveText("Make the sky purple");
  await page.getByRole("button", { name: "Close panel", exact: true }).click();
  await page.getByRole("button", { name: "Pause generation", exact: true }).click();
  await expect(page.getByRole("button", { name: "Use voice input", exact: true })).toBeDisabled();
  expect(
    await page.evaluate(
      () => (window as unknown as { testSpeech: { onresult: unknown } }).testSpeech.onresult,
    ),
  ).toBeNull();
});
