import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import type * as Clip from "../src/worlds/intelligence/clip";

test("replay highlights encode a local clip without uploading the recording", async ({ page }) => {
  await page.goto("/worlds.html");
  const uploads: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") uploads.push(request.url());
  });
  const fixture = await readFile(new URL("./fixtures/replay-test.webm", import.meta.url));
  const result = await page.evaluate(async (encoded) => {
    const original = new Blob([Uint8Array.from(atob(encoded), (value) => value.charCodeAt(0))], {
      type: "video/webm",
    });
    const path = "/src/worlds/intelligence/clip.ts";
    const module = (await import(path)) as typeof Clip;
    const clip = await module.extractReplayClip(original, 0.2, 3.5, new AbortController().signal);
    const video = document.createElement("video");
    const url = URL.createObjectURL(clip);
    const loaded = new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error("Exported clip does not decode"));
    });
    video.src = url;
    await loaded;
    const dimensions = [video.videoWidth, video.videoHeight];
    URL.revokeObjectURL(url);
    return { dimensions, originalBytes: original.size, clipBytes: clip.size, mime: clip.type };
  }, fixture.toString("base64"));
  expect(result.dimensions).toEqual([160, 90]);
  expect(result.originalBytes).toBeGreaterThan(100);
  expect(result.clipBytes).toBeGreaterThan(100);
  expect(result.mime).toContain("video/");
  expect(uploads).toEqual([]);
});
