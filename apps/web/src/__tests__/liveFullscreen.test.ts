import { afterEach, expect, it, vi } from "vitest";
import { liveFullscreen } from "@/view/liveFullscreen";

afterEach(() => {
  document.body.replaceChildren();
});

it("expands without the Fullscreen API and cleans up Escape and button handlers", () => {
  const root = document.createElement("div"),
    panel = document.createElement("section"),
    button = document.createElement("button");
  root.append(panel);
  panel.append(button);
  document.body.append(root);
  const stop = liveFullscreen(panel, button, root);
  button.click();
  expect(panel.classList.contains("lc-expanded")).toBe(true);
  expect(root.classList.contains("lc-fullscreen")).toBe(true);
  expect(button.getAttribute("aria-label")).toBe("Exit fullscreen");
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  expect(panel.classList.contains("lc-expanded")).toBe(false);
  button.click();
  button.click();
  expect(panel.classList.contains("lc-expanded")).toBe(false);
  button.click();
  stop();
  expect(panel.classList.contains("lc-expanded")).toBe(false);
  button.click();
  expect(panel.classList.contains("lc-expanded")).toBe(false);
});

it("keeps the in-page fallback when the browser denies fullscreen", async () => {
  const root = document.createElement("div"),
    panel = document.createElement("section"),
    button = document.createElement("button");
  const request = vi.fn().mockRejectedValue(new Error("Unavailable"));
  panel.requestFullscreen = request;
  const stop = liveFullscreen(panel, button, root);
  button.click();
  await Promise.resolve();
  await Promise.resolve();
  expect(request).toHaveBeenCalledOnce();
  expect(panel.classList.contains("lc-expanded")).toBe(true);
  stop();
});
