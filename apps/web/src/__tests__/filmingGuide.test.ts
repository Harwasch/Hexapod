import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountFilmingGuide } from "@/upload/filmingGuide";

import uploadHtml from "../../upload.html?raw";

/** Use the actual page so the copy, controls and placement are covered together. */
function loadPage() {
  document.body.innerHTML = new DOMParser().parseFromString(uploadHtml, "text/html").body.innerHTML;
  mountFilmingGuide();
  return {
    panel: document.querySelector<HTMLElement>("#filming-guide")!,
    dismiss: document.querySelector<HTMLButtonElement>("#filming-guide-dismiss")!,
    reopen: document.querySelector<HTMLButtonElement>("#filming-guide-reopen")!,
  };
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = "";
  window.localStorage.clear();
});

describe("the filming guide", () => {
  it("shows six tips before the file picker on the first visit", () => {
    const { panel, reopen } = loadPage();
    expect(panel).toBeVisible();
    expect(reopen).not.toBeVisible();
    expect(panel.querySelector("h2")).toHaveTextContent("How to film a good scan");
    expect(
      Array.from(panel.querySelectorAll("li"), (li) =>
        (li.textContent ?? "").replace(/\s+/g, " ").trim(),
      ),
    ).toEqual([
      "Film for 30–90 seconds, walking slowly; no fast pans.",
      "Walk all the way around the subject or area: a full circle.",
      "Do two loops: one at chest height looking slightly down, one low at knee height.",
      "Keep the subject in frame and the light steady; daylight, no zoom.",
      "Turn on Location for the Camera (Settings → Privacy → Location Services → Camera), so the video carries its GPS.",
      "For two dates of the same place, start from the same spot and film the same way.",
    ]);
    expect(panel.compareDocumentPosition(document.querySelector("#file")!)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("remembers dismissal across a fresh page init with the same storage", () => {
    const { panel, dismiss, reopen } = loadPage();
    dismiss.click();
    expect(panel).not.toBeVisible();
    expect(reopen).toBeVisible();
    expect(reopen).toHaveFocus();
    expect(reopen).toHaveAttribute("aria-expanded", "false");

    const reloaded = loadPage();
    expect(reloaded.panel).not.toBeVisible();
    expect(reloaded.reopen).toBeVisible();
  });

  it('reopens with "How to film" and remembers that choice', () => {
    loadPage().dismiss.click();
    const { panel, reopen } = loadPage();
    expect(reopen).toHaveTextContent("How to film");
    reopen.click();
    expect(panel).toBeVisible();
    expect(reopen).not.toBeVisible();
    expect(reopen).toHaveAttribute("aria-expanded", "true");
    expect(panel.querySelector("h2")).toHaveFocus();
    expect(loadPage().panel).toBeVisible();
  });

  it.each(["methods throw", "access throws", "missing"])(
    "still shows, dismisses and reopens for this visit when storage is %s",
    (failure) => {
      const unavailable = () => {
        throw new Error("Storage unavailable");
      };
      if (failure === "methods throw") {
        vi.spyOn(Storage.prototype, "getItem").mockImplementation(unavailable);
        vi.spyOn(Storage.prototype, "setItem").mockImplementation(unavailable);
      } else {
        const storage = vi.spyOn(window, "localStorage", "get");
        if (failure === "access throws") storage.mockImplementation(unavailable);
        else storage.mockReturnValue(undefined as unknown as Storage);
      }

      const { panel, dismiss, reopen } = loadPage();
      expect(panel).toBeVisible();
      dismiss.click();
      expect(panel).not.toBeVisible();
      expect(reopen).toBeVisible();
      expect(document.querySelector("#file")).toBeEnabled();
      reopen.click();
      expect(panel).toBeVisible();
      dismiss.click();
      expect(panel).not.toBeVisible();
    },
  );
});
