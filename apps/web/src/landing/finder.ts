/**
 * "Find my land": opens the site finder (finder.html, src/finder/) in the page, in the request
 * form, and takes back what the visitor chose. The finder is a page of its own on purpose: it
 * runs the app's globe, and in a frame CesiumJS and the app's styles stay out of this page
 * (check-bundle.mjs holds the landing page to that, on demand too). The frame is made on the
 * first open, so nothing of the finder loads until a visitor asks for it. Its page is the one
 * the site lets be framed, and only by this site (infra/pages/_headers).
 */
import { readFinderMessage, type FoundLand } from "../finder/handoff";

export interface FinderOptions {
  /** The button that opens (and closes) the finder. */
  open: HTMLButtonElement;
  /** Where the finder appears, in the page. */
  region: HTMLElement;
  onFound: (land: FoundLand) => void;
}

const OPEN_LABEL = "Find my land";

/** Wires the button to the finder's place in the page. Returns a function that opens it. */
export function startFinder({ open, region, onFound }: FinderOptions): () => void {
  let frame: HTMLIFrameElement | null = null;
  let closedLabel = OPEN_LABEL;
  const show = () => {
    if (!frame) {
      frame = document.createElement("iframe");
      frame.title = "Find your land";
      frame.src = "/finder.html";
      frame.allow = "fullscreen";
      region.append(frame);
    }
    region.hidden = false;
    open.setAttribute("aria-expanded", "true");
    open.textContent = "Hide the map";
    region.scrollIntoView({ block: "nearest" });
  };
  const hide = () => {
    region.hidden = true;
    open.setAttribute("aria-expanded", "false");
    open.textContent = closedLabel;
  };
  open.addEventListener("click", () => (region.hidden ? show() : hide()));
  window.addEventListener("message", (event) => {
    // Only the frame this page made, from this page's own origin.
    if (event.source !== frame?.contentWindow) return;
    if (event.origin !== window.location.origin) return;
    const message = readFinderMessage(event.data);
    if (!message) return;
    if (message.type === "found") {
      onFound(message.land);
      closedLabel = "Change my land";
    }
    hide();
    open.focus();
  });
  return show;
}
