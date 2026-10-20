/**
 * "Find my land": opens the site finder (finder.html, src/finder/) in a dialog and takes back
 * what the visitor chose. The finder is a page of its own on purpose: it runs the app's globe,
 * and in a frame CesiumJS and the app's styles stay out of this page (check-bundle.mjs holds
 * the landing page to that, on demand too). The frame is made on the first open, so nothing
 * of the finder loads until a visitor asks for it.
 */
import { readFinderMessage, type FoundLand } from "../finder/handoff";

export interface FinderOptions {
  open: HTMLButtonElement;
  dialog: HTMLDialogElement;
  onFound: (land: FoundLand) => void;
}

/** Wires the button to the dialog. Returns a function that opens it. */
export function startFinder({ open, dialog, onFound }: FinderOptions): () => void {
  let frame: HTMLIFrameElement | null = null;
  const show = () => {
    if (!frame) {
      frame = document.createElement("iframe");
      frame.title = "Find your land";
      frame.src = "/finder.html";
      frame.allow = "fullscreen";
      dialog.append(frame);
    }
    dialog.showModal();
  };
  open.addEventListener("click", show);
  window.addEventListener("message", (event) => {
    // Only the frame this page made, from this page's own origin.
    if (event.source !== frame?.contentWindow) return;
    if (event.origin !== window.location.origin) return;
    const message = readFinderMessage(event.data);
    if (!message) return;
    if (message.type === "found") onFound(message.land);
    dialog.close();
    open.focus();
  });
  return show;
}
