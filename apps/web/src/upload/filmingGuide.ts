/** Remember this browser's choice without making storage a requirement for uploading. */
const STORAGE_KEY = "twin.filmingGuide.dismissed.v1";

export function mountFilmingGuide(root: Document = document): void {
  const panel = root.getElementById("filming-guide");
  const dismiss = root.getElementById("filming-guide-dismiss");
  const reopen = root.getElementById("filming-guide-reopen");
  const heading = root.getElementById("filming-guide-title");
  if (!panel || !dismiss || !reopen || !heading) return;

  const show = (visible: boolean): void => {
    panel.hidden = !visible;
    reopen.hidden = visible;
    reopen.setAttribute("aria-expanded", String(visible));
  };
  const remember = (dismissed: boolean): void => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(dismissed));
    } catch {
      // With storage unavailable, the controls still work for this visit.
    }
  };

  let dismissed = false;
  try {
    dismissed = window.localStorage.getItem(STORAGE_KEY) === "true";
  } catch {
    // First visit, or storage blocked: show the advice.
  }
  show(!dismissed);

  dismiss.addEventListener("click", () => {
    show(false);
    remember(true);
    reopen.focus();
  });
  reopen.addEventListener("click", () => {
    show(true);
    remember(false);
    heading.focus();
  });
}
