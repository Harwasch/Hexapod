/**
 * The review switcher (`?review`): Harrison compares the 3D scenes' looks (looks.ts) on the page
 * itself. Loaded only with `?review`; it goes once the
 * choices are made.
 */
import { LOOKS } from "./looks";

function go(change: (url: URL) => void) {
  const url = new URL(window.location.href);
  change(url);
  window.location.assign(url);
}

export function mountReview(lookId: string): void {
  const panel = document.createElement("aside");
  panel.className = "review";
  panel.setAttribute("aria-label", "Review options");
  const index = Math.max(
    0,
    LOOKS.findIndex((look) => look.id === lookId),
  );
  const current = LOOKS[index];

  const title = document.createElement("p");
  title.className = "review__title";
  title.textContent = `Look ${index + 1} of ${LOOKS.length}`;

  const select = document.createElement("select");
  select.setAttribute("aria-label", "Look");
  for (const look of LOOKS) {
    const option = document.createElement("option");
    option.value = look.id;
    option.textContent = look.name;
    option.selected = look.id === current?.id;
    select.append(option);
  }
  select.addEventListener("change", () => go((url) => url.searchParams.set("look", select.value)));

  const step = (by: number, label: string) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      const next = LOOKS[(index + by + LOOKS.length) % LOOKS.length];
      if (next) go((url) => url.searchParams.set("look", next.id));
    });
    return button;
  };

  const blurb = document.createElement("p");
  blurb.className = "review__blurb";
  blurb.textContent = current?.blurb ?? "";

  const row = document.createElement("div");
  row.className = "review__row";
  row.append(step(-1, "‹"), select, step(1, "›"));

  panel.append(title, row, blurb);
  document.body.append(panel);
}
