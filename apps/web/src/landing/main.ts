/**
 * The landing page (landing.html): the page around the "Fleet Home" design's scenes.
 *
 * This module is small on purpose. The text, the theme toggle and the request form work as soon
 * as it runs; the 3D scenes (scene.js, three.js and the world data) load on demand behind it, and
 * when they cannot start, for want of WebGL, the hero shows a still and the fleet cards keep their
 * concept renders. No CesiumJS anywhere: scripts/check-bundle.mjs and e2e/landing.spec.ts check.
 */
import "./landing.css";

import { BRAND } from "./brand";
import { REQUEST_ENDPOINT, validateRequest, type RequestErrors, type RequestField } from "./form";
import { startSitePicker, type SitePicker } from "./picker.js";
import type * as SceneModule from "./scene.js";
import type { Hero, HeroLabel, HeroStop, HeroZone } from "./scene.js";

/** The page's one element for a selector, of the type the code expects. */
function $<T extends HTMLElement>(selector: string, type: new () => T): T {
  const el = document.querySelector(selector);
  if (!(el instanceof type)) throw new Error(`landing: ${selector} is missing from the page`);
  return el;
}

// ── request form: first, so nothing below can keep its submit guard from attaching ──
// landing.html's inline guard covers the page before this runs, and without JavaScript.
const form = $("#f", HTMLFormElement);
form.addEventListener("submit", (event) => event.preventDefault());
let picker: SitePicker | null = null;
try {
  picker = startSitePicker();
} catch (error) {
  // The map is a convenience: the acres field still says how big the site is.
  console.warn("landing: the site picker could not start", error);
  $("#pk-hint", HTMLElement).textContent = "The map could not load. Enter the acres instead.";
}
const status = $("#f-ok", HTMLElement);
const fields: Record<Exclude<RequestField, "area">, HTMLInputElement> = {
  name: $("#f-name", HTMLInputElement),
  email: $("#f-email", HTMLInputElement),
  acres: $("#f-acres", HTMLInputElement),
};
const errorOf: Record<RequestField, HTMLElement> = {
  name: $("#f-name-err", HTMLElement),
  email: $("#f-email-err", HTMLElement),
  acres: $("#f-acres-err", HTMLElement),
  area: $("#pk-err", HTMLElement),
};

function showErrors(errors: RequestErrors) {
  for (const key of Object.keys(errorOf) as RequestField[]) {
    const message = errors[key];
    const out = errorOf[key];
    out.textContent = message ?? "";
    out.hidden = !message;
    if (key === "area") continue;
    const input = fields[key];
    if (message) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", out.id);
    } else {
      input.removeAttribute("aria-invalid");
      input.removeAttribute("aria-describedby");
    }
  }
}

form.addEventListener("submit", () => {
  const errors = validateRequest({
    name: fields.name.value,
    organization: $("#f-org", HTMLInputElement).value,
    email: fields.email.value,
    acres: fields.acres.value,
    message: $("#f-msg", HTMLTextAreaElement).value,
    areaCount: picker?.count() ?? 0,
  });
  showErrors(errors);
  const first = (["name", "email", "acres"] as const).find((k) => errors[k]);
  if (first) {
    status.hidden = true;
    fields[first].focus();
    return;
  }
  if (errors.area) {
    status.hidden = true;
    return;
  }
  if (REQUEST_ENDPOINT === null) {
    status.textContent =
      `Your request is complete (${picker?.count() ? picker.summary() : `${fields.acres.value} acres`}), ` +
      "but this form is not connected yet, so nothing was sent.";
    status.hidden = false;
  }
});

// ── brand ──
for (const el of document.querySelectorAll<HTMLElement>("[data-brand]")) el.textContent = BRAND;
document.title = `${BRAND} · Robots for everything outside`;

// ── theme: light by default, dark by the system, either by the toggle ──
const dark = matchMedia("(prefers-color-scheme: dark)");
const toggle = $("#tgl", HTMLButtonElement);
const isDark = () => {
  const pick = document.documentElement.dataset.theme;
  return pick ? pick === "dark" : dark.matches;
};
const labelToggle = () =>
  toggle.setAttribute("aria-label", isDark() ? "Switch to light mode" : "Switch to dark mode");
labelToggle();
toggle.addEventListener("click", () => {
  document.documentElement.dataset.theme = isDark() ? "light" : "dark";
  labelToggle();
});
dark.addEventListener("change", labelToggle);

// ── hero: nine crews, a camera touring them, and the examples strip under it ──
const still = matchMedia("(prefers-reduced-motion: reduce)");
const ZONES: HeroZone[] = [
  { id: "nature", c: [-238, 172], phase: 0.15 },
  { id: "range", c: [-175, 158], phase: 0.55 },
  { id: "forest", c: [-112, 162], phase: 0.3 },
  { id: "infra", c: [-48, 172], phase: 0.7, lz: 12 },
  { id: "solar", c: [8, 170], phase: 0.5, lz: 5, wide: 30, clear: 16 },
  { id: "pipeline", c: [55, 195], phase: 0.35, wide: 32, clear: 14 },
  { id: "fuels", c: [-62, 238], phase: 0.45 },
  { id: "agri", c: [-132, 236], phase: 0.6 },
  { id: "reclaim", c: [-200, 232], phase: 0.25, lz: 8 },
];
const ROUTE: HeroStop[] = [
  { c: [-238, 172] },
  { c: [-175, 158] },
  { c: [-112, 162] },
  { c: [-48, 172], zoom: 1.25 },
  { c: [8, 170], zoom: 1.1, az: 32, el: 36 },
  { c: [55, 195], zoom: 1.1 },
  { c: [24, 236], zoom: 1.2 },
  { c: [-62, 238], zoom: 1.15 },
  { c: [-132, 236], az: 22, el: 40 },
  { c: [-200, 232], zoom: 1.1 },
];
/** Each example's work cell and its stop on the route (the route has one stop between). */
const EXAMPLES = ZONES.map((z, i) => ({ id: z.id, c: z.c, ri: i < 6 ? i : i + 1 }));

const heroEl = $("#top", HTMLElement);
const heroCanvas = $("canvas.gl", HTMLCanvasElement);
const zoneLabels = [...document.querySelectorAll<HTMLAnchorElement>("#zl a")];
const exButtons = [...document.querySelectorAll<HTMLButtonElement>("#ex button")];
const exList = $("#ex", HTMLElement);
const exNow = $("#ex-now", HTMLElement);
let hero: Hero | null = null;
let exCurrent: string | null = null;
let exLock: { id: string; until: number } | null = null;

/** Highlights the example in view, and names it on narrow screens. */
function markExample(id: string) {
  if (id === exCurrent) return;
  exCurrent = id;
  const k = exButtons.findIndex((b) => b.dataset.z === id);
  const button = exButtons[k];
  if (button) {
    const n = document.createElement("em");
    n.textContent = String(k + 1).padStart(2, "0");
    const of = document.createElement("span");
    of.textContent = `${k + 1} of ${exButtons.length}`;
    exNow.replaceChildren(n, button.querySelector("span")?.textContent ?? "", " ", of);
  }
  for (const b of exButtons) b.setAttribute("aria-current", b.dataset.z === id ? "true" : "false");
  const li = button?.parentElement;
  if (li && exList.scrollWidth > exList.clientWidth) {
    exList.scrollTo({ left: li.offsetLeft - (exList.clientWidth - li.offsetWidth) / 2 });
  }
}

for (const b of exButtons) {
  b.addEventListener("click", () => {
    const ex = EXAMPLES.find((e) => e.id === b.dataset.z);
    if (!ex) return;
    markExample(ex.id);
    if (hero) {
      exLock = { id: ex.id, until: performance.now() + 10_000 };
      hero.goTo(ex.ri);
    } else {
      // No scene to fly: the example's own card further down the page.
      document.getElementById(`use-${ex.id}`)?.scrollIntoView();
    }
  });
}

/** Places the crews' labels over the scene, faded out near the edges and behind the copy. */
function placeLabels(list: HeroLabel[]) {
  for (const it of list) {
    const el = zoneLabels.find((a) => a.dataset.z === it.id);
    if (!el) continue;
    const [x, y] = it.xy;
    const { W, H } = it;
    const inside = x > 40 && x < W - 40 && y > 90 && y < H - 60;
    const overCopy = W >= 980 && x < Math.min(640, W * 0.45) && y > H * 0.42;
    const cx = Math.abs(x - W * 0.55) / (W * 0.55);
    const a = inside && !overCopy ? Math.max(0, Math.min(1, 1.6 - cx * 1.4)) : 0;
    el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
    el.style.opacity = a.toFixed(2);
    el.style.pointerEvents = a > 0.5 ? "auto" : "none";
  }
}

/** Follows the camera: the example nearest its focus is the one in view. */
function followCamera(h: Hero) {
  const f = h.focus();
  const fx = f[0] ?? 0;
  const fy = f[1] ?? 0;
  if (exLock) {
    const lock = exLock;
    const ex = EXAMPLES.find((e) => e.id === lock.id);
    if (ex && Math.hypot(ex.c[0] - fx, ex.c[1] - fy) > 30 && performance.now() < lock.until) return;
    exLock = null;
  }
  let best: string | null = null;
  let bestDistance = Infinity;
  for (const ex of EXAMPLES) {
    const d = Math.hypot(ex.c[0] - fx, ex.c[1] - fy);
    if (d < bestDistance) {
      bestDistance = d;
      best = ex.id;
    }
  }
  if (best && bestDistance < 45) markExample(best);
}

function heroFallback(reason: unknown) {
  console.warn("landing: the hero scene could not start; showing a still", reason);
  heroEl.dataset.fallback = "";
  heroCanvas.hidden = true;
  heroEl.dataset.scene = "fallback";
}

async function startScenes() {
  let scene: typeof SceneModule;
  try {
    scene = await import("./scene.js");
    hero = scene.startHeroScene({
      canvas: heroCanvas,
      startAtPoint: 3,
      themeEl: $(".stage", HTMLElement),
      zones: ZONES,
      route: ROUTE,
      still,
      maxPixels: () => (innerWidth < 700 ? 1.6e6 : 4.4e6),
      width: () => (innerWidth < 700 ? 30 : innerWidth < 1100 ? 40 : 46),
      shift: (W) => (W >= 980 ? { x: 0.2, y: -0.12 } : { y: 0.2 }),
      labels: placeLabels,
    });
  } catch (error) {
    heroFallback(error);
    return;
  }
  heroCanvas.classList.add("on");
  heroEl.dataset.scene = "live";
  const live = hero;
  setInterval(() => followCamera(live), 250);

  // Fleet cards: their turntables start once the fleet section comes near the viewport.
  const fleet = $(".fleet", HTMLElement);
  const near = new IntersectionObserver(
    (entries, observer) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      observer.disconnect();
      try {
        scene.startFleetCards({
          cards: [...fleet.querySelectorAll<HTMLCanvasElement>("canvas[data-unit]")].map(
            (canvas) => ({ canvas, unit: canvas.dataset.unit ?? "" }),
          ),
          themeEl: $("#fleet-theme", HTMLElement),
          still,
          maxPixels: () => 0.6e6,
        });
      } catch (error) {
        // The cards keep their concept renders.
        console.warn("landing: the fleet cards could not start", error);
      }
    },
    { rootMargin: "400px 0px" },
  );
  near.observe(fleet);
}

// As the design does: after the first frame, so the page paints before the scene builds.
requestAnimationFrame(() => setTimeout(() => void startScenes(), 30));
